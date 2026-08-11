"""Two-pass layer selection and paper-faithful ridge fitting."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass

import torch

from kvbridge.cache import KVCache
from kvbridge.config import FitConfig, ModelSignature
from kvbridge.features import flatten_head_tokens, flatten_tokens, selected_layer_features
from kvbridge.mapper import CrossModelKVMapper
from kvbridge.ridge import RidgeAccumulator


@dataclass(frozen=True, slots=True)
class CalibrationPair:
    """Aligned source and target caches for the same token sequence."""

    source: KVCache
    target: KVCache


CalibrationSource = Sequence[CalibrationPair] | Callable[[], Iterable[CalibrationPair]]


def _iterate(examples: CalibrationSource) -> Iterator[CalibrationPair]:
    return iter(examples() if callable(examples) else examples)


def _validate_example(
    example: CalibrationPair,
    index: int,
    source_signature: ModelSignature,
    target_signature: ModelSignature,
) -> None:
    source_shape, target_shape = example.source.shape, example.target.shape
    if source_shape[0] != source_signature.num_layers:
        raise ValueError(f"source calibration pair {index} has the wrong layer count")
    if (
        source_shape[2] != source_signature.num_kv_heads
        or source_shape[4] != source_signature.head_dim
    ):
        raise ValueError(f"source calibration pair {index} does not match its signature")
    if target_shape[0] != target_signature.num_layers:
        raise ValueError(f"target calibration pair {index} has the wrong layer count")
    if (
        target_shape[2] != target_signature.num_kv_heads
        or target_shape[4] != target_signature.head_dim
    ):
        raise ValueError(f"target calibration pair {index} does not match its signature")
    if source_shape[1:2] + source_shape[3:4] != target_shape[1:2] + target_shape[3:4]:
        raise ValueError(f"calibration pair {index} does not align batch/token axes")


def _content_pair(pair: CalibrationPair, enabled: bool) -> CalibrationPair:
    if not enabled:
        return pair
    return CalibrationPair(pair.source.to_content_space(), pair.target.to_content_space())


def _select_layers(
    examples: CalibrationSource,
    source: ModelSignature,
    target: ModelSignature,
    config: FitConfig,
    dtype: torch.dtype,
) -> tuple[list[list[int]], list[list[float]]]:
    """Select source layers by key/value, head-averaged single-source R²."""
    selected: list[list[int]] = [[] for _ in range(target.num_layers)]
    all_scores: list[list[float]] = [[] for _ in range(target.num_layers)]
    block_size = config.selection_target_layer_block_size
    for block_start in range(0, target.num_layers, block_size):
        block = range(block_start, min(block_start + block_size, target.num_layers))
        accumulators: dict[tuple[int, int, int, str], RidgeAccumulator] = {}
        for target_layer in block:
            for source_layer in range(source.num_layers):
                for head in range(target.num_kv_heads):
                    for kind in ("key", "value"):
                        accumulators[(target_layer, source_layer, head, kind)] = RidgeAccumulator(
                            source.head_dim, target.head_dim, dtype=dtype
                        )
        pair_count = 0
        for pair_index, raw_pair in enumerate(_iterate(examples)):
            _validate_example(raw_pair, pair_index, source, target)
            pair = _content_pair(raw_pair, config.content_space)
            pair_count += 1
            for target_layer in block:
                for source_layer in range(source.num_layers):
                    for head in range(target.num_kv_heads):
                        accumulators[(target_layer, source_layer, head, "key")].update(
                            flatten_head_tokens(pair.source.keys[source_layer], head),
                            flatten_head_tokens(pair.target.keys[target_layer], head),
                        )
                        accumulators[(target_layer, source_layer, head, "value")].update(
                            flatten_head_tokens(pair.source.values[source_layer], head),
                            flatten_head_tokens(pair.target.values[target_layer], head),
                        )
        if pair_count == 0:
            raise ValueError("at least one calibration pair is required")
        for target_layer in block:
            layer_scores: list[float] = []
            for source_layer in range(source.num_layers):
                scores = [
                    accumulators[(target_layer, source_layer, head, kind)]
                    .solve(config.selection_alpha)
                    .r2
                    for head in range(target.num_kv_heads)
                    for kind in ("key", "value")
                ]
                layer_scores.append(sum(scores) / len(scores))
            ranked = sorted(range(source.num_layers), key=layer_scores.__getitem__, reverse=True)
            selected[target_layer] = ranked[: min(config.top_k, source.num_layers)]
            all_scores[target_layer] = layer_scores
    return selected, all_scores


def fit_mapper(
    examples: CalibrationSource,
    source_signature: ModelSignature,
    target_signature: ModelSignature,
    config: FitConfig | None = None,
) -> CrossModelKVMapper:
    """Fit a cross-model mapper from aligned cache pairs.

    The function deliberately accepts a re-iterable sequence because the paper's
    method is two-pass: select source layers, then fit the final multi-layer map.
    """
    config = config or FitConfig()
    source_signature.validate_pair(target_signature, require_matched_kv=config.require_matched_kv)
    if not config.require_matched_kv:
        raise NotImplementedError(
            "unmatched-KV selection is intentionally outside the validated v0.1 path"
        )
    dtype = torch.float64 if config.accumulation_dtype == "float64" else torch.float32
    selected, selection_scores = _select_layers(
        examples, source_signature, target_signature, config, dtype
    )

    layers = target_signature.num_layers
    key_weights: list[torch.Tensor | None] = [None] * layers
    value_weights: list[torch.Tensor | None] = [None] * layers
    key_biases: list[torch.Tensor | None] = [None] * layers
    value_biases: list[torch.Tensor | None] = [None] * layers
    key_r2: list[float] = [0.0] * layers
    value_r2: list[float] = [0.0] * layers
    block_size = config.target_layer_block_size
    for block_start in range(0, layers, block_size):
        block = range(block_start, min(block_start + block_size, layers))
        key_accumulators: dict[int, RidgeAccumulator] = {}
        value_accumulators: dict[int, RidgeAccumulator] = {}
        for target_layer in block:
            feature_count = (
                len(selected[target_layer])
                * source_signature.num_kv_heads
                * source_signature.head_dim
            )
            output_count = target_signature.num_kv_heads * target_signature.head_dim
            key_accumulators[target_layer] = RidgeAccumulator(
                feature_count, output_count, dtype=dtype
            )
            value_accumulators[target_layer] = RidgeAccumulator(
                feature_count, output_count, dtype=dtype
            )
        second_pass_count = 0
        for pair_index, raw_pair in enumerate(_iterate(examples)):
            _validate_example(raw_pair, pair_index, source_signature, target_signature)
            pair = _content_pair(raw_pair, config.content_space)
            second_pass_count += 1
            for target_layer in block:
                source_layers = selected[target_layer]
                key_accumulators[target_layer].update(
                    selected_layer_features(pair.source.keys, source_layers),
                    flatten_tokens(pair.target.keys[target_layer]),
                )
                value_accumulators[target_layer].update(
                    selected_layer_features(pair.source.values, source_layers),
                    flatten_tokens(pair.target.values[target_layer]),
                )
        if second_pass_count == 0:
            raise ValueError(
                "calibration source was empty on a later pass; pass a sequence or factory"
            )
        for target_layer in block:
            source_layers = selected[target_layer]
            feature_count = (
                len(source_layers) * source_signature.num_kv_heads * source_signature.head_dim
            )
            key_solution = key_accumulators[target_layer].solve(config.ridge_alpha)
            value_solution = value_accumulators[target_layer].solve(config.ridge_alpha)
            # [features, heads*dim] -> [heads, features, dim]
            key_weights[target_layer] = (
                key_solution.weight.reshape(
                    feature_count, target_signature.num_kv_heads, target_signature.head_dim
                )
                .permute(1, 0, 2)
                .float()
                .contiguous()
            )
            value_weights[target_layer] = (
                value_solution.weight.reshape(
                    feature_count, target_signature.num_kv_heads, target_signature.head_dim
                )
                .permute(1, 0, 2)
                .float()
                .contiguous()
            )
            key_biases[target_layer] = (
                key_solution.bias.reshape(target_signature.num_kv_heads, target_signature.head_dim)
                .float()
                .contiguous()
            )
            value_biases[target_layer] = (
                value_solution.bias.reshape(
                    target_signature.num_kv_heads, target_signature.head_dim
                )
                .float()
                .contiguous()
            )
            key_r2[target_layer] = key_solution.r2
            value_r2[target_layer] = value_solution.r2

    if any(item is None for item in (*key_weights, *value_weights, *key_biases, *value_biases)):
        raise RuntimeError("internal error: incomplete target-layer fit")

    return CrossModelKVMapper(
        source_signature=source_signature,
        target_signature=target_signature,
        config=config,
        selected_layers=selected,
        key_weights=[item for item in key_weights if item is not None],
        value_weights=[item for item in value_weights if item is not None],
        key_biases=[item for item in key_biases if item is not None],
        value_biases=[item for item in value_biases if item is not None],
        selection_scores=selection_scores,
        fit_key_r2=key_r2,
        fit_value_r2=value_r2,
    )
