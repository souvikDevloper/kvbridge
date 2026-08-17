"""Two-pass layer selection and paper-faithful ridge fitting."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from kvbridge.cache import KVCache
from kvbridge.checkpoints import FitCheckpointStore, LayerCheckpoint
from kvbridge.config import FitConfig, ModelSignature
from kvbridge.features import flatten_tokens, selected_layer_features
from kvbridge.mapper import CrossModelKVMapper
from kvbridge.ridge import RidgeAccumulator
from kvbridge.selection import BatchedSelectionAccumulator


@dataclass(frozen=True, slots=True)
class CalibrationPair:
    """Aligned source and target caches for the same token sequence."""

    source: KVCache
    target: KVCache
    sampled_stride: int = 1

    def __post_init__(self) -> None:
        if self.sampled_stride <= 0:
            raise ValueError("calibration pair sampled_stride must be positive")


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


def _prepare_pair(pair: CalibrationPair, config: FitConfig) -> CalibrationPair:
    """Normalize a pair while keeping the full cache on its storage device.

    Accumulators copy only the source/target slices needed for their current
    block.  This matters at paper scale: moving an entire 14B/32B cache pair to
    the fitting device for every target layer would dominate both HBM and I/O.
    Accelerator capture jobs should strip RoPE before moving sampled caches to
    host memory, so this fallback conversion is normally a no-op there.
    """
    if config.content_space:
        source = (
            pair.source
            if pair.source.keys_are_content
            else pair.source.to_content_space()
        )
        target = (
            pair.target
            if pair.target.keys_are_content
            else pair.target.to_content_space()
        )
        pair = CalibrationPair(source, target, pair.sampled_stride)
    if config.token_stride % pair.sampled_stride:
        raise ValueError("pre-sampled calibration stride does not divide the configured stride")
    remaining_stride = config.token_stride // pair.sampled_stride
    return CalibrationPair(
        pair.source.sample_tokens(remaining_stride),
        pair.target.sample_tokens(remaining_stride),
        config.token_stride,
    )


def _stack_selection_layers(layers: Sequence[torch.Tensor]) -> torch.Tensor:
    """Convert cache layers to ``[layer, head, observation, dim]``."""
    stacked = torch.stack(list(layers), dim=0)
    layer_count, batch, heads, tokens, dimension = stacked.shape
    return (
        stacked.permute(0, 2, 1, 3, 4)
        .reshape(layer_count, heads, batch * tokens, dimension)
        .contiguous()
    )


def _select_layers(
    examples: CalibrationSource,
    source: ModelSignature,
    target: ModelSignature,
    config: FitConfig,
    dtype: torch.dtype,
    device: torch.device,
    checkpoints: FitCheckpointStore | None = None,
) -> tuple[list[list[int]], list[list[float]]]:
    """Select source layers by key/value, head-averaged single-source R²."""
    selected: list[list[int]] = [[] for _ in range(target.num_layers)]
    all_scores: list[list[float]] = [[] for _ in range(target.num_layers)]
    block_size = config.selection_target_layer_block_size
    for block_start in range(0, target.num_layers, block_size):
        block = list(range(block_start, min(block_start + block_size, target.num_layers)))
        if checkpoints is not None:
            restored = checkpoints.load_selection(block_start, block[-1] + 1)
            if restored is not None:
                restored_selected, restored_scores = restored
                for offset, target_layer in enumerate(block):
                    if len(restored_selected[offset]) != min(config.top_k, source.num_layers):
                        raise ValueError("selection checkpoint top-k differs from fit config")
                    if len(restored_scores[offset]) != source.num_layers:
                        raise ValueError("selection checkpoint source-layer count is invalid")
                    selected[target_layer] = restored_selected[offset]
                    all_scores[target_layer] = restored_scores[offset]
                continue
        key_accumulator = BatchedSelectionAccumulator(
            source.num_layers,
            len(block),
            target.num_kv_heads,
            source.head_dim,
            target.head_dim,
            dtype=dtype,
            device=device,
        )
        value_accumulator = BatchedSelectionAccumulator(
            source.num_layers,
            len(block),
            target.num_kv_heads,
            source.head_dim,
            target.head_dim,
            dtype=dtype,
            device=device,
        )
        pair_count = 0
        for pair_index, raw_pair in enumerate(_iterate(examples)):
            _validate_example(raw_pair, pair_index, source, target)
            pair = _prepare_pair(raw_pair, config)
            pair_count += 1
            key_accumulator.update(
                _stack_selection_layers(pair.source.keys),
                _stack_selection_layers([pair.target.keys[layer] for layer in block]),
            )
            value_accumulator.update(
                _stack_selection_layers(pair.source.values),
                _stack_selection_layers([pair.target.values[layer] for layer in block]),
            )
        if pair_count == 0:
            raise ValueError("at least one calibration pair is required")
        combined_scores = (
            key_accumulator.solve(config.selection_alpha).r2
            + value_accumulator.solve(config.selection_alpha).r2
        ).mean(dim=-1) / 2
        for offset, target_layer in enumerate(block):
            layer_scores = combined_scores[offset].detach().float().cpu().tolist()
            ranked = sorted(range(source.num_layers), key=layer_scores.__getitem__, reverse=True)
            selected[target_layer] = ranked[: min(config.top_k, source.num_layers)]
            all_scores[target_layer] = layer_scores
        if checkpoints is not None:
            checkpoints.save_selection(
                block_start,
                block[-1] + 1,
                [selected[layer] for layer in block],
                [all_scores[layer] for layer in block],
            )
    return selected, all_scores


def fit_mapper(
    examples: CalibrationSource,
    source_signature: ModelSignature,
    target_signature: ModelSignature,
    config: FitConfig | None = None,
    *,
    checkpoint_dir: str | Path | None = None,
    resume: bool = True,
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
    device = torch.device(config.accumulation_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA accumulation requested by FitConfig but CUDA is unavailable")
    checkpoints = (
        FitCheckpointStore(
            checkpoint_dir,
            source_signature,
            target_signature,
            config,
            resume=resume,
        )
        if checkpoint_dir is not None
        else None
    )
    selected, selection_scores = _select_layers(
        examples,
        source_signature,
        target_signature,
        config,
        dtype,
        device,
        checkpoints,
    )

    layers = target_signature.num_layers
    key_weights: list[torch.Tensor | None] = [None] * layers
    value_weights: list[torch.Tensor | None] = [None] * layers
    key_biases: list[torch.Tensor | None] = [None] * layers
    value_biases: list[torch.Tensor | None] = [None] * layers
    key_r2: list[float] = [0.0] * layers
    value_r2: list[float] = [0.0] * layers
    if checkpoints is not None:
        for target_layer in range(layers):
            feature_count = (
                len(selected[target_layer])
                * source_signature.num_kv_heads
                * source_signature.head_dim
            )
            restored = checkpoints.load_layer(
                target_layer,
                selected[target_layer],
                weight_shape=(
                    target_signature.num_kv_heads,
                    feature_count,
                    target_signature.head_dim,
                ),
                bias_shape=(target_signature.num_kv_heads, target_signature.head_dim),
            )
            if restored is not None:
                key_weights[target_layer] = restored.key_weight
                value_weights[target_layer] = restored.value_weight
                key_biases[target_layer] = restored.key_bias
                value_biases[target_layer] = restored.value_bias
                key_r2[target_layer] = restored.key_r2
                value_r2[target_layer] = restored.value_r2
    block_size = config.target_layer_block_size
    for block_start in range(0, layers, block_size):
        block = [
            layer
            for layer in range(block_start, min(block_start + block_size, layers))
            if key_weights[layer] is None
        ]
        if not block:
            continue
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
                feature_count, output_count, dtype=dtype, device=device
            )
            value_accumulators[target_layer] = RidgeAccumulator(
                feature_count, output_count, dtype=dtype, device=device
            )
        second_pass_count = 0
        for pair_index, raw_pair in enumerate(_iterate(examples)):
            _validate_example(raw_pair, pair_index, source_signature, target_signature)
            pair = _prepare_pair(raw_pair, config)
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
            key_weight = (
                key_solution.weight.reshape(
                    feature_count, target_signature.num_kv_heads, target_signature.head_dim
                )
                .permute(1, 0, 2)
                .float()
                .cpu()
                .contiguous()
            )
            value_weight = (
                value_solution.weight.reshape(
                    feature_count, target_signature.num_kv_heads, target_signature.head_dim
                )
                .permute(1, 0, 2)
                .float()
                .cpu()
                .contiguous()
            )
            key_bias = (
                key_solution.bias.reshape(target_signature.num_kv_heads, target_signature.head_dim)
                .float()
                .cpu()
                .contiguous()
            )
            value_bias = (
                value_solution.bias.reshape(
                    target_signature.num_kv_heads, target_signature.head_dim
                )
                .float()
                .cpu()
                .contiguous()
            )
            key_weights[target_layer] = key_weight
            value_weights[target_layer] = value_weight
            key_biases[target_layer] = key_bias
            value_biases[target_layer] = value_bias
            key_r2[target_layer] = key_solution.r2
            value_r2[target_layer] = value_solution.r2
            if checkpoints is not None:
                checkpoints.save_layer(
                    target_layer,
                    source_layers,
                    LayerCheckpoint(
                        key_weight,
                        value_weight,
                        key_bias,
                        value_bias,
                        key_r2[target_layer],
                        value_r2[target_layer],
                    ),
                )

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
