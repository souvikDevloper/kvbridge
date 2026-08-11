"""Deterministic synthetic family used for smoke tests and the zero-download demo."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from kvbridge.cache import KVCache, RotaryFactors
from kvbridge.config import FitConfig, ModelSignature
from kvbridge.features import flatten_tokens
from kvbridge.fit import CalibrationPair, fit_mapper
from kvbridge.mapper import CrossModelKVMapper


@dataclass(frozen=True, slots=True)
class SyntheticProblem:
    source: ModelSignature
    target: ModelSignature
    calibration: list[CalibrationPair]
    evaluation: CalibrationPair
    true_layers: list[int]


def rotary_factors(
    tokens: int,
    dim: int,
    *,
    theta: float,
    batch: int = 1,
    interleaved: bool = False,
) -> RotaryFactors:
    positions = torch.arange(tokens, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    frequencies = torch.outer(positions, inv_freq)
    embedding = (
        frequencies.repeat_interleave(2, dim=-1)
        if interleaved
        else torch.cat((frequencies, frequencies), dim=-1)
    )
    cos = embedding.cos().unsqueeze(0).expand(batch, -1, -1).contiguous()
    sin = embedding.sin().unsqueeze(0).expand(batch, -1, -1).contiguous()
    return RotaryFactors(cos, sin, interleaved=interleaved)


def _make_pair(
    generator: torch.Generator,
    *,
    batch: int,
    tokens: int,
    source_layers: int,
    target_layers: int,
    heads: int,
    dim: int,
    true_layers: list[int],
    key_weight: list[torch.Tensor],
    value_weight: list[torch.Tensor],
    key_bias: list[torch.Tensor],
    value_bias: list[torch.Tensor],
    noise: float,
) -> CalibrationPair:
    source_keys_content = [
        torch.randn((batch, heads, tokens, dim), generator=generator) for _ in range(source_layers)
    ]
    source_values = [
        torch.randn((batch, heads, tokens, dim), generator=generator) for _ in range(source_layers)
    ]
    target_keys_content, target_values = [], []
    for layer in range(target_layers):
        x_key = flatten_tokens(source_keys_content[true_layers[layer]])
        x_value = flatten_tokens(source_values[true_layers[layer]])
        key = x_key @ key_weight[layer] + key_bias[layer]
        value = x_value @ value_weight[layer] + value_bias[layer]
        key += noise * torch.randn(key.shape, generator=generator)
        value += noise * torch.randn(value.shape, generator=generator)
        target_keys_content.append(key.reshape(batch, tokens, heads, dim).permute(0, 2, 1, 3))
        target_values.append(value.reshape(batch, tokens, heads, dim).permute(0, 2, 1, 3))
    source_rope = rotary_factors(tokens, dim, theta=10_000.0, batch=batch)
    target_rope = rotary_factors(tokens, dim, theta=50_000.0, batch=batch)
    source_cache = KVCache(source_keys_content, source_values, keys_are_content=True).apply_rotary(
        source_rope
    )
    target_cache = KVCache(target_keys_content, target_values, keys_are_content=True).apply_rotary(
        target_rope
    )
    return CalibrationPair(source_cache, target_cache)


def make_problem(
    *,
    seed: int = 7,
    calibration_pairs: int = 6,
    tokens: int = 24,
    noise: float = 1e-3,
    source_layers: int = 3,
    target_layers: int = 2,
    heads: int = 2,
    dim: int = 4,
) -> SyntheticProblem:
    generator = torch.Generator().manual_seed(seed)
    if source_layers < 2 or target_layers < 1 or heads < 1 or dim < 2 or dim % 2:
        raise ValueError(
            "synthetic dimensions require >=2 source layers and an even head dimension"
        )
    true_layers = [
        int((layer * source_layers / target_layers + source_layers - 1) % source_layers)
        for layer in range(target_layers)
    ]
    features, outputs = heads * dim, heads * dim
    key_weight = [
        torch.randn((features, outputs), generator=generator) / math.sqrt(features)
        for _ in range(target_layers)
    ]
    value_weight = [
        torch.randn((features, outputs), generator=generator) / math.sqrt(features)
        for _ in range(target_layers)
    ]
    key_bias = [torch.randn(outputs, generator=generator) * 0.1 for _ in range(target_layers)]
    value_bias = [torch.randn(outputs, generator=generator) * 0.1 for _ in range(target_layers)]
    pairs = [
        _make_pair(
            generator,
            batch=1,
            tokens=tokens,
            source_layers=source_layers,
            target_layers=target_layers,
            heads=heads,
            dim=dim,
            true_layers=true_layers,
            key_weight=key_weight,
            value_weight=value_weight,
            key_bias=key_bias,
            value_bias=value_bias,
            noise=noise,
        )
        for _ in range(calibration_pairs + 1)
    ]
    tokenizer_hash = "synthetic-shared-tokenizer-v1"
    source = ModelSignature(
        f"synthetic-{source_layers}l", "v1", tokenizer_hash, source_layers, heads, dim
    )
    target = ModelSignature(
        f"synthetic-{target_layers}l", "v1", tokenizer_hash, target_layers, heads, dim
    )
    return SyntheticProblem(source, target, pairs[:-1], pairs[-1], true_layers)


def fit_demo(problem: SyntheticProblem) -> CrossModelKVMapper:
    return fit_mapper(
        problem.calibration,
        problem.source,
        problem.target,
        FitConfig(top_k=1, ridge_alpha=0.01, content_space=True),
    )


def cache_r2(predicted: KVCache, expected: KVCache) -> float:
    pred = torch.cat([*predicted.keys, *predicted.values]).double().flatten()
    truth = torch.cat([*expected.keys, *expected.values]).double().flatten()
    residual = ((truth - pred) ** 2).sum()
    total = ((truth - truth.mean()) ** 2).sum()
    return float((1 - residual / total).item())
