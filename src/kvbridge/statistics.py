"""Dependency-free uncertainty estimates for experiment reporting."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    low: float
    center: float
    high: float
    confidence: float
    resamples: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate(values: Sequence[float], confidence: float, resamples: int) -> list[float]:
    normalized = [float(value) for value in values]
    if not normalized:
        raise ValueError("bootstrap input cannot be empty")
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("bootstrap input must be finite")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    return normalized


def _quantile(ordered: Sequence[float], fraction: float) -> float:
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> ConfidenceInterval:
    """Return a deterministic percentile-bootstrap interval for a sample mean."""
    sample = _validate(values, confidence, resamples)
    generator = random.Random(seed)
    size = len(sample)
    means = sorted(
        sum(sample[generator.randrange(size)] for _ in range(size)) / size
        for _ in range(resamples)
    )
    tail = (1.0 - confidence) / 2.0
    return ConfidenceInterval(
        low=_quantile(means, tail),
        center=sum(sample) / size,
        high=_quantile(means, 1.0 - tail),
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )


def paired_bootstrap_difference(
    left: Sequence[float],
    right: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> ConfidenceInterval:
    """Estimate paired mean difference ``left - right`` with shared resamples."""
    if len(left) != len(right):
        raise ValueError("paired bootstrap inputs must have equal lengths")
    differences = [float(a) - float(b) for a, b in zip(left, right, strict=True)]
    return bootstrap_mean_interval(
        differences, confidence=confidence, resamples=resamples, seed=seed
    )
