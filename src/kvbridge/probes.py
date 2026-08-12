"""Concrete fail-closed runtime quality probes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from torch import Tensor

from kvbridge.cache import KVCache
from kvbridge.metrics import logit_kl_divergence


@dataclass(frozen=True, slots=True)
class QualityProbeResult:
    name: str
    accepted: bool
    value: float
    threshold: float
    metadata: dict[str, object] = field(default_factory=dict)

    def event_fields(self) -> dict[str, object]:
        return {
            "quality_probe": self.name,
            "quality_accepted": self.accepted,
            "quality_value": self.value,
            "quality_threshold": self.threshold,
        }


@dataclass(frozen=True, slots=True)
class LogitKLPolicy:
    max_kl: float
    temperature: float = 1.0
    max_suffix_tokens: int = 8
    symmetric: bool = False

    def __post_init__(self) -> None:
        if self.max_kl < 0:
            raise ValueError("max_kl cannot be negative")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.max_suffix_tokens <= 0:
            raise ValueError("max_suffix_tokens must be positive")

    def evaluate(self, candidate_logits: Tensor, reference_logits: Tensor) -> QualityProbeResult:
        value = logit_kl_divergence(
            candidate_logits,
            reference_logits,
            temperature=self.temperature,
            max_suffix_tokens=self.max_suffix_tokens,
            symmetric=self.symmetric,
        )
        return QualityProbeResult(
            name="short_suffix_logit_kl",
            accepted=value <= self.max_kl,
            value=value,
            threshold=self.max_kl,
            metadata={
                "max_suffix_tokens": self.max_suffix_tokens,
                "temperature": self.temperature,
                "symmetric": self.symmetric,
            },
        )


class ShadowLogitKLProbe:
    """Compare mapped-cache logits with a full-prefill shadow on sampled traffic."""

    def __init__(
        self,
        *,
        candidate_logits: Callable[[KVCache], Tensor],
        reference_logits: Callable[[], Tensor],
        policy: LogitKLPolicy,
    ) -> None:
        self.candidate_logits = candidate_logits
        self.reference_logits = reference_logits
        self.policy = policy

    def __call__(self, mapped_cache: KVCache) -> QualityProbeResult:
        return self.policy.evaluate(
            self.candidate_logits(mapped_cache),
            self.reference_logits(),
        )
