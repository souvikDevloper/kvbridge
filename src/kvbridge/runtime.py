"""Guarded transfer orchestration with explicit fallback and telemetry hooks."""

from __future__ import annotations

import hashlib
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from kvbridge.cache import KVCache, RotaryFactors
from kvbridge.mapper import CrossModelKVMapper, TransferReport
from kvbridge.probes import QualityProbeResult

T = TypeVar("T")
TransferStatus = Literal["accepted", "fallback"]


@dataclass(frozen=True, slots=True)
class GuardPolicy:
    max_abs_value: float = 1e4
    max_transfer_ms: float | None = None
    max_batch_size: int | None = None
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.max_abs_value <= 0:
            raise ValueError("max_abs_value must be positive")
        if self.max_transfer_ms is not None and self.max_transfer_ms <= 0:
            raise ValueError("max_transfer_ms must be positive")
        if self.max_batch_size is not None and self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")


@dataclass(frozen=True, slots=True)
class GuardedResult(Generic[T]):
    value: T
    status: TransferStatus
    reason: str
    transfer_report: TransferReport | None
    quality_probe: QualityProbeResult | None
    total_elapsed_ms: float


@dataclass(frozen=True, slots=True)
class ShadowSamplingPolicy:
    """Deterministically select a bounded fraction of requests for a quality oracle."""

    rate: float = 0.0
    salt: str = "kvbridge-shadow-v1"

    def __post_init__(self) -> None:
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError("shadow sampling rate must be between zero and one")
        if not self.salt:
            raise ValueError("shadow sampling salt is required")

    def select(self, request_id: str | None) -> bool:
        if self.rate <= 0.0:
            return False
        if self.rate >= 1.0:
            return True
        if request_id is None:
            return random.SystemRandom().random() < self.rate
        digest = hashlib.sha256(f"{self.salt}:{request_id}".encode()).digest()
        bucket = int.from_bytes(digest[:8], "big") / 2**64
        return bucket < self.rate


class GuardedTransferEngine(Generic[T]):
    """Fail closed: malformed/slow/failed transfers visibly use re-prefill fallback."""

    def __init__(
        self,
        mapper: CrossModelKVMapper,
        *,
        policy: GuardPolicy | None = None,
        event_sink: Callable[[dict[str, object]], None] | None = None,
        shadow_sampling: ShadowSamplingPolicy | None = None,
    ) -> None:
        self.mapper = mapper
        self.policy = policy or GuardPolicy()
        self.event_sink = event_sink or (lambda _: None)
        # Preserve the pre-sampling fail-closed contract: callers that provide a
        # quality probe get that probe on every request unless they explicitly
        # opt into a lower sampling rate.
        self.shadow_sampling = shadow_sampling or ShadowSamplingPolicy(rate=1.0)

    def run(
        self,
        source: KVCache,
        *,
        target_rotary: RotaryFactors | None,
        accept: Callable[[KVCache], bool] | None,
        on_accept: Callable[[KVCache], T],
        fallback: Callable[[], T],
        quality_probe: Callable[[KVCache], QualityProbeResult] | None = None,
        request_id: str | None = None,
    ) -> GuardedResult[T]:
        started = time.perf_counter()
        transfer_report: TransferReport | None = None
        probe_result: QualityProbeResult | None = None
        status: TransferStatus
        reason = "transfer passed all configured gates"
        shadow_selected = quality_probe is not None and self.shadow_sampling.select(request_id)
        try:
            if (
                self.policy.max_batch_size is not None
                and source.shape[0] > self.policy.max_batch_size
            ):
                raise ValueError("source cache exceeds the configured batch-size bound")
            if self.policy.max_tokens is not None and source.shape[3] > self.policy.max_tokens:
                raise ValueError("source cache exceeds the configured token bound")
            mapped, transfer_report = self.mapper.transfer(source, target_rotary=target_rotary)
            finite = all(
                tensor.isfinite().all().item() for tensor in (*mapped.keys, *mapped.values)
            )
            bounded = all(
                math.isfinite(float(tensor.abs().max().item()))
                and float(tensor.abs().max().item()) <= self.policy.max_abs_value
                for tensor in (*mapped.keys, *mapped.values)
            )
            if not finite:
                raise ValueError("mapped cache contains NaN or infinity")
            if not bounded:
                raise ValueError("mapped cache exceeds the configured magnitude bound")
            if (
                self.policy.max_transfer_ms is not None
                and transfer_report.elapsed_ms > self.policy.max_transfer_ms
            ):
                raise TimeoutError("transfer exceeded its latency budget")
            if accept is not None and not accept(mapped):
                raise ValueError("application quality gate rejected the mapped cache")
            if shadow_selected and quality_probe is not None:
                probe_result = quality_probe(mapped)
                if not probe_result.accepted:
                    raise ValueError(
                        f"quality probe {probe_result.name} rejected mapped cache: "
                        f"{probe_result.value:.6g} exceeds {probe_result.threshold:.6g}"
                    )
            value, status = on_accept(mapped), "accepted"
        except (
            Exception
        ) as error:  # fallback boundary intentionally catches mapper/backend failures
            reason = f"{type(error).__name__}: {error}"
            value, status = fallback(), "fallback"
        elapsed = (time.perf_counter() - started) * 1000
        event: dict[str, object] = {
            "event": "kvbridge_handoff",
            "status": status,
            "reason": reason,
            "source_model": self.mapper.source_signature.model_id,
            "target_model": self.mapper.target_signature.model_id,
            "tokens": source.shape[3],
            "elapsed_ms": elapsed,
            "transfer_ms": transfer_report.elapsed_ms if transfer_report is not None else None,
            "shadow_selected": shadow_selected,
        }
        if probe_result is not None:
            event.update(probe_result.event_fields())
        self.event_sink(event)
        return GuardedResult(value, status, reason, transfer_report, probe_result, elapsed)
