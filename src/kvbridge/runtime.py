"""Guarded transfer orchestration with explicit fallback and telemetry hooks."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from kvbridge.cache import KVCache, RotaryFactors
from kvbridge.mapper import CrossModelKVMapper, TransferReport

T = TypeVar("T")
TransferStatus = Literal["accepted", "fallback"]


@dataclass(frozen=True, slots=True)
class GuardPolicy:
    max_abs_value: float = 1e4
    max_transfer_ms: float | None = None

    def __post_init__(self) -> None:
        if self.max_abs_value <= 0:
            raise ValueError("max_abs_value must be positive")
        if self.max_transfer_ms is not None and self.max_transfer_ms <= 0:
            raise ValueError("max_transfer_ms must be positive")


@dataclass(frozen=True, slots=True)
class GuardedResult(Generic[T]):
    value: T
    status: TransferStatus
    reason: str
    transfer_report: TransferReport | None
    total_elapsed_ms: float


class GuardedTransferEngine(Generic[T]):
    """Fail closed: malformed/slow/failed transfers visibly use re-prefill fallback."""

    def __init__(
        self,
        mapper: CrossModelKVMapper,
        *,
        policy: GuardPolicy | None = None,
        event_sink: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.mapper = mapper
        self.policy = policy or GuardPolicy()
        self.event_sink = event_sink or (lambda _: None)

    def run(
        self,
        source: KVCache,
        *,
        target_rotary: RotaryFactors | None,
        accept: Callable[[KVCache], bool] | None,
        on_accept: Callable[[KVCache], T],
        fallback: Callable[[], T],
    ) -> GuardedResult[T]:
        started = time.perf_counter()
        transfer_report: TransferReport | None = None
        reason = "transfer passed all configured gates"
        try:
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
            value, status = on_accept(mapped), "accepted"
        except (
            Exception
        ) as error:  # fallback boundary intentionally catches mapper/backend failures
            reason = f"{type(error).__name__}: {error}"
            value, status = fallback(), "fallback"
        elapsed = (time.perf_counter() - started) * 1000
        self.event_sink(
            {
                "event": "kvbridge_handoff",
                "status": status,
                "reason": reason,
                "source_model": self.mapper.source_signature.model_id,
                "target_model": self.mapper.target_signature.model_id,
                "tokens": source.shape[3],
                "elapsed_ms": elapsed,
            }
        )
        return GuardedResult(value, status, reason, transfer_report, elapsed)
