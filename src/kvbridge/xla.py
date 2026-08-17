"""Lazy PyTorch/XLA SPMD helpers for large-model cache capture.

This module intentionally imports ``torch_xla`` only inside functions.  The
core package and CPU/CUDA test suite therefore remain installable on ordinary
hosts while TPU jobs use the PyTorch/XLA version supplied by their runtime.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from kvbridge.cache import KVCache, RotaryFactors
from kvbridge.errors import CompatibilityError


@dataclass(frozen=True, slots=True)
class XLAContext:
    device: torch.device
    mesh: Any
    runtime_devices: int


def initialize_xla(
    *, min_devices: int = 8, compilation_cache: str | Path | None = None
) -> XLAContext:
    """Enable single-process SPMD and build an FSDP mesh."""
    try:
        import torch_xla.distributed.spmd as xs  # type: ignore[import-not-found, unused-ignore]
        import torch_xla.runtime as xr  # type: ignore[import-not-found, unused-ignore]
    except ImportError as error:
        raise RuntimeError(
            "torch_xla is required; select a TPU runtime before using --execute"
        ) from error

    if compilation_cache is not None:
        cache_path = Path(compilation_cache)
        cache_path.mkdir(parents=True, exist_ok=True)
        xr.initialize_cache(str(cache_path))
    xr.use_spmd()
    runtime_devices = int(xr.global_runtime_device_count())
    if runtime_devices < min_devices:
        raise RuntimeError(
            f"the experiment requires at least {min_devices} XLA devices; found {runtime_devices}"
        )
    mesh = xs.Mesh(
        np.arange(runtime_devices),
        (runtime_devices, 1),
        ("fsdp", "model"),
    )
    xs.set_global_mesh(mesh)
    return XLAContext(torch.device("xla"), mesh, runtime_devices)


def _decoder_layers(model: Any) -> Any:
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "model", None), "layers", None),
    ]
    for candidate in candidates:
        if candidate is not None and len(candidate) > 0:
            return candidate
    raise CompatibilityError("model does not expose supported decoder layers for XLA FSDP")


def _cache_layers(cache: Any) -> list[tuple[Tensor, Tensor]]:
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    if isinstance(cache, tuple | list):
        return [(layer[0], layer[1]) for layer in cache]
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    return []


def wrap_model_for_fsdp(model: Any, context: XLAContext) -> Any:
    """Move a decoder-only model to XLA and shard parameters/activations."""
    try:
        import torch_xla.distributed.spmd as xs  # type: ignore[import-not-found, unused-ignore]
        from torch_xla.distributed.fsdp.wrap import (  # type: ignore[import-not-found, unused-ignore]
            transformer_auto_wrap_policy,
        )
        from torch_xla.experimental.spmd_fully_sharded_data_parallel import (  # type: ignore[import-not-found, unused-ignore]
            SpmdFullyShardedDataParallel as FSDPv2,
        )
    except ImportError as error:
        raise RuntimeError("this torch_xla build does not provide SPMD FSDPv2") from error

    decoder_type = type(_decoder_layers(model)[0])
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={decoder_type},
    )
    def shard_output(output: Any, mesh: Any) -> None:
        for name in ("last_hidden_state", "logits"):
            tensor = getattr(output, name, None)
            if isinstance(tensor, Tensor):
                xs.mark_sharding(tensor, mesh, ("fsdp", None, None))
        for key, value in _cache_layers(getattr(output, "past_key_values", None)):
            xs.mark_sharding(key, mesh, ("fsdp", None, None, None))
            xs.mark_sharding(value, mesh, ("fsdp", None, None, None))

    return FSDPv2(
        model,
        mesh=context.mesh,
        auto_wrap_policy=auto_wrap_policy,
        shard_output=shard_output,
    )


def shard_batch(tensor: Tensor, context: XLAContext) -> Tensor:
    """Place a rank-2 token/mask batch on XLA and shard its batch axis."""
    if tensor.ndim != 2:
        raise ValueError("TPU input tensors must be rank-2")
    if tensor.shape[0] % context.runtime_devices:
        raise ValueError("TPU batch size must be divisible by the runtime device count")
    try:
        import torch_xla.distributed.spmd as xs  # type: ignore[import-not-found, unused-ignore]
    except ImportError as error:  # pragma: no cover - guarded by initialize_xla
        raise RuntimeError("torch_xla is unavailable") from error
    sharded = xs.mark_sharding(
        tensor.to(context.device), context.mesh, ("fsdp", None)
    )
    return cast(Tensor, sharded)


def shard_cache(cache: KVCache, context: XLAContext) -> KVCache:
    """Move a cache to XLA and shard every batch axis across the FSDP mesh."""
    try:
        import torch_xla.distributed.spmd as xs  # type: ignore[import-not-found, unused-ignore]
    except ImportError as error:  # pragma: no cover - guarded by initialize_xla
        raise RuntimeError("torch_xla is unavailable") from error

    def sharded(tensor: Tensor) -> Tensor:
        value = xs.mark_sharding(
            tensor.to(context.device), context.mesh, ("fsdp", None, None, None)
        )
        return cast(Tensor, value)

    rotary = None
    if cache.rotary is not None:
        cos = xs.mark_sharding(
            cache.rotary.cos.to(context.device),
            context.mesh,
            ("fsdp", None, None),
        )
        sin = xs.mark_sharding(
            cache.rotary.sin.to(context.device),
            context.mesh,
            ("fsdp", None, None),
        )
        rotary = RotaryFactors(
            cast(Tensor, cos), cast(Tensor, sin), cache.rotary.interleaved
        )
    return KVCache(
        [sharded(tensor) for tensor in cache.keys],
        [sharded(tensor) for tensor in cache.values],
        rotary,
        cache.keys_are_content,
    )


def sync_xla() -> None:
    """Materialize pending lazy XLA operations."""
    try:
        import torch_xla  # type: ignore[import-not-found, unused-ignore]
    except ImportError as error:  # pragma: no cover - guarded by initialize_xla
        raise RuntimeError("torch_xla is unavailable") from error
    torch_xla.sync()


def xla_runtime_manifest(context: XLAContext) -> dict[str, Any]:
    try:
        import torch_xla  # type: ignore[import-not-found, unused-ignore]
        import torch_xla.runtime as xr  # type: ignore[import-not-found, unused-ignore]
    except ImportError as error:  # pragma: no cover - guarded by initialize_xla
        raise RuntimeError("torch_xla is unavailable") from error
    device_type = getattr(xr, "device_type", lambda: "unknown")()
    addressable = getattr(
        xr, "addressable_runtime_device_count", lambda: context.runtime_devices
    )()
    return {
        "torch_xla": getattr(torch_xla, "__version__", "unknown"),
        "device_type": str(device_type),
        "global_runtime_devices": context.runtime_devices,
        "addressable_runtime_devices": int(addressable),
        "spmd": bool(xr.is_spmd()),
    }
