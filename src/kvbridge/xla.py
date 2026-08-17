"""Lazy PyTorch/XLA SPMD helpers for large-model cache capture.

This module intentionally imports ``torch_xla`` only inside functions.  The
core package and CPU/CUDA test suite therefore remain installable on ordinary
hosts while TPU jobs use the PyTorch/XLA version supplied by their runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from kvbridge.cache import KVCache, RotaryFactors


@dataclass(frozen=True, slots=True)
class XLAContext:
    device: torch.device
    mesh: Any
    runtime_devices: int


def logical_to_host(tensor: Tensor, logical_batch: int) -> Tensor:
    """Transfer a fixed-shape XLA result before trimming padded rows.

    Slicing on XLA changes the output graph for a partial final batch. Large
    models can then pay another full compilation for every cache output. Keep
    the accelerator-side shape stable and perform the inexpensive trim on CPU.
    """
    host = tensor.detach().to("cpu")
    if logical_batch == host.shape[0]:
        return host
    return host[:logical_batch].clone()


def initialize_xla(
    *, min_devices: int = 8, compilation_cache: str | Path | None = None
) -> XLAContext:
    """Enable single-process SPMD and build an FSDP-style parameter mesh."""
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


def _largest_axis_partition_spec(tensor: Tensor) -> tuple[str | None, ...]:
    """Shard a tensor on its largest axis, matching XLA FSDPv2's policy."""
    if tensor.ndim == 0:
        return ()
    largest_axis = max(range(tensor.ndim), key=lambda axis: tensor.shape[axis])
    return tuple(
        "fsdp" if axis == largest_axis else None for axis in range(tensor.ndim)
    )


def shard_model_for_inference(model: Any, context: XLAContext) -> Any:
    """Move a model to XLA and apply stable FSDP-style SPMD parameter sharding.

    PyTorch/XLA 2.8's experimental FSDPv2 module wrapper can propagate an
    incomplete ``XLAShardedTensor`` through Hugging Face cache outputs and crash
    the PJRT worker during inference.  Inference does not need FSDP's backward
    hooks, so this follows the underlying documented SPMD formulation directly:
    parameters are sharded on their largest axis and input batches on axis zero.
    """
    try:
        import torch_xla.distributed.spmd as xs  # type: ignore[import-not-found, unused-ignore]
    except ImportError as error:
        raise RuntimeError("torch_xla is unavailable") from error

    model = model.to(context.device)
    for parameter in model.parameters():
        xs.mark_sharding(
            parameter,
            context.mesh,
            _largest_axis_partition_spec(parameter),
        )
    return model


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
        "model_sharding_strategy": "spmd_largest_axis",
    }
