"""Safe, out-of-core calibration shard persistence."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from kvbridge.cache import KVCache, RotaryFactors
from kvbridge.errors import ArtifactError
from kvbridge.fit import CalibrationPair


def _cache_tensors(prefix: str, cache: KVCache) -> dict[str, torch.Tensor]:
    tensors = {
        f"{prefix}.keys": torch.stack(cache.keys).detach().cpu().contiguous(),
        f"{prefix}.values": torch.stack(cache.values).detach().cpu().contiguous(),
    }
    if cache.rotary is not None:
        tensors[f"{prefix}.rope_cos"] = cache.rotary.cos.detach().cpu().contiguous()
        tensors[f"{prefix}.rope_sin"] = cache.rotary.sin.detach().cpu().contiguous()
    return tensors


def save_calibration_shard(
    path: str | Path,
    pair: CalibrationPair,
    *,
    sequence_id: str,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ArtifactError(f"refusing to overwrite calibration shard: {destination}")
    tensors = {
        **_cache_tensors("source", pair.source),
        **_cache_tensors("target", pair.target),
    }
    metadata = {
        "format": "kvbridge-calibration",
        "schema": "1",
        "sequence_id": sequence_id,
        "source_keys_are_content": str(pair.source.keys_are_content).lower(),
        "target_keys_are_content": str(pair.target.keys_are_content).lower(),
        "source_interleaved": str(
            pair.source.rotary.interleaved if pair.source.rotary else False
        ).lower(),
        "target_interleaved": str(
            pair.target.rotary.interleaved if pair.target.rotary else False
        ).lower(),
    }
    save_file(tensors, str(destination), metadata=metadata)
    return destination


def _bool(value: str | None) -> bool:
    return value == "true"


def load_calibration_shard(path: str | Path) -> CalibrationPair:
    source_path = Path(path)
    with safe_open(  # type: ignore[no-untyped-call]
        source_path, framework="pt", device="cpu"
    ) as stream:
        metadata = stream.metadata() or {}
        if metadata.get("format") != "kvbridge-calibration" or metadata.get("schema") != "1":
            raise ArtifactError(f"unsupported calibration shard: {source_path}")
        names = set(stream.keys())
        required = {"source.keys", "source.values", "target.keys", "target.values"}
        if not required.issubset(names):
            raise ArtifactError(f"calibration shard is missing tensors: {sorted(required - names)}")
        tensors = {name: stream.get_tensor(name) for name in names}

    def build(prefix: str) -> KVCache:
        rotary = None
        if f"{prefix}.rope_cos" in tensors and f"{prefix}.rope_sin" in tensors:
            rotary = RotaryFactors(
                tensors[f"{prefix}.rope_cos"],
                tensors[f"{prefix}.rope_sin"],
                _bool(metadata.get(f"{prefix}_interleaved")),
            )
        return KVCache(
            tensors[f"{prefix}.keys"],
            tensors[f"{prefix}.values"],
            rotary,
            _bool(metadata.get(f"{prefix}_keys_are_content")),
        )

    return CalibrationPair(build("source"), build("target"))


def calibration_shard_factory(
    directory: str | Path,
    *,
    pattern: str = "*.safetensors",
) -> Callable[[], Iterable[CalibrationPair]]:
    root = Path(directory)

    def iterate() -> Iterable[CalibrationPair]:
        paths = sorted(root.glob(pattern))
        if not paths:
            raise ArtifactError(f"no calibration shards matched {root / pattern}")
        for path in paths:
            yield load_calibration_shard(path)

    return iterate
