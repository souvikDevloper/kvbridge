"""Deterministic memory/storage planning before expensive model downloads."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kvbridge.config import FitConfig, ModelSignature

GIB = 1024**3


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str
    source: ModelSignature
    target: ModelSignature
    fit: FitConfig
    calibration_sequences: int
    calibration_tokens: int
    token_stride: int = 4
    cache_dtype_bytes: int = 2
    artifact_dtype_bytes: int = 4

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=payload["name"],
            source=ModelSignature.from_dict(payload["source"]),
            target=ModelSignature.from_dict(payload["target"]),
            fit=FitConfig.from_dict(payload["fit"]),
            calibration_sequences=int(payload["calibration"]["sequences"]),
            calibration_tokens=int(payload["calibration"]["tokens"]),
            token_stride=int(payload["calibration"].get("stride", 4)),
            cache_dtype_bytes=int(payload.get("cache_dtype_bytes", 2)),
            artifact_dtype_bytes=int(payload.get("artifact_dtype_bytes", 4)),
        )


@dataclass(frozen=True, slots=True)
class ScalePlan:
    name: str
    observations: int
    mapper_parameters: int
    mapper_gib: float
    fit_block_working_set_gib: float
    selection_block_working_set_gib: float
    calibration_cache_pair_gib: float
    calibration_data_passes: int
    estimated_mapper_load_ms_pcie25: float
    estimated_mapper_load_ms_pcie50: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cache_bytes(signature: ModelSignature, sequences: int, tokens: int, dtype_bytes: int) -> int:
    return (
        2
        * signature.num_layers
        * sequences
        * tokens
        * signature.num_kv_heads
        * signature.head_dim
        * dtype_bytes
    )


def build_scale_plan(config: ExperimentConfig) -> ScalePlan:
    config.source.validate_pair(config.target, require_matched_kv=config.fit.require_matched_kv)
    k = min(config.fit.top_k, config.source.num_layers)
    source_features = k * config.source.num_kv_heads * config.source.head_dim
    target_features = config.target.num_kv_heads * config.target.head_dim
    mapper_parameters = 2 * config.target.num_layers * target_features * (source_features + 1)
    accumulator_bytes = 8 if config.fit.accumulation_dtype == "float64" else 4
    # K and V each retain X'X, X'Y, and first moments for every layer in a block.
    per_fit_layer = (
        2
        * (
            source_features**2
            + source_features * target_features
            + source_features
            + target_features
            + 1
        )
        * accumulator_bytes
    )
    selection_items = (
        config.source.num_layers
        * config.target.num_kv_heads
        * 2
        * (
            config.source.head_dim**2
            + config.source.head_dim * config.target.head_dim
            + config.source.head_dim
            + config.target.head_dim
            + 1
        )
        * accumulator_bytes
    )
    source_cache = _cache_bytes(
        config.source, 1, config.calibration_tokens, config.cache_dtype_bytes
    )
    target_cache = _cache_bytes(
        config.target, 1, config.calibration_tokens, config.cache_dtype_bytes
    )
    mapper_bytes = mapper_parameters * config.artifact_dtype_bytes
    selection_passes = math.ceil(
        config.target.num_layers / config.fit.selection_target_layer_block_size
    )
    fit_passes = math.ceil(config.target.num_layers / config.fit.target_layer_block_size)
    return ScalePlan(
        name=config.name,
        observations=config.calibration_sequences
        * math.ceil(config.calibration_tokens / config.token_stride),
        mapper_parameters=mapper_parameters,
        mapper_gib=mapper_bytes / GIB,
        fit_block_working_set_gib=(per_fit_layer * config.fit.target_layer_block_size / GIB),
        selection_block_working_set_gib=(
            selection_items * config.fit.selection_target_layer_block_size / GIB
        ),
        calibration_cache_pair_gib=(source_cache + target_cache) / GIB,
        calibration_data_passes=selection_passes + fit_passes,
        estimated_mapper_load_ms_pcie25=mapper_bytes / 25e9 * 1000,
        estimated_mapper_load_ms_pcie50=mapper_bytes / 50e9 * 1000,
    )
