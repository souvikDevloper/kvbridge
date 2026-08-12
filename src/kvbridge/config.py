"""Versioned configuration and model identity types."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

from kvbridge.errors import CompatibilityError

AttentionKind = Literal["dense", "sliding_window", "hybrid", "unknown"]


@dataclass(frozen=True, slots=True)
class ModelSignature:
    """Minimum architecture contract needed to validate a cache transfer.

    `tokenizer_hash` should be the SHA-256 of a canonical tokenizer description,
    not merely a mutable model name.
    """

    model_id: str
    revision: str
    tokenizer_hash: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    attention_kind: AttentionKind = "dense"
    architecture: str = "unknown"

    def __post_init__(self) -> None:
        if min(self.num_layers, self.num_kv_heads, self.head_dim) <= 0:
            raise ValueError("layer, KV-head, and head dimensions must be positive")
        if not self.model_id or not self.revision or not self.tokenizer_hash:
            raise ValueError("model_id, revision, and tokenizer_hash are required")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate_pair(self, target: ModelSignature, *, require_matched_kv: bool = True) -> None:
        if self.tokenizer_hash != target.tokenizer_hash:
            raise CompatibilityError(
                "source and target tokenizers differ; token positions would no longer align"
            )
        if self.attention_kind != "dense" or target.attention_kind != "dense":
            raise CompatibilityError(
                "v0.1 supports dense full-attention models only; hybrid/windowed caches are unsafe"
            )
        if require_matched_kv and (
            self.num_kv_heads != target.num_kv_heads or self.head_dim != target.head_dim
        ):
            raise CompatibilityError(
                "this artifact requires matched KV heads and head dimensions, matching the paper's "
                "validated regime"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ModelSignature:
        return cls(**value)


@dataclass(frozen=True, slots=True)
class FitConfig:
    """Mapper fitting knobs with paper-faithful defaults."""

    top_k: int = 4
    ridge_alpha: float = 0.01
    content_space: bool = True
    selection_alpha: float = 1e-6
    accumulation_dtype: str = "float64"
    accumulation_device: str = "cpu"
    require_matched_kv: bool = True
    target_layer_block_size: int = 1
    selection_target_layer_block_size: int = 1
    token_stride: int = 1

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.ridge_alpha < 0 or self.selection_alpha < 0:
            raise ValueError("ridge penalties cannot be negative")
        if self.accumulation_dtype not in {"float32", "float64"}:
            raise ValueError("accumulation_dtype must be float32 or float64")
        if self.accumulation_device not in {"cpu", "cuda"}:
            raise ValueError("accumulation_device must be cpu or cuda")
        if self.target_layer_block_size <= 0:
            raise ValueError("target_layer_block_size must be positive")
        if self.selection_target_layer_block_size <= 0:
            raise ValueError("selection_target_layer_block_size must be positive")
        if self.token_stride <= 0:
            raise ValueError("token_stride must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FitConfig:
        return cls(**value)
