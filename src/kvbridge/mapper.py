"""Inference mapper plus safe, versioned artifact serialization."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from kvbridge.cache import KVCache, RotaryFactors
from kvbridge.config import FitConfig, ModelSignature
from kvbridge.errors import ArtifactError, CacheValidationError
from kvbridge.features import selected_layer_features

SCHEMA_VERSION = 1
PAPER_URL = "https://arxiv.org/abs/2608.03893"


@dataclass(frozen=True, slots=True)
class TransferReport:
    source_model: str
    target_model: str
    tokens: int
    elapsed_ms: float
    content_space: bool
    artifact_schema: int = SCHEMA_VERSION


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CrossModelKVMapper:
    """Map a source cache into a target cache without target-model prefill."""

    def __init__(
        self,
        *,
        source_signature: ModelSignature,
        target_signature: ModelSignature,
        config: FitConfig,
        selected_layers: Sequence[Sequence[int]],
        key_weights: Sequence[Tensor],
        value_weights: Sequence[Tensor],
        key_biases: Sequence[Tensor],
        value_biases: Sequence[Tensor],
        selection_scores: Sequence[Sequence[float]] | None = None,
        fit_key_r2: Sequence[float] | None = None,
        fit_value_r2: Sequence[float] | None = None,
    ) -> None:
        self.source_signature = source_signature
        self.target_signature = target_signature
        self.config = config
        self.selected_layers = [list(row) for row in selected_layers]
        self.key_weights = list(key_weights)
        self.value_weights = list(value_weights)
        self.key_biases = list(key_biases)
        self.value_biases = list(value_biases)
        self.selection_scores = [list(row) for row in selection_scores or []]
        self.fit_key_r2 = list(fit_key_r2 or [])
        self.fit_value_r2 = list(fit_value_r2 or [])
        self._validate()

    def _validate(self) -> None:
        layers = self.target_signature.num_layers
        collections = (
            self.selected_layers,
            self.key_weights,
            self.value_weights,
            self.key_biases,
            self.value_biases,
        )
        if any(len(items) != layers for items in collections):
            raise ArtifactError("artifact tensors do not cover every target layer")
        for layer in range(layers):
            feature_count = (
                len(self.selected_layers[layer])
                * self.source_signature.num_kv_heads
                * self.source_signature.head_dim
            )
            weight_shape = (
                self.target_signature.num_kv_heads,
                feature_count,
                self.target_signature.head_dim,
            )
            bias_shape = (self.target_signature.num_kv_heads, self.target_signature.head_dim)
            if (
                self.key_weights[layer].shape != weight_shape
                or self.value_weights[layer].shape != weight_shape
            ):
                raise ArtifactError(f"invalid weight shape at target layer {layer}")
            if (
                self.key_biases[layer].shape != bias_shape
                or self.value_biases[layer].shape != bias_shape
            ):
                raise ArtifactError(f"invalid bias shape at target layer {layer}")
            if any(
                index < 0 or index >= self.source_signature.num_layers
                for index in self.selected_layers[layer]
            ):
                raise ArtifactError(f"source-layer index out of range at target layer {layer}")

    def _validate_cache(self, source: KVCache) -> None:
        layers, _, heads, _, dim = source.shape
        expected = (
            self.source_signature.num_layers,
            self.source_signature.num_kv_heads,
            self.source_signature.head_dim,
        )
        if (layers, heads, dim) != expected:
            raise CacheValidationError(
                f"source cache {(layers, heads, dim)} does not match artifact {expected}"
            )

    @staticmethod
    def _project(
        features: Tensor, weight: Tensor, bias: Tensor, output_dtype: torch.dtype
    ) -> Tensor:
        compute_dtype = torch.float32
        projected = torch.einsum(
            "nf,hfd->nhd",
            features.to(dtype=compute_dtype),
            weight.to(device=features.device, dtype=compute_dtype),
        )
        projected += bias.to(device=features.device, dtype=compute_dtype).unsqueeze(0)
        return projected.to(output_dtype)

    def map(self, source: KVCache, *, target_rotary: RotaryFactors | None = None) -> KVCache:
        self._validate_cache(source)
        if self.config.content_space:
            source = source.to_content_space()
        batch, _, tokens, _ = source.keys[0].shape
        mapped_keys: list[Tensor] = []
        mapped_values: list[Tensor] = []
        for layer, selected in enumerate(self.selected_layers):
            key_features = selected_layer_features(source.keys, selected)
            value_features = selected_layer_features(source.values, selected)
            key = self._project(
                key_features, self.key_weights[layer], self.key_biases[layer], source.keys[0].dtype
            )
            value = self._project(
                value_features,
                self.value_weights[layer],
                self.value_biases[layer],
                source.values[0].dtype,
            )
            # [batch*tokens, heads, dim] -> [batch, heads, tokens, dim]
            mapped_keys.append(key.reshape(batch, tokens, *key.shape[1:]).permute(0, 2, 1, 3))
            mapped_values.append(value.reshape(batch, tokens, *value.shape[1:]).permute(0, 2, 1, 3))
        cache = KVCache(mapped_keys, mapped_values, keys_are_content=self.config.content_space)
        if self.config.content_space:
            if target_rotary is None:
                raise CacheValidationError(
                    "target RoPE factors are required for content-space mapping"
                )
            cache = cache.apply_rotary(target_rotary)
        return cache

    def transfer(
        self, source: KVCache, *, target_rotary: RotaryFactors | None = None
    ) -> tuple[KVCache, TransferReport]:
        started = time.perf_counter()
        result = self.map(source, target_rotary=target_rotary)
        report = TransferReport(
            source_model=self.source_signature.model_id,
            target_model=self.target_signature.model_id,
            tokens=source.shape[3],
            elapsed_ms=(time.perf_counter() - started) * 1000,
            content_space=self.config.content_space,
        )
        return result, report

    def _manifest(self, weights_sha256: str) -> dict[str, Any]:
        return {
            "format": "kvbridge",
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "paper": PAPER_URL,
            "weights_sha256": weights_sha256,
            "source_signature": self.source_signature.to_dict(),
            "target_signature": self.target_signature.to_dict(),
            "source_fingerprint": self.source_signature.fingerprint,
            "target_fingerprint": self.target_signature.fingerprint,
            "fit_config": self.config.to_dict(),
            "selected_layers": self.selected_layers,
            "selection_scores": self.selection_scores,
            "fit_key_r2": self.fit_key_r2,
            "fit_value_r2": self.fit_value_r2,
        }

    def save(self, directory: str | Path, *, overwrite: bool = False) -> Path:
        root = Path(directory)
        if root.exists() and any(root.iterdir()) and not overwrite:
            raise ArtifactError(f"refusing to overwrite non-empty artifact directory: {root}")
        root.mkdir(parents=True, exist_ok=True)
        tensors: dict[str, Tensor] = {}
        for layer in range(self.target_signature.num_layers):
            tensors[f"key_weight.{layer}"] = self.key_weights[layer].detach().cpu().contiguous()
            tensors[f"value_weight.{layer}"] = self.value_weights[layer].detach().cpu().contiguous()
            tensors[f"key_bias.{layer}"] = self.key_biases[layer].detach().cpu().contiguous()
            tensors[f"value_bias.{layer}"] = self.value_biases[layer].detach().cpu().contiguous()
        weights_path, temp_weights = root / "mapper.safetensors", root / ".mapper.safetensors.tmp"
        manifest_path, temp_manifest = root / "manifest.json", root / ".manifest.json.tmp"
        save_file(
            tensors,
            str(temp_weights),
            metadata={"format": "kvbridge", "schema": str(SCHEMA_VERSION)},
        )
        os.replace(temp_weights, weights_path)
        temp_manifest.write_text(
            json.dumps(self._manifest(_sha256(weights_path)), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_manifest, manifest_path)
        return root

    @classmethod
    def load(cls, directory: str | Path) -> CrossModelKVMapper:
        root = Path(directory)
        manifest_path, weights_path = root / "manifest.json", root / "mapper.safetensors"
        if not manifest_path.is_file() or not weights_path.is_file():
            raise ArtifactError("artifact must contain manifest.json and mapper.safetensors")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactError("could not parse artifact manifest") from error
        if manifest.get("format") != "kvbridge" or manifest.get("schema_version") != SCHEMA_VERSION:
            raise ArtifactError("unsupported artifact format/schema")
        if not isinstance(manifest.get("weights_sha256"), str) or not hmac.compare_digest(
            manifest["weights_sha256"], _sha256(weights_path)
        ):
            raise ArtifactError("artifact checksum verification failed")
        tensors = load_file(str(weights_path), device="cpu")
        source = ModelSignature.from_dict(manifest["source_signature"])
        target = ModelSignature.from_dict(manifest["target_signature"])
        layers = target.num_layers
        try:
            return cls(
                source_signature=source,
                target_signature=target,
                config=FitConfig.from_dict(manifest["fit_config"]),
                selected_layers=manifest["selected_layers"],
                key_weights=[tensors[f"key_weight.{layer}"] for layer in range(layers)],
                value_weights=[tensors[f"value_weight.{layer}"] for layer in range(layers)],
                key_biases=[tensors[f"key_bias.{layer}"] for layer in range(layers)],
                value_biases=[tensors[f"value_bias.{layer}"] for layer in range(layers)],
                selection_scores=manifest.get("selection_scores"),
                fit_key_r2=manifest.get("fit_key_r2"),
                fit_value_r2=manifest.get("fit_value_r2"),
            )
        except KeyError as error:
            raise ArtifactError(f"artifact is missing tensor {error}") from error
