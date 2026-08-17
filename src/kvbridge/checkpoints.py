"""Fail-closed, restartable checkpoints for long paper-scale fits."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from kvbridge.config import FitConfig, ModelSignature
from kvbridge.errors import ArtifactError

CHECKPOINT_SCHEMA = 1


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class LayerCheckpoint:
    key_weight: Tensor
    value_weight: Tensor
    key_bias: Tensor
    value_bias: Tensor
    key_r2: float
    value_r2: float


class FitCheckpointStore:
    """Revision/config-bound checkpoint store using JSON and SafeTensors only."""

    def __init__(
        self,
        root: str | Path,
        source: ModelSignature,
        target: ModelSignature,
        config: FitConfig,
        *,
        resume: bool,
    ) -> None:
        self.root = Path(root)
        self.contract = {
            "source_signature": source.to_dict(),
            "target_signature": target.to_dict(),
            "fit_config": config.to_dict(),
        }
        self.contract_sha256 = _canonical_sha256(self.contract)
        manifest_path = self.root / "checkpoint_manifest.json"
        if self.root.exists() and any(self.root.iterdir()):
            if not resume:
                raise ArtifactError(
                    f"checkpoint directory is non-empty and resume is disabled: {self.root}"
                )
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ArtifactError("checkpoint manifest is missing or invalid") from error
            if (
                manifest.get("format") != "kvbridge-fit-checkpoint"
                or manifest.get("schema_version") != CHECKPOINT_SCHEMA
                or manifest.get("contract_sha256") != self.contract_sha256
                or manifest.get("contract") != self.contract
            ):
                raise ArtifactError("checkpoint contract differs from this fit")
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            manifest = {
                "format": "kvbridge-fit-checkpoint",
                "schema_version": CHECKPOINT_SCHEMA,
                "contract_sha256": self.contract_sha256,
                "contract": self.contract,
            }
            _atomic_text(
                manifest_path,
                json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            )

    def _selection_path(self, block_start: int, block_end: int) -> Path:
        return self.root / f"selection.{block_start:05d}-{block_end - 1:05d}.json"

    def load_selection(
        self, block_start: int, block_end: int
    ) -> tuple[list[list[int]], list[list[float]]] | None:
        path = self._selection_path(block_start, block_end)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactError(f"invalid selection checkpoint: {path}") from error
        if (
            payload.get("format") != "kvbridge-selection-checkpoint"
            or payload.get("schema_version") != CHECKPOINT_SCHEMA
            or payload.get("contract_sha256") != self.contract_sha256
            or payload.get("block") != [block_start, block_end]
        ):
            raise ArtifactError(f"selection checkpoint metadata mismatch: {path}")
        selected = payload.get("selected_layers")
        scores = payload.get("selection_scores")
        expected_rows = block_end - block_start
        if not isinstance(selected, list) or not isinstance(scores, list):
            raise ArtifactError(f"selection checkpoint payload is malformed: {path}")
        if len(selected) != expected_rows or len(scores) != expected_rows:
            raise ArtifactError(f"selection checkpoint row count is invalid: {path}")
        if not all(
            isinstance(row, list) and all(isinstance(item, int) for item in row)
            for row in selected
        ):
            raise ArtifactError(f"selection checkpoint indices are malformed: {path}")
        if not all(
            isinstance(row, list)
            and all(isinstance(item, int | float) for item in row)
            for row in scores
        ):
            raise ArtifactError(f"selection checkpoint scores are malformed: {path}")
        return selected, [[float(item) for item in row] for row in scores]

    def save_selection(
        self,
        block_start: int,
        block_end: int,
        selected: list[list[int]],
        scores: list[list[float]],
    ) -> None:
        payload = {
            "format": "kvbridge-selection-checkpoint",
            "schema_version": CHECKPOINT_SCHEMA,
            "contract_sha256": self.contract_sha256,
            "block": [block_start, block_end],
            "selected_layers": selected,
            "selection_scores": scores,
        }
        _atomic_text(
            self._selection_path(block_start, block_end),
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )

    def _layer_path(self, target_layer: int) -> Path:
        return self.root / f"fit-layer.{target_layer:05d}.safetensors"

    def load_layer(
        self,
        target_layer: int,
        selected_layers: list[int],
        *,
        weight_shape: tuple[int, int, int],
        bias_shape: tuple[int, int],
    ) -> LayerCheckpoint | None:
        path = self._layer_path(target_layer)
        if not path.exists():
            return None
        try:
            with safe_open(  # type: ignore[no-untyped-call, unused-ignore]
                path, framework="pt", device="cpu"
            ) as stream:
                metadata = stream.metadata() or {}
                names = set(stream.keys())
                expected_names = {"key_weight", "value_weight", "key_bias", "value_bias"}
                if names != expected_names:
                    raise ArtifactError(f"layer checkpoint tensor set is invalid: {path}")
                tensors = {name: stream.get_tensor(name) for name in names}
        except OSError as error:
            raise ArtifactError(f"could not read layer checkpoint: {path}") from error
        if (
            metadata.get("format") != "kvbridge-layer-checkpoint"
            or metadata.get("schema") != str(CHECKPOINT_SCHEMA)
            or metadata.get("contract_sha256") != self.contract_sha256
            or metadata.get("target_layer") != str(target_layer)
            or metadata.get("selected_layers")
            != json.dumps(selected_layers, separators=(",", ":"))
        ):
            raise ArtifactError(f"layer checkpoint metadata mismatch: {path}")
        if any(
            tensors[name].shape != weight_shape for name in ("key_weight", "value_weight")
        ) or any(tensors[name].shape != bias_shape for name in ("key_bias", "value_bias")):
            raise ArtifactError(f"layer checkpoint tensor shape mismatch: {path}")
        if any(
            tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all().item())
            for tensor in tensors.values()
        ):
            raise ArtifactError(f"layer checkpoint tensors must be finite FP32: {path}")
        try:
            key_r2 = float(metadata["key_r2"])
            value_r2 = float(metadata["value_r2"])
        except (KeyError, ValueError) as error:
            raise ArtifactError(f"layer checkpoint diagnostics are invalid: {path}") from error
        if not torch.isfinite(torch.tensor([key_r2, value_r2])).all():
            raise ArtifactError(f"layer checkpoint diagnostics are non-finite: {path}")
        return LayerCheckpoint(
            tensors["key_weight"],
            tensors["value_weight"],
            tensors["key_bias"],
            tensors["value_bias"],
            key_r2,
            value_r2,
        )

    def save_layer(
        self,
        target_layer: int,
        selected_layers: list[int],
        checkpoint: LayerCheckpoint,
    ) -> None:
        path = self._layer_path(target_layer)
        tensors = {
            "key_weight": checkpoint.key_weight.detach().float().cpu().contiguous(),
            "value_weight": checkpoint.value_weight.detach().float().cpu().contiguous(),
            "key_bias": checkpoint.key_bias.detach().float().cpu().contiguous(),
            "value_bias": checkpoint.value_bias.detach().float().cpu().contiguous(),
        }
        if any(not bool(torch.isfinite(tensor).all().item()) for tensor in tensors.values()):
            raise ArtifactError("refusing to checkpoint non-finite mapper tensors")
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            save_file(
                tensors,
                str(temporary),
                metadata={
                    "format": "kvbridge-layer-checkpoint",
                    "schema": str(CHECKPOINT_SCHEMA),
                    "contract_sha256": self.contract_sha256,
                    "target_layer": str(target_layer),
                    "selected_layers": json.dumps(selected_layers, separators=(",", ":")),
                    "key_r2": repr(checkpoint.key_r2),
                    "value_r2": repr(checkpoint.value_r2),
                },
            )
            with temporary.open("rb+") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
