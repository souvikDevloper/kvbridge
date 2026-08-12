"""Fail-closed validation for real-model experiment evidence."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from pathlib import Path
from typing import Any

from safetensors import SafetensorError, safe_open

from kvbridge.errors import ArtifactError
from kvbridge.mapper import CrossModelKVMapper
from kvbridge.planning import ExperimentConfig


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"could not read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ArtifactError(f"{label} must contain a JSON object")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactError(message)


def _same_digest(actual: Any, expected: str) -> bool:
    return isinstance(actual, str) and hmac.compare_digest(actual, expected)


def validate_capture_evidence(
    config_path: str | Path,
    calibration_dir: str | Path,
    *,
    require_shard_hashes: bool = True,
) -> dict[str, Any]:
    """Validate completeness, provenance, metadata, and hashes for calibration shards."""
    config_file = Path(config_path)
    root = Path(calibration_dir)
    config = ExperimentConfig.load(config_file)
    manifest_path = root / "capture_manifest.json"
    manifest = _load_json(manifest_path, "capture manifest")
    _require(manifest.get("schema_version") == 1, "unsupported capture manifest schema")
    _require(
        _same_digest(manifest.get("config_sha256"), sha256_file(config_file)),
        "capture manifest config hash differs from the requested config",
    )
    _require(
        manifest.get("sequences") == config.calibration_sequences,
        "capture manifest sequence count differs from the config",
    )
    _require(
        manifest.get("tokens") == config.calibration_tokens,
        "capture manifest token count differs from the config",
    )
    _require(
        isinstance(manifest.get("code_revision"), str) and bool(manifest["code_revision"]),
        "capture manifest has no code revision",
    )

    shard_paths = sorted(root.glob("*.safetensors"))
    expected_names = [f"{index:05}.safetensors" for index in range(config.calibration_sequences)]
    _require(
        [path.name for path in shard_paths] == expected_names,
        "calibration shard set is incomplete or unexpectedly named",
    )
    records = manifest.get("shards")
    if records is None and not require_shard_hashes:
        records = []
    if not isinstance(records, list):
        raise ArtifactError("capture manifest is missing shard integrity records")
    if require_shard_hashes:
        _require(
            len(records) == len(shard_paths),
            "capture manifest shard record count differs from the shard set",
        )

    records_by_name = {
        record.get("name"): record for record in records if isinstance(record, dict)
    }
    for path in shard_paths:
        record = records_by_name.get(path.name)
        if require_shard_hashes:
            _require(record is not None, f"capture manifest has no record for {path.name}")
        try:
            with safe_open(  # type: ignore[no-untyped-call]
                path, framework="pt", device="cpu"
            ) as stream:
                metadata = stream.metadata() or {}
                names = set(stream.keys())
        except (OSError, SafetensorError) as error:
            raise ArtifactError(f"could not inspect calibration shard: {path.name}") from error
        required_tensors = {
            "source.keys",
            "source.values",
            "target.keys",
            "target.values",
        }
        _require(
            metadata.get("format") == "kvbridge-calibration"
            and metadata.get("schema") == "1",
            f"unsupported calibration shard metadata: {path.name}",
        )
        _require(
            required_tensors.issubset(names),
            f"calibration shard is missing required tensors: {path.name}",
        )
        if record is not None:
            _require(record.get("bytes") == path.stat().st_size, f"size mismatch: {path.name}")
            _require(
                _same_digest(record.get("sha256"), sha256_file(path)),
                f"checksum mismatch: {path.name}",
            )
            _require(
                record.get("sequence_id") == metadata.get("sequence_id"),
                f"sequence id mismatch: {path.name}",
            )
    return {
        "stage": "capture",
        "config_sha256": sha256_file(config_file),
        "manifest_sha256": sha256_file(manifest_path),
        "code_revision": manifest["code_revision"],
        "shards": len(shard_paths),
        "shard_hashes_verified": bool(records),
    }


def validate_mapper_evidence(
    config_path: str | Path,
    calibration_dir: str | Path,
    artifact_dir: str | Path,
) -> dict[str, Any]:
    """Validate a fitted mapper and bind it to its config and capture manifest."""
    config_file = Path(config_path)
    calibration_root = Path(calibration_dir)
    artifact_root = Path(artifact_dir)
    config = ExperimentConfig.load(config_file)
    capture = validate_capture_evidence(config_file, calibration_root)
    mapper = CrossModelKVMapper.load(artifact_root)
    fit_run_path = artifact_root / "fit_run.json"
    fit_run = _load_json(fit_run_path, "fit run")
    _require(fit_run.get("schema_version") == 1, "unsupported fit run schema")
    _require(
        _same_digest(fit_run.get("config_sha256"), sha256_file(config_file)),
        "fit run config hash differs from the requested config",
    )
    _require(
        _same_digest(fit_run.get("capture_manifest_sha256"), capture["manifest_sha256"]),
        "fit run capture hash differs from the validated capture",
    )
    _require(
        fit_run.get("code_revision") == capture["code_revision"],
        "fit run code revision differs from the validated capture",
    )
    _require(mapper.config == config.fit, "mapper fit configuration differs from the config")
    _require(
        mapper.source_signature.model_id == config.source.model_id
        and mapper.source_signature.revision == config.source.revision,
        "mapper source identity differs from the config",
    )
    _require(
        mapper.target_signature.model_id == config.target.model_id
        and mapper.target_signature.revision == config.target.revision,
        "mapper target identity differs from the config",
    )
    return {
        "stage": "fit",
        "config_sha256": sha256_file(config_file),
        "capture_manifest_sha256": capture["manifest_sha256"],
        "code_revision": capture["code_revision"],
        "artifact_manifest_sha256": sha256_file(artifact_root / "manifest.json"),
        "fit_run_sha256": sha256_file(fit_run_path),
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _close(actual: Any, expected: float) -> bool:
    return isinstance(actual, int | float) and math.isclose(
        float(actual), expected, rel_tol=1e-9, abs_tol=1e-12
    )


def validate_result_evidence(
    config_path: str | Path,
    calibration_dir: str | Path,
    artifact_dir: str | Path,
    result_path: str | Path,
) -> dict[str, Any]:
    """Validate result provenance and recompute its aggregate metrics from raw cases."""
    config_file = Path(config_path)
    artifact_root = Path(artifact_dir)
    result_file = Path(result_path)
    fit = validate_mapper_evidence(config_file, calibration_dir, artifact_root)
    raw_config = _load_json(config_file, "experiment config")
    payload = _load_json(result_file, "evaluation result")
    _require(payload.get("schema_version") == 1, "unsupported evaluation result schema")
    _require(
        _same_digest(payload.get("config_sha256"), sha256_file(config_file)),
        "evaluation config hash differs from the requested config",
    )
    _require(
        _same_digest(
            payload.get("artifact_manifest_sha256"), fit["artifact_manifest_sha256"]
        ),
        "evaluation artifact hash differs from the validated mapper",
    )
    _require(
        payload.get("code_revision") == fit["code_revision"],
        "evaluation code revision differs from the validated fit",
    )
    cases = payload.get("cases")
    summary = payload.get("summary")
    if not isinstance(cases, list) or not cases:
        raise ArtifactError("evaluation result has no raw cases")
    if not isinstance(summary, dict):
        raise ArtifactError("evaluation result has no summary")
    desired = int(raw_config["evaluation"]["sequences"])
    _require(len(cases) == desired, "evaluation case count differs from the config")
    _require(summary.get("sequences") == desired, "summary sequence count is inconsistent")

    cache_r2 = [float(case["cache_r2"]) for case in cases]
    attention = [float(case["attention_cosine_mean"]) for case in cases]
    attention_min = [float(case["attention_cosine_min"]) for case in cases]
    kl = [float(case["logit_kl"]) for case in cases]
    transfer = [float(case["transfer_ms"]) for case in cases]
    target_prefill = [float(case["target_prefix_prefill_ms"]) for case in cases]
    expected = {
        "cache_r2_mean": sum(cache_r2) / desired,
        "attention_cosine_mean": sum(attention) / desired,
        "attention_cosine_min": min(attention_min),
        "logit_kl_mean": sum(kl) / desired,
        "logit_kl_p95": _percentile(kl, 0.95),
        "next_token_agreement": sum(bool(case["next_token_agreement"]) for case in cases)
        / desired,
        "transfer_ms_median": _percentile(transfer, 0.50),
        "transfer_ms_p95": _percentile(transfer, 0.95),
        "target_prefix_prefill_ms_median": _percentile(target_prefill, 0.50),
        "prefill_to_transfer_speed_ratio_median": _percentile(
            [baseline / mapped for baseline, mapped in zip(target_prefill, transfer, strict=True)],
            0.50,
        ),
    }
    for name, value in expected.items():
        _require(_close(summary.get(name), value), f"summary metric is inconsistent: {name}")

    evaluation = raw_config["evaluation"]
    attention_passed = expected["attention_cosine_min"] >= float(
        evaluation["attention_cosine_floor"]
    )
    kl_passed = expected["logit_kl_p95"] <= float(evaluation["logit_kl_ceiling"])
    _require(
        summary.get("attention_gate_passed") is attention_passed,
        "attention gate outcome is inconsistent",
    )
    _require(summary.get("logit_kl_gate_passed") is kl_passed, "KL gate outcome is inconsistent")
    _require(
        summary.get("all_quality_gates_passed") is (attention_passed and kl_passed),
        "combined quality gate outcome is inconsistent",
    )
    return {
        "stage": "evaluation",
        "config_sha256": sha256_file(config_file),
        "artifact_manifest_sha256": fit["artifact_manifest_sha256"],
        "code_revision": fit["code_revision"],
        "result_sha256": sha256_file(result_file),
        "sequences": desired,
        "all_quality_gates_passed": attention_passed and kl_passed,
    }
