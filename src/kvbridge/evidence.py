"""Fail-closed validation for real-model experiment evidence."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn

from safetensors import SafetensorError, safe_open

from kvbridge.errors import ArtifactError
from kvbridge.mapper import CrossModelKVMapper
from kvbridge.planning import ExperimentConfig, build_scale_plan
from kvbridge.statistics import bootstrap_mean_interval

# safetensors releases have alternated between typed and untyped ``safe_open``
# exports.  Giving the compatibility boundary one explicit type avoids
# version-dependent ``no-untyped-call`` / ``unused-ignore`` failures while
# keeping the rest of this validation module under strict mypy checking.
_safe_open: Any = safe_open


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibration_contract_sha256(config_path: str | Path) -> str:
    """Hash only inputs that determine captured cache tensors.

    Fit, stride, storage, and evaluation knobs are deliberately excluded so an
    immutable capture can be reused for controlled ablations without weakening
    model/dataset provenance.
    """
    payload = _load_json(Path(config_path), "experiment config")
    calibration = payload.get("calibration")
    if not isinstance(calibration, dict):
        raise ArtifactError("experiment config has no calibration object")
    contract = {
        "source": payload.get("source"),
        "target": payload.get("target"),
        "calibration": {
            "dataset": calibration.get("dataset"),
            "dataset_revision": calibration.get("dataset_revision"),
            "split": calibration.get("split", "train"),
            "sequences": calibration.get("sequences"),
            "tokens": calibration.get("tokens"),
        },
        "model_dtype": payload.get("model_dtype", "auto"),
        "attention_implementation": payload.get("attention_implementation", "sdpa"),
    }
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def index_legacy_capture_evidence(
    config_path: str | Path,
    calibration_dir: str | Path,
    *,
    index_code_revision: str,
) -> dict[str, Any]:
    """Add a recoverable integrity index to a legacy, exact-config capture.

    The original manifest is retained byte-for-byte. This migration refuses
    cross-config input; ablation reuse is enabled only after every shard has
    been inspected and hashed under the requested legacy config.
    """
    from kvbridge.io import atomic_write_bytes, atomic_write_text

    if not index_code_revision:
        raise ArtifactError("integrity-index code revision is required")
    config_file = Path(config_path)
    root = Path(calibration_dir)
    manifest_path = root / "capture_manifest.json"
    original_bytes = manifest_path.read_bytes()
    try:
        original_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ArtifactError("legacy capture manifest is not valid UTF-8") from error
    manifest = _load_json(manifest_path, "capture manifest")
    if manifest.get("calibration_contract_sha256") is not None:
        return validate_capture_evidence(config_file, root)
    _require(
        _same_digest(manifest.get("config_sha256"), sha256_file(config_file)),
        "legacy capture must match the exact config before integrity indexing",
    )
    config = ExperimentConfig.load(config_file)
    _require(
        manifest.get("sequences") == config.calibration_sequences,
        "legacy capture sequence count differs from the config",
    )
    _require(
        manifest.get("tokens") == config.calibration_tokens,
        "legacy capture token count differs from the config",
    )
    paths = sorted(root.glob("*.safetensors"))
    expected_names = [f"{index:05}.safetensors" for index in range(config.calibration_sequences)]
    _require(
        [path.name for path in paths] == expected_names,
        "legacy calibration shard set is incomplete or unexpectedly named",
    )
    records: list[dict[str, Any]] = []
    for path in paths:
        try:
            with _safe_open(path, framework="pt", device="cpu") as stream:
                metadata = stream.metadata() or {}
                names = set(stream.keys())
        except (OSError, SafetensorError) as error:
            raise ArtifactError(f"could not inspect calibration shard: {path.name}") from error
        _require(
            metadata.get("format") == "kvbridge-calibration"
            and metadata.get("schema") == "1",
            f"unsupported calibration shard metadata: {path.name}",
        )
        _require(
            {"source.keys", "source.values", "target.keys", "target.values"}.issubset(names),
            f"calibration shard is missing required tensors: {path.name}",
        )
        sequence_id = metadata.get("sequence_id")
        _require(bool(sequence_id), f"calibration shard has no sequence id: {path.name}")
        records.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "sequence_id": sequence_id,
            }
        )

    original_sha256 = hashlib.sha256(original_bytes).hexdigest()
    backup_path = root / f"capture_manifest.legacy-{original_sha256[:12]}.json"
    if backup_path.exists():
        _require(
            backup_path.read_bytes() == original_bytes,
            "legacy manifest backup path already contains different content",
        )
    else:
        atomic_write_bytes(backup_path, original_bytes)
    upgraded = dict(manifest)
    upgraded.update(
        {
            "calibration_contract_sha256": calibration_contract_sha256(config_file),
            "shards": records,
            "integrity_source_manifest_sha256": original_sha256,
            "integrity_index_code_revision": index_code_revision,
            "integrity_indexed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    atomic_write_text(
        manifest_path, json.dumps(upgraded, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return validate_capture_evidence(config_file, root)


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ArtifactError(f"could not read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ArtifactError(f"{label} must contain a JSON object")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactError(message)


def _same_digest(actual: Any, expected: str) -> bool:
    return isinstance(actual, str) and hmac.compare_digest(actual, expected)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _close(actual: Any, expected: float) -> bool:
    return (
        _finite_number(actual)
        and math.isfinite(expected)
        and math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-12)
    )


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
    contract_digest = calibration_contract_sha256(config_file)
    recorded_contract = manifest.get("calibration_contract_sha256")
    if recorded_contract is None:
        _require(
            _same_digest(manifest.get("config_sha256"), sha256_file(config_file)),
            "legacy capture manifest config hash differs from the requested config",
        )
    else:
        _require(
            _same_digest(recorded_contract, contract_digest),
            "capture manifest calibration contract differs from the requested config",
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
            with _safe_open(path, framework="pt", device="cpu") as stream:
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
        "calibration_contract_sha256": contract_digest,
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
        fit_run.get("capture_code_revision") == capture["code_revision"],
        "fit run does not identify the validated capture code revision",
    )
    _require(
        isinstance(fit_run.get("code_revision"), str) and bool(fit_run["code_revision"]),
        "fit run has no code revision",
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
    calibration_paths = sorted(calibration_root.glob("*.safetensors"))
    calibration_bytes = sum(path.stat().st_size for path in calibration_paths)
    plan = build_scale_plan(config)
    _require(
        fit_run.get("calibration_shards") == config.calibration_sequences,
        "fit run calibration shard count is inconsistent",
    )
    _require(
        fit_run.get("calibration_bytes") == calibration_bytes,
        "fit run calibration byte count is inconsistent",
    )
    _require(
        fit_run.get("calibration_data_passes") == plan.calibration_data_passes,
        "fit run calibration pass count is inconsistent",
    )
    _require(
        fit_run.get("estimated_calibration_bytes_read")
        == calibration_bytes * plan.calibration_data_passes,
        "fit run estimated calibration I/O is inconsistent",
    )
    _require(
        fit_run.get("artifact_storage_dtype") == mapper.storage_dtype,
        "fit run artifact dtype is inconsistent",
    )
    elapsed_seconds = fit_run.get("elapsed_seconds")
    elapsed_value = (
        float(elapsed_seconds)
        if isinstance(elapsed_seconds, int | float) and not isinstance(elapsed_seconds, bool)
        else math.nan
    )
    _require(
        math.isfinite(elapsed_value) and elapsed_value > 0,
        "fit run elapsed time must be finite and positive",
    )
    expected_key_r2 = sum(mapper.fit_key_r2) / len(mapper.fit_key_r2)
    expected_value_r2 = sum(mapper.fit_value_r2) / len(mapper.fit_value_r2)
    _require(
        _close(fit_run.get("fit_key_r2_mean"), expected_key_r2),
        "fit run key R2 is inconsistent with the artifact",
    )
    _require(
        _close(fit_run.get("fit_value_r2_mean"), expected_value_r2),
        "fit run value R2 is inconsistent with the artifact",
    )
    return {
        "stage": "fit",
        "config_sha256": sha256_file(config_file),
        "capture_manifest_sha256": capture["manifest_sha256"],
        "capture_code_revision": capture["code_revision"],
        "code_revision": fit_run["code_revision"],
        "artifact_manifest_sha256": sha256_file(artifact_root / "manifest.json"),
        "fit_run_sha256": sha256_file(fit_run_path),
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _case_metric(case: dict[str, Any], name: str, index: int) -> float:
    value = case.get(name)
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ArtifactError(f"evaluation case {index} has non-finite metric: {name}")
    return float(value)


def _validate_result_payload(
    raw_config: dict[str, Any], payload: dict[str, Any]
) -> tuple[int, bool, bool]:
    """Recompute all published aggregates, intervals, and gate outcomes."""
    cases = payload.get("cases")
    summary = payload.get("summary")
    if not isinstance(cases, list) or not cases:
        raise ArtifactError("evaluation result has no raw cases")
    if not isinstance(summary, dict):
        raise ArtifactError("evaluation result has no summary")
    desired = int(raw_config["evaluation"]["sequences"])
    _require(len(cases) == desired, "evaluation case count differs from the config")
    _require(summary.get("sequences") == desired, "summary sequence count is inconsistent")

    if not all(isinstance(case, dict) for case in cases):
        raise ArtifactError("evaluation cases must contain JSON objects")
    typed_cases = [case for case in cases if isinstance(case, dict)]
    cache_r2 = [_case_metric(case, "cache_r2", index) for index, case in enumerate(typed_cases)]
    attention = [
        _case_metric(case, "attention_cosine_mean", index)
        for index, case in enumerate(typed_cases)
    ]
    attention_min = [
        _case_metric(case, "attention_cosine_min", index)
        for index, case in enumerate(typed_cases)
    ]
    kl = [_case_metric(case, "logit_kl", index) for index, case in enumerate(typed_cases)]
    transfer = [
        _case_metric(case, "transfer_ms", index) for index, case in enumerate(typed_cases)
    ]
    target_prefill = [
        _case_metric(case, "target_prefix_prefill_ms", index)
        for index, case in enumerate(typed_cases)
    ]
    _require(all(value > 0 for value in transfer), "transfer latency must be positive")
    _require(
        all(value > 0 for value in target_prefill), "target-prefill latency must be positive"
    )
    for index, case in enumerate(typed_cases):
        _require(
            isinstance(case.get("next_token_agreement"), bool),
            f"evaluation case {index} has invalid next-token agreement",
        )
        per_layer = case.get("attention_cosine_per_layer")
        if per_layer is not None:
            _require(
                isinstance(per_layer, list)
                and bool(per_layer)
                and all(
                    isinstance(value, int | float)
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in per_layer
                ),
                f"evaluation case {index} has invalid per-layer attention metrics",
            )
    expected = {
        "cache_r2_mean": sum(cache_r2) / desired,
        "attention_cosine_mean": sum(attention) / desired,
        "attention_cosine_min": min(attention_min),
        "logit_kl_mean": sum(kl) / desired,
        "logit_kl_p95": _percentile(kl, 0.95),
        "next_token_agreement": sum(
            bool(case["next_token_agreement"]) for case in typed_cases
        )
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

    intervals = summary.get("confidence_intervals")
    if intervals is not None:
        if not isinstance(intervals, dict):
            raise ArtifactError("confidence intervals must contain a JSON object")
        interval_inputs = {
            "cache_r2_mean": cache_r2,
            "attention_cosine_mean": attention,
            "logit_kl_mean": kl,
            "next_token_agreement": [
                float(bool(case["next_token_agreement"])) for case in typed_cases
            ],
        }
        for name, values in interval_inputs.items():
            record = intervals.get(name)
            if not isinstance(record, dict):
                raise ArtifactError(f"missing confidence interval: {name}")
            try:
                recomputed = bootstrap_mean_interval(
                    values,
                    confidence=float(record["confidence"]),
                    resamples=int(record["resamples"]),
                    seed=int(record["seed"]),
                ).to_dict()
            except (KeyError, TypeError, ValueError) as error:
                raise ArtifactError(f"invalid confidence interval: {name}") from error
            for field in ("low", "center", "high"):
                _require(
                    _close(record.get(field), float(recomputed[field])),
                    f"confidence interval is inconsistent: {name}.{field}",
                )

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
    return desired, attention_passed, kl_passed


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

    if not all(isinstance(case, dict) for case in cases):
        raise ArtifactError("evaluation cases must contain JSON objects")
    typed_cases = [case for case in cases if isinstance(case, dict)]
    cache_r2 = [_case_metric(case, "cache_r2", index) for index, case in enumerate(typed_cases)]
    attention = [
        _case_metric(case, "attention_cosine_mean", index)
        for index, case in enumerate(typed_cases)
    ]
    attention_min = [
        _case_metric(case, "attention_cosine_min", index)
        for index, case in enumerate(typed_cases)
    ]
    kl = [_case_metric(case, "logit_kl", index) for index, case in enumerate(typed_cases)]
    transfer = [
        _case_metric(case, "transfer_ms", index) for index, case in enumerate(typed_cases)
    ]
    target_prefill = [
        _case_metric(case, "target_prefix_prefill_ms", index)
        for index, case in enumerate(typed_cases)
    ]
    _require(all(value > 0 for value in transfer), "transfer latency must be positive")
    _require(
        all(value > 0 for value in target_prefill), "target-prefill latency must be positive"
    )
    for index, case in enumerate(typed_cases):
        _require(
            isinstance(case.get("next_token_agreement"), bool),
            f"evaluation case {index} has invalid next-token agreement",
        )
        per_layer = case.get("attention_cosine_per_layer")
        if per_layer is not None:
            _require(
                isinstance(per_layer, list)
                and bool(per_layer)
                and all(
                    isinstance(value, int | float)
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in per_layer
                ),
                f"evaluation case {index} has invalid per-layer attention metrics",
            )
    expected = {
        "cache_r2_mean": sum(cache_r2) / desired,
        "attention_cosine_mean": sum(attention) / desired,
        "attention_cosine_min": min(attention_min),
        "logit_kl_mean": sum(kl) / desired,
        "logit_kl_p95": _percentile(kl, 0.95),
        "next_token_agreement": sum(
            bool(case["next_token_agreement"]) for case in typed_cases
        )
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

    intervals = summary.get("confidence_intervals")
    if intervals is not None:
        if not isinstance(intervals, dict):
            raise ArtifactError("confidence intervals must contain a JSON object")
        interval_inputs = {
            "cache_r2_mean": cache_r2,
            "attention_cosine_mean": attention,
            "logit_kl_mean": kl,
            "next_token_agreement": [
                float(bool(case["next_token_agreement"])) for case in typed_cases
            ],
        }
        for name, values in interval_inputs.items():
            record = intervals.get(name)
            if not isinstance(record, dict):
                raise ArtifactError(f"missing confidence interval: {name}")
            try:
                recomputed = bootstrap_mean_interval(
                    values,
                    confidence=float(record["confidence"]),
                    resamples=int(record["resamples"]),
                    seed=int(record["seed"]),
                ).to_dict()
            except (KeyError, TypeError, ValueError) as error:
                raise ArtifactError(f"invalid confidence interval: {name}") from error
            for field in ("low", "center", "high"):
                _require(
                    _close(record.get(field), float(recomputed[field])),
                    f"confidence interval is inconsistent: {name}.{field}",
                )

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


def validate_published_result_evidence(
    config_path: str | Path, evidence_dir: str | Path
) -> dict[str, Any]:
    """Validate a lightweight published result without large shards or weights.

    This verifies the configuration/capture/fit/mapper/result hash chain and
    recomputes every result aggregate, confidence interval, and quality gate.
    It deliberately reports that calibration shard contents and mapper tensor
    contents were not re-verified because those large files are not published
    in the Git repository.
    """
    config_file = Path(config_path)
    root = Path(evidence_dir)
    config = ExperimentConfig.load(config_file)
    raw_config = _load_json(config_file, "experiment config")
    capture_path = root / "capture_manifest.json"
    mapper_path = root / "mapper_manifest.json"
    fit_path = root / "fit_run.json"
    result_path = root / "result.json"
    capture = _load_json(capture_path, "published capture manifest")
    mapper = _load_json(mapper_path, "published mapper manifest")
    fit = _load_json(fit_path, "published fit run")
    result = _load_json(result_path, "published evaluation result")

    config_digest = sha256_file(config_file)
    capture_digest = sha256_file(capture_path)
    mapper_digest = sha256_file(mapper_path)
    _require(capture.get("schema_version") == 1, "unsupported capture manifest schema")
    _require(mapper.get("schema_version") == 1, "unsupported mapper manifest schema")
    _require(fit.get("schema_version") == 1, "unsupported fit run schema")
    _require(result.get("schema_version") == 1, "unsupported evaluation result schema")
    _require(
        _same_digest(capture.get("config_sha256"), config_digest),
        "published capture config hash differs from the requested config",
    )
    _require(
        _same_digest(
            capture.get("calibration_contract_sha256"),
            calibration_contract_sha256(config_file),
        ),
        "published capture calibration contract differs from the requested config",
    )
    _require(
        capture.get("sequences") == config.calibration_sequences
        and capture.get("tokens") == config.calibration_tokens,
        "published capture dimensions differ from the requested config",
    )
    _require(
        isinstance(capture.get("code_revision"), str) and bool(capture["code_revision"]),
        "published capture has no code revision",
    )

    source_signature = capture.get("source_signature")
    target_signature = capture.get("target_signature")
    if isinstance(source_signature, dict) and isinstance(target_signature, dict):
        for side, recorded, requested in (
            ("source", source_signature, raw_config["source"]),
            ("target", target_signature, raw_config["target"]),
        ):
            for field in (
                "model_id",
                "revision",
                "num_layers",
                "num_kv_heads",
                "head_dim",
                "attention_kind",
                "architecture",
            ):
                _require(
                    recorded.get(field) == requested.get(field),
                    f"published {side} signature differs from the config: {field}",
                )
        _require(
            mapper.get("source_signature") == source_signature
            and mapper.get("target_signature") == target_signature,
            "published mapper signatures differ from the capture",
        )
    for side in ("source", "target"):
        captured_fingerprint = capture.get(f"{side}_fingerprint")
        if captured_fingerprint is not None:
            _require(
                mapper.get(f"{side}_fingerprint") == captured_fingerprint,
                f"published mapper {side} fingerprint differs from the capture",
            )
    _require(mapper.get("fit_config") == config.fit.to_dict(), "published fit config differs")
    _require(
        mapper.get("storage_dtype") == fit.get("artifact_storage_dtype"),
        "published artifact storage dtype is inconsistent",
    )

    _require(
        _same_digest(fit.get("config_sha256"), config_digest),
        "published fit config hash differs from the requested config",
    )
    _require(
        _same_digest(fit.get("capture_manifest_sha256"), capture_digest),
        "published fit capture hash differs from the capture manifest",
    )
    _require(
        fit.get("capture_code_revision") == capture.get("code_revision"),
        "published fit does not identify the capture code revision",
    )
    _require(
        fit.get("calibration_shards") == config.calibration_sequences,
        "published fit calibration shard count is inconsistent",
    )
    _require(
        fit.get("calibration_data_passes") == build_scale_plan(config).calibration_data_passes,
        "published fit calibration pass count is inconsistent",
    )
    key_r2 = mapper.get("fit_key_r2")
    value_r2 = mapper.get("fit_value_r2")
    for r2_name, r2_values, r2_recorded in (
        ("key", key_r2, fit.get("fit_key_r2_mean")),
        ("value", value_r2, fit.get("fit_value_r2_mean")),
    ):
        if not isinstance(r2_values, list) or not r2_values:
            raise ArtifactError(f"published mapper {r2_name} R2 is invalid")
        _require(
            all(_finite_number(value) for value in r2_values),
            f"published mapper {r2_name} R2 is invalid",
        )
        _require(
            _close(
                r2_recorded,
                sum(float(value) for value in r2_values) / len(r2_values),
            ),
            f"published fit {r2_name} R2 is inconsistent",
        )

    _require(
        _same_digest(result.get("config_sha256"), config_digest),
        "published result config hash differs from the requested config",
    )
    _require(
        _same_digest(result.get("artifact_manifest_sha256"), mapper_digest),
        "published result mapper hash differs from the mapper manifest",
    )
    _require(
        result.get("code_revision") == fit.get("code_revision"),
        "published result code revision differs from the fit",
    )
    desired, attention_passed, kl_passed = _validate_result_payload(raw_config, result)
    return {
        "stage": "published-evaluation",
        "config_sha256": config_digest,
        "capture_manifest_sha256": capture_digest,
        "mapper_manifest_sha256": mapper_digest,
        "fit_run_sha256": sha256_file(fit_path),
        "result_sha256": sha256_file(result_path),
        "code_revision": fit["code_revision"],
        "sequences": desired,
        "all_quality_gates_passed": attention_passed and kl_passed,
        "calibration_shards_verified": False,
        "mapper_weights_verified": False,
    }
