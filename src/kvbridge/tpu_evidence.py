"""Fail-closed validation for TPU scale-fit and held-out evaluation evidence."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from pathlib import Path
from typing import Any, NoReturn, cast

from kvbridge.errors import ArtifactError
from kvbridge.evidence import calibration_contract_sha256
from kvbridge.mapper import CrossModelKVMapper
from kvbridge.planning import ExperimentConfig, build_scale_plan


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant: {value}")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_constant
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ArtifactError(f"could not read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ArtifactError(f"{label} must be a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactError(message)


def _digest(actual: Any, expected: str) -> bool:
    return isinstance(actual, str) and hmac.compare_digest(actual, expected)


def _finite(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _close(actual: Any, expected: float) -> bool:
    return _finite(actual) and math.isclose(
        float(actual), expected, rel_tol=1e-9, abs_tol=1e-12
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _validate_xla_environment(payload: dict[str, Any]) -> None:
    environment = payload.get("environment")
    _require(isinstance(environment, dict), "evidence environment is missing")
    environment = cast(dict[str, Any], environment)
    xla = environment.get("xla")
    _require(isinstance(xla, dict), "evidence has no XLA runtime manifest")
    xla = cast(dict[str, Any], xla)
    _require(xla.get("spmd") is True, "evidence was not recorded in XLA SPMD mode")
    _require(
        xla.get("model_sharding_strategy") == "spmd_largest_axis",
        "evidence has an unsupported XLA model-sharding strategy",
    )
    devices = xla.get("global_runtime_devices")
    _require(
        isinstance(devices, int) and devices >= 8,
        "evidence has an invalid XLA runtime device count",
    )
    _require(bool(xla.get("device_type")), "evidence has no XLA device type")


def validate_tpu_fit(config_path: str | Path, run_dir: str | Path) -> dict[str, Any]:
    """Validate a complete mapper fit against its immutable experiment inputs."""
    config_file = Path(config_path)
    root = Path(run_dir)
    config = ExperimentConfig.load(config_file)
    plan = build_scale_plan(config)
    artifact_dir = root / "artifact"
    mapper = CrossModelKVMapper.load(artifact_dir)
    fit_run = _load_json(root / "fit_run.json", "TPU fit run")
    _require(
        fit_run.get("format") == "kvbridge-tpu-scale-run"
        and fit_run.get("schema_version") == 1,
        "unsupported TPU fit evidence schema",
    )
    _require(
        fit_run.get("status") == "fit-complete-evaluation-pending",
        "TPU fit status is not complete",
    )
    _require(_digest(fit_run.get("config_sha256"), _sha256(config_file)), "config hash mismatch")
    _require(
        _digest(
            fit_run.get("calibration_contract_sha256"),
            calibration_contract_sha256(config_file),
        ),
        "calibration contract hash mismatch",
    )
    _require(
        _digest(
            fit_run.get("artifact_manifest_sha256"),
            _sha256(artifact_dir / "manifest.json"),
        ),
        "artifact manifest hash mismatch",
    )
    _require(mapper.config == config.fit, "mapper fit configuration differs from config")
    for role, actual, expected_signature in (
        ("source", mapper.source_signature, config.source),
        ("target", mapper.target_signature, config.target),
    ):
        _require(
            actual.model_id == expected_signature.model_id
            and actual.revision == expected_signature.revision,
            f"mapper {role} identity differs from config",
        )
        _require(
            fit_run.get(f"{role}_fingerprint") == actual.fingerprint,
            f"fit {role} fingerprint differs from mapper",
        )
    raw = _load_json(config_file, "experiment config")
    calibration = raw["calibration"]
    for field, expected_value in (
        ("dataset", calibration["dataset"]),
        ("dataset_revision", calibration["dataset_revision"]),
        ("split", calibration.get("split", "train")),
        ("sequences", config.calibration_sequences),
        ("tokens_per_sequence", config.calibration_tokens),
        ("stride", config.token_stride),
        ("observations", plan.observations),
    ):
        _require(fit_run.get(field) == expected_value, f"fit {field} differs from config")
    row_indices = fit_run.get("dataset_row_indices")
    _require(
        isinstance(row_indices, list)
        and len(row_indices) == config.calibration_sequences
        and all(isinstance(value, int) and value >= 0 for value in row_indices)
        and len(set(row_indices)) == len(row_indices),
        "fit dataset row index set is invalid",
    )
    _require(
        _close(fit_run.get("fit_key_r2_mean"), sum(mapper.fit_key_r2) / len(mapper.fit_key_r2)),
        "fit key R2 mean differs from artifact",
    )
    _require(
        _close(
            fit_run.get("fit_value_r2_mean"),
            sum(mapper.fit_value_r2) / len(mapper.fit_value_r2),
        ),
        "fit value R2 mean differs from artifact",
    )
    _require(
        isinstance(fit_run.get("code_revision"), str) and bool(fit_run["code_revision"]),
        "fit run has no code revision",
    )
    _validate_xla_environment(fit_run)
    return {
        "stage": "tpu-fit",
        "config_sha256": _sha256(config_file),
        "artifact_manifest_sha256": _sha256(artifact_dir / "manifest.json"),
        "observations": plan.observations,
        "code_revision": fit_run["code_revision"],
    }


def validate_tpu_evaluation(
    config_path: str | Path, run_dir: str | Path, result_path: str | Path
) -> dict[str, Any]:
    """Validate raw held-out TPU metrics and recompute their aggregate gates."""
    fit = validate_tpu_fit(config_path, run_dir)
    config_file = Path(config_path)
    result_file = Path(result_path)
    root = Path(run_dir)
    raw = _load_json(config_file, "experiment config")
    evaluation = raw["evaluation"]
    result = _load_json(result_file, "TPU evaluation")
    _require(
        result.get("format") == "kvbridge-tpu-evaluation"
        and result.get("schema_version") == 1,
        "unsupported TPU evaluation schema",
    )
    _require(
        _digest(result.get("config_sha256"), _sha256(config_file)),
        "evaluation config hash mismatch",
    )
    _require(
        _digest(
            result.get("artifact_manifest_sha256"),
            _sha256(root / "artifact" / "manifest.json"),
        ),
        "evaluation artifact hash mismatch",
    )
    mapper = CrossModelKVMapper.load(root / "artifact")
    _require(
        result.get("source_fingerprint") == mapper.source_signature.fingerprint
        and result.get("target_fingerprint") == mapper.target_signature.fingerprint,
        "evaluation model fingerprints differ from artifact",
    )
    thresholds = result.get("thresholds")
    _require(isinstance(thresholds, dict), "evaluation thresholds are missing")
    thresholds = cast(dict[str, Any], thresholds)
    _require(
        thresholds.get("attention_cosine_floor") == evaluation["attention_cosine_floor"]
        and thresholds.get("logit_kl_ceiling") == evaluation["logit_kl_ceiling"],
        "evaluation thresholds differ from config",
    )
    timing = result.get("timing_policy")
    _require(isinstance(timing, dict), "evaluation timing policy is missing")
    timing = cast(dict[str, Any], timing)
    batch_size = timing.get("batch_size")
    _require(isinstance(batch_size, int) and batch_size > 0, "evaluation batch size is invalid")
    batch_size = cast(int, batch_size)
    _require(
        timing.get("hardware_local_only") is True,
        "evaluation does not mark timings hardware-local",
    )
    cases = result.get("cases")
    _require(isinstance(cases, list) and bool(cases), "evaluation has no raw cases")
    cases = cast(list[Any], cases)
    expected_sequences = int(evaluation["sequences"])
    _require(
        len(cases) * batch_size == expected_sequences,
        "evaluation case count differs from config",
    )
    metric_names = (
        "cache_r2",
        "attention_cosine_mean",
        "attention_cosine_min",
        "logit_kl",
        "next_token_agreement",
        "transfer_ms",
        "target_prefix_prefill_ms",
        "target_full_prefill_ms",
        "suffix_decode_ms",
    )
    sequence_ids: list[str] = []
    for index, case in enumerate(cases):
        _require(isinstance(case, dict), f"evaluation case {index} is not an object")
        case = cast(dict[str, Any], case)
        _require(
            case.get("batch_index") == index and case.get("batch_size") == batch_size,
            f"evaluation case {index} batch metadata is invalid",
        )
        _require(
            all(_finite(case.get(name)) for name in metric_names),
            f"evaluation case {index} has non-finite metrics",
        )
        identifiers = case.get("sequence_ids")
        _require(
            isinstance(identifiers, list)
            and len(identifiers) == batch_size
            and all(isinstance(item, str) and item for item in identifiers),
            f"evaluation case {index} sequence ids are invalid",
        )
        identifiers = cast(list[str], identifiers)
        sequence_ids.extend(identifiers)
        input_digest = case.get("input_sha256")
        _require(
            isinstance(input_digest, str)
            and len(input_digest) == 64
            and all(character in "0123456789abcdef" for character in input_digest),
            f"evaluation case {index} input hash is invalid",
        )
    _require(len(sequence_ids) == len(set(sequence_ids)), "evaluation sequence ids are duplicated")

    summary = result.get("summary")
    _require(isinstance(summary, dict), "evaluation summary is missing")
    summary = cast(dict[str, Any], summary)
    cache_values = [float(case["cache_r2"]) for case in cases]
    attention = [float(case["attention_cosine_mean"]) for case in cases]
    attention_min = [float(case["attention_cosine_min"]) for case in cases]
    kl = [float(case["logit_kl"]) for case in cases]
    agreement = [float(case["next_token_agreement"]) for case in cases]
    transfer = [float(case["transfer_ms"]) for case in cases]
    prefill = [float(case["target_prefix_prefill_ms"]) for case in cases]
    expected_metrics = {
        "sequences": float(expected_sequences),
        "batches": float(len(cases)),
        "cache_r2_mean": sum(cache_values) / len(cases),
        "attention_cosine_mean": sum(attention) / len(cases),
        "attention_cosine_min": min(attention_min),
        "logit_kl_mean": sum(kl) / len(cases),
        "logit_kl_p95": _percentile(kl, 0.95),
        "next_token_agreement": sum(agreement) / len(cases),
        "transfer_ms_batch_median": _percentile(transfer, 0.50),
        "target_prefix_prefill_ms_batch_median": _percentile(prefill, 0.50),
        "prefill_to_transfer_ratio_batch_median": _percentile(
            [base / mapped for base, mapped in zip(prefill, transfer, strict=True)], 0.50
        ),
    }
    _require(
        all(_close(summary.get(name), value) for name, value in expected_metrics.items()),
        "evaluation summary differs from raw cases",
    )
    attention_passed = expected_metrics["attention_cosine_min"] >= float(
        evaluation["attention_cosine_floor"]
    )
    kl_passed = expected_metrics["logit_kl_p95"] <= float(evaluation["logit_kl_ceiling"])
    _require(
        summary.get("attention_gate_passed") is attention_passed
        and summary.get("logit_kl_gate_passed") is kl_passed
        and summary.get("all_quality_gates_passed") is (attention_passed and kl_passed),
        "evaluation gate summary is inconsistent",
    )
    _validate_xla_environment(result)
    return {
        **fit,
        "stage": "tpu-evaluation",
        "result_sha256": _sha256(result_file),
        "sequences": expected_sequences,
        "all_quality_gates_passed": attention_passed and kl_passed,
    }
