import hashlib
import json
from pathlib import Path

import pytest

from kvbridge.errors import ArtifactError
from kvbridge.evidence import calibration_contract_sha256
from kvbridge.synthetic import fit_demo, make_problem
from kvbridge.tpu_evidence import validate_tpu_evaluation, validate_tpu_fit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    problem = make_problem()
    mapper = fit_demo(problem)
    config_path = tmp_path / "config.json"
    config = {
        "name": "synthetic-tpu-evidence",
        "evidence_tier": "T3",
        "source": problem.source.to_dict(),
        "target": problem.target.to_dict(),
        "fit": mapper.config.to_dict(),
        "calibration": {
            "dataset": "synthetic/dataset",
            "dataset_revision": "dataset-revision",
            "split": "train",
            "sequences": len(problem.calibration),
            "tokens": 24,
            "stride": 1,
        },
        "evaluation": {
            "sequences": 2,
            "tokens": 10,
            "dataset_skip": 100,
            "max_suffix_tokens": 1,
            "attention_cosine_floor": 0.90,
            "logit_kl_ceiling": 0.20,
        },
        "cache_dtype_bytes": 4,
        "artifact_dtype_bytes": 4,
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    run_dir = tmp_path / "run"
    artifact_dir = run_dir / "artifact"
    mapper.save(artifact_dir)
    fit_run = {
        "format": "kvbridge-tpu-scale-run",
        "schema_version": 1,
        "status": "fit-complete-evaluation-pending",
        "config_sha256": _sha256(config_path),
        "calibration_contract_sha256": calibration_contract_sha256(config_path),
        "artifact_manifest_sha256": _sha256(artifact_dir / "manifest.json"),
        "code_revision": "test-revision",
        "dataset": "synthetic/dataset",
        "dataset_revision": "dataset-revision",
        "split": "train",
        "dataset_row_indices": list(range(len(problem.calibration))),
        "sequences": len(problem.calibration),
        "tokens_per_sequence": 24,
        "stride": 1,
        "observations": len(problem.calibration) * 24,
        "source_fingerprint": problem.source.fingerprint,
        "target_fingerprint": problem.target.fingerprint,
        "fit_key_r2_mean": sum(mapper.fit_key_r2) / len(mapper.fit_key_r2),
        "fit_value_r2_mean": sum(mapper.fit_value_r2) / len(mapper.fit_value_r2),
        "environment": {
            "xla": {
                "torch_xla": "test",
                "device_type": "TPU",
                "global_runtime_devices": 8,
                "addressable_runtime_devices": 8,
                "spmd": True,
            }
        },
    }
    run_dir.mkdir(exist_ok=True)
    (run_dir / "fit_run.json").write_text(json.dumps(fit_run), encoding="utf-8")

    cases = [
        {
            "batch_index": 0,
            "sequence_ids": ["synthetic/dataset:100"],
            "input_sha256": "a" * 64,
            "batch_size": 1,
            "tokens": 10,
            "cache_r2": 0.9,
            "attention_cosine_mean": 0.95,
            "attention_cosine_min": 0.91,
            "logit_kl": 0.1,
            "next_token_agreement": 1.0,
            "transfer_ms": 2.0,
            "target_prefix_prefill_ms": 4.0,
            "target_full_prefill_ms": 5.0,
            "suffix_decode_ms": 1.0,
        },
        {
            "batch_index": 1,
            "sequence_ids": ["synthetic/dataset:101"],
            "input_sha256": "b" * 64,
            "batch_size": 1,
            "tokens": 10,
            "cache_r2": 0.8,
            "attention_cosine_mean": 0.94,
            "attention_cosine_min": 0.92,
            "logit_kl": 0.15,
            "next_token_agreement": 0.0,
            "transfer_ms": 3.0,
            "target_prefix_prefill_ms": 6.0,
            "target_full_prefill_ms": 7.0,
            "suffix_decode_ms": 1.5,
        },
    ]
    result = {
        "format": "kvbridge-tpu-evaluation",
        "schema_version": 1,
        "code_revision": "test-revision",
        "config_sha256": _sha256(config_path),
        "artifact_manifest_sha256": _sha256(artifact_dir / "manifest.json"),
        "source_fingerprint": problem.source.fingerprint,
        "target_fingerprint": problem.target.fingerprint,
        "timing_policy": {
            "warmup_batches": 1,
            "batch_size": 1,
            "cache_host_to_device_excluded": True,
            "hardware_local_only": True,
        },
        "thresholds": {
            "attention_cosine_floor": 0.90,
            "logit_kl_ceiling": 0.20,
        },
        "environment": fit_run["environment"],
        "summary": {
            "sequences": 2,
            "batches": 2,
            "cache_r2_mean": 0.85,
            "attention_cosine_mean": 0.945,
            "attention_cosine_min": 0.91,
            "logit_kl_mean": 0.125,
            "logit_kl_p95": 0.15,
            "next_token_agreement": 0.5,
            "transfer_ms_batch_median": 2.0,
            "target_prefix_prefill_ms_batch_median": 4.0,
            "prefill_to_transfer_ratio_batch_median": 2.0,
            "attention_gate_passed": True,
            "logit_kl_gate_passed": True,
            "all_quality_gates_passed": True,
        },
        "cases": cases,
    }
    result_path = run_dir / "evaluation.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return config_path, run_dir, result_path


def test_tpu_fit_and_evaluation_evidence_validate(tmp_path: Path) -> None:
    config, run, result = _write_fixture(tmp_path)

    assert validate_tpu_fit(config, run)["observations"] == 144
    report = validate_tpu_evaluation(config, run, result)
    assert report["all_quality_gates_passed"] is True


def test_tpu_evaluation_rejects_summary_tampering(tmp_path: Path) -> None:
    config, run, result = _write_fixture(tmp_path)
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["cases"][0]["logit_kl"] = 9.0
    result.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactError, match="summary differs"):
        validate_tpu_evaluation(config, run, result)
