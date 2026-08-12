import json
from pathlib import Path

import pytest
import torch

from kvbridge.errors import ArtifactError
from kvbridge.evidence import (
    calibration_contract_sha256,
    index_legacy_capture_evidence,
    sha256_file,
    validate_capture_evidence,
    validate_mapper_evidence,
    validate_result_evidence,
)
from kvbridge.io import (
    atomic_write_bytes,
    atomic_write_text,
    calibration_shard_factory,
    load_calibration_shard,
    save_calibration_shard,
)
from kvbridge.statistics import bootstrap_mean_interval
from kvbridge.synthetic import fit_demo, make_problem


def test_calibration_shard_round_trip(tmp_path: Path) -> None:
    pair = make_problem(calibration_pairs=2, tokens=8).calibration[0]
    path = save_calibration_shard(tmp_path / "000.safetensors", pair, sequence_id="seq-0")

    restored = load_calibration_shard(path)

    torch.testing.assert_close(restored.source.keys[0], pair.source.keys[0])
    torch.testing.assert_close(restored.target.values[-1], pair.target.values[-1])
    assert restored.source.rotary is not None
    assert not list(tmp_path.glob(".*.tmp"))


def test_factory_is_reiterable(tmp_path: Path) -> None:
    problem = make_problem(calibration_pairs=2, tokens=8)
    for index, pair in enumerate(problem.calibration):
        save_calibration_shard(tmp_path / f"{index:03}.safetensors", pair, sequence_id=str(index))
    factory = calibration_shard_factory(tmp_path)

    assert len(list(factory())) == 2
    assert len(list(factory())) == 2


def test_atomic_text_write_replaces_complete_content(tmp_path: Path) -> None:
    destination = tmp_path / "manifest.json"
    destination.write_text("old", encoding="utf-8")

    atomic_write_text(destination, "new\n")

    assert destination.read_text(encoding="utf-8") == "new\n"
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_bytes_preserve_exact_content(tmp_path: Path) -> None:
    destination = tmp_path / "manifest.json"
    payload = b'{"line_endings":"preserved"}\r\n'

    atomic_write_bytes(destination, payload)

    assert destination.read_bytes() == payload


def _capture_evidence(tmp_path: Path) -> tuple[Path, Path]:
    problem = make_problem(calibration_pairs=2, tokens=8)
    config_path = tmp_path / "config.json"
    config = {
        "name": "synthetic-evidence",
        "source": problem.source.to_dict(),
        "target": problem.target.to_dict(),
        "fit": {
            "top_k": 1,
            "ridge_alpha": 0.01,
            "content_space": True,
            "selection_alpha": 1e-6,
            "accumulation_dtype": "float64",
            "accumulation_device": "cpu",
            "require_matched_kv": True,
            "target_layer_block_size": 1,
            "selection_target_layer_block_size": 1,
        },
        "calibration": {
            "dataset": "synthetic",
            "dataset_revision": "v1",
            "sequences": 2,
            "tokens": 8,
            "stride": 1,
        },
        "evaluation": {
            "sequences": 1,
            "tokens": 8,
            "attention_cosine_floor": 0.9,
            "logit_kl_ceiling": 0.2,
        },
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    calibration_dir = tmp_path / "calibration"
    records = []
    for index, pair in enumerate(problem.calibration):
        sequence_id = f"synthetic:{index}"
        path = save_calibration_shard(
            calibration_dir / f"{index:05}.safetensors", pair, sequence_id=sequence_id
        )
        records.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "sequence_id": sequence_id,
            }
        )
    manifest = {
        "schema_version": 1,
        "code_revision": "capture-revision",
        "config_sha256": sha256_file(config_path),
        "calibration_contract_sha256": calibration_contract_sha256(config_path),
        "sequences": 2,
        "tokens": 8,
        "shards": records,
    }
    atomic_write_text(
        calibration_dir / "capture_manifest.json", json.dumps(manifest) + "\n"
    )
    return config_path, calibration_dir


def test_capture_evidence_validates_shard_hashes(tmp_path: Path) -> None:
    config_path, calibration_dir = _capture_evidence(tmp_path)

    report = validate_capture_evidence(config_path, calibration_dir)

    assert report["shards"] == 2
    assert report["shard_hashes_verified"] is True


def test_capture_evidence_rejects_tampered_shard(tmp_path: Path) -> None:
    config_path, calibration_dir = _capture_evidence(tmp_path)
    shard = calibration_dir / "00000.safetensors"
    payload = bytearray(shard.read_bytes())
    payload[-1] ^= 1
    shard.write_bytes(payload)

    with pytest.raises(ArtifactError, match="checksum mismatch"):
        validate_capture_evidence(config_path, calibration_dir)


def test_capture_evidence_is_reusable_across_fit_ablations(tmp_path: Path) -> None:
    config_path, calibration_dir = _capture_evidence(tmp_path)
    ablation_path = tmp_path / "ablation.json"
    ablation = json.loads(config_path.read_text(encoding="utf-8"))
    ablation["name"] = "synthetic-ridge-ablation"
    ablation["fit"]["ridge_alpha"] = 1.0
    ablation["evaluation"]["logit_kl_ceiling"] = 0.5
    ablation_path.write_text(json.dumps(ablation), encoding="utf-8")

    report = validate_capture_evidence(ablation_path, calibration_dir)

    assert report["calibration_contract_sha256"] == calibration_contract_sha256(
        config_path
    )


def test_capture_evidence_rejects_changed_calibration_contract(tmp_path: Path) -> None:
    config_path, calibration_dir = _capture_evidence(tmp_path)
    incompatible_path = tmp_path / "incompatible.json"
    incompatible = json.loads(config_path.read_text(encoding="utf-8"))
    incompatible["calibration"]["tokens"] = 16
    incompatible_path.write_text(json.dumps(incompatible), encoding="utf-8")

    with pytest.raises(ArtifactError, match="calibration contract differs"):
        validate_capture_evidence(incompatible_path, calibration_dir)


def test_legacy_capture_integrity_index_is_recoverable(tmp_path: Path) -> None:
    config_path, calibration_dir = _capture_evidence(tmp_path)
    manifest_path = calibration_dir / "capture_manifest.json"
    legacy = json.loads(manifest_path.read_text(encoding="utf-8"))
    legacy.pop("calibration_contract_sha256")
    legacy.pop("shards")
    original = (json.dumps(legacy) + "\r\n").encode()
    atomic_write_bytes(manifest_path, original)

    report = index_legacy_capture_evidence(
        config_path, calibration_dir, index_code_revision="index-revision"
    )

    assert report["shard_hashes_verified"] is True
    backups = list(calibration_dir.glob("capture_manifest.legacy-*.json"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original


def _complete_evidence(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    config_path, calibration_dir = _capture_evidence(tmp_path)
    problem = make_problem(calibration_pairs=2, tokens=8)
    mapper = fit_demo(problem)
    artifact_dir = mapper.save(tmp_path / "artifact")
    capture_manifest = calibration_dir / "capture_manifest.json"
    calibration_bytes = sum(
        path.stat().st_size for path in calibration_dir.glob("*.safetensors")
    )
    fit_run = {
        "schema_version": 1,
        "code_revision": "fit-revision",
        "capture_code_revision": "capture-revision",
        "config_sha256": sha256_file(config_path),
        "capture_manifest_sha256": sha256_file(capture_manifest),
        "calibration_shards": 2,
        "calibration_bytes": calibration_bytes,
        "calibration_data_passes": 4,
        "estimated_calibration_bytes_read": calibration_bytes * 4,
        "artifact_storage_dtype": "float32",
        "elapsed_seconds": 1.0,
        "fit_key_r2_mean": sum(mapper.fit_key_r2) / len(mapper.fit_key_r2),
        "fit_value_r2_mean": sum(mapper.fit_value_r2) / len(mapper.fit_value_r2),
    }
    atomic_write_text(artifact_dir / "fit_run.json", json.dumps(fit_run) + "\n")
    case = {
        "cache_r2": 0.8,
        "attention_cosine_mean": 0.95,
        "attention_cosine_min": 0.91,
        "logit_kl": 0.1,
        "next_token_agreement": True,
        "transfer_ms": 2.0,
        "target_prefix_prefill_ms": 8.0,
    }
    summary = {
        "sequences": 1,
        "cache_r2_mean": 0.8,
        "attention_cosine_mean": 0.95,
        "attention_cosine_min": 0.91,
        "logit_kl_mean": 0.1,
        "logit_kl_p95": 0.1,
        "next_token_agreement": 1.0,
        "transfer_ms_median": 2.0,
        "transfer_ms_p95": 2.0,
        "target_prefix_prefill_ms_median": 8.0,
        "prefill_to_transfer_speed_ratio_median": 4.0,
        "attention_gate_passed": True,
        "logit_kl_gate_passed": True,
        "all_quality_gates_passed": True,
        "confidence_intervals": {
            "cache_r2_mean": bootstrap_mean_interval([0.8], resamples=100).to_dict(),
            "attention_cosine_mean": bootstrap_mean_interval(
                [0.95], resamples=100
            ).to_dict(),
            "logit_kl_mean": bootstrap_mean_interval([0.1], resamples=100).to_dict(),
            "next_token_agreement": bootstrap_mean_interval(
                [1.0], resamples=100
            ).to_dict(),
        },
    }
    result_path = tmp_path / "result.json"
    result = {
        "schema_version": 1,
        "code_revision": "fit-revision",
        "config_sha256": sha256_file(config_path),
        "artifact_manifest_sha256": sha256_file(artifact_dir / "manifest.json"),
        "summary": summary,
        "cases": [case],
    }
    atomic_write_text(result_path, json.dumps(result) + "\n")
    return config_path, calibration_dir, artifact_dir, result_path


def test_complete_evidence_chain_validates(tmp_path: Path) -> None:
    config_path, calibration_dir, artifact_dir, result_path = _complete_evidence(tmp_path)

    fit_report = validate_mapper_evidence(config_path, calibration_dir, artifact_dir)
    result_report = validate_result_evidence(
        config_path, calibration_dir, artifact_dir, result_path
    )

    assert fit_report["stage"] == "fit"
    assert result_report["stage"] == "evaluation"
    assert result_report["all_quality_gates_passed"] is True


def test_mapper_evidence_recomputes_fit_summary(tmp_path: Path) -> None:
    config_path, calibration_dir, artifact_dir, _ = _complete_evidence(tmp_path)
    fit_run_path = artifact_dir / "fit_run.json"
    payload = json.loads(fit_run_path.read_text(encoding="utf-8"))
    payload["fit_value_r2_mean"] = -100.0
    atomic_write_text(fit_run_path, json.dumps(payload) + "\n")

    with pytest.raises(ArtifactError, match="value R2 is inconsistent"):
        validate_mapper_evidence(config_path, calibration_dir, artifact_dir)


def test_result_evidence_recomputes_summary(tmp_path: Path) -> None:
    config_path, calibration_dir, artifact_dir, result_path = _complete_evidence(tmp_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["summary"]["logit_kl_mean"] = 0.01
    atomic_write_text(result_path, json.dumps(payload) + "\n")

    with pytest.raises(ArtifactError, match="summary metric is inconsistent"):
        validate_result_evidence(config_path, calibration_dir, artifact_dir, result_path)


def test_result_evidence_recomputes_confidence_interval(tmp_path: Path) -> None:
    config_path, calibration_dir, artifact_dir, result_path = _complete_evidence(tmp_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["summary"]["confidence_intervals"]["logit_kl_mean"]["high"] = 0.5
    atomic_write_text(result_path, json.dumps(payload) + "\n")

    with pytest.raises(ArtifactError, match="confidence interval is inconsistent"):
        validate_result_evidence(config_path, calibration_dir, artifact_dir, result_path)


def test_result_evidence_rejects_nonstandard_numeric_constants(tmp_path: Path) -> None:
    config_path, calibration_dir, artifact_dir, result_path = _complete_evidence(tmp_path)
    payload = result_path.read_text(encoding="utf-8").replace(
        '"logit_kl": 0.1', '"logit_kl": NaN', 1
    )
    atomic_write_text(result_path, payload)

    with pytest.raises(ArtifactError, match="could not read evaluation result"):
        validate_result_evidence(config_path, calibration_dir, artifact_dir, result_path)
