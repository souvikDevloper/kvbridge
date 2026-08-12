import json
from pathlib import Path

import pytest
import torch

from kvbridge.errors import ArtifactError
from kvbridge.evidence import (
    sha256_file,
    validate_capture_evidence,
    validate_mapper_evidence,
    validate_result_evidence,
)
from kvbridge.io import (
    atomic_write_text,
    calibration_shard_factory,
    load_calibration_shard,
    save_calibration_shard,
)
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
        "code_revision": "test-revision",
        "config_sha256": sha256_file(config_path),
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


def _complete_evidence(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    config_path, calibration_dir = _capture_evidence(tmp_path)
    problem = make_problem(calibration_pairs=2, tokens=8)
    artifact_dir = fit_demo(problem).save(tmp_path / "artifact")
    capture_manifest = calibration_dir / "capture_manifest.json"
    fit_run = {
        "schema_version": 1,
        "code_revision": "test-revision",
        "config_sha256": sha256_file(config_path),
        "capture_manifest_sha256": sha256_file(capture_manifest),
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
    }
    result_path = tmp_path / "result.json"
    result = {
        "schema_version": 1,
        "code_revision": "test-revision",
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


def test_result_evidence_recomputes_summary(tmp_path: Path) -> None:
    config_path, calibration_dir, artifact_dir, result_path = _complete_evidence(tmp_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["summary"]["logit_kl_mean"] = 0.01
    atomic_write_text(result_path, json.dumps(payload) + "\n")

    with pytest.raises(ArtifactError, match="summary metric is inconsistent"):
        validate_result_evidence(config_path, calibration_dir, artifact_dir, result_path)
