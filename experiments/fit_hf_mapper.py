"""Fit a revision-bound mapper from captured SafeTensors calibration shards.

Dry-run is the default. Pass --execute on a CUDA host after capture completes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch

from kvbridge.config import ModelSignature
from kvbridge.evidence import validate_capture_evidence
from kvbridge.fit import fit_mapper
from kvbridge.io import atomic_write_text, calibration_shard_factory
from kvbridge.planning import ExperimentConfig, build_scale_plan
from kvbridge.provenance import code_revision, package_versions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not read capture manifest: {path}") from error
    if payload.get("schema_version") != 1:
        raise RuntimeError("unsupported capture manifest schema")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--calibration-dir", type=Path, default=Path("data/calibration"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/hf-mapper"))
    parser.add_argument("--storage-dtype", choices=("float32", "bfloat16"), default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    plan = build_scale_plan(config)
    print(json.dumps(plan.to_dict(), indent=2))
    if not args.execute:
        print("Dry-run only. Re-run with --execute after calibration capture completes.")
        return 0
    if config.fit.accumulation_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("this config requires CUDA ridge accumulation, but CUDA is unavailable")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("output directory must be empty to prevent artifact mixing")

    manifest_path = args.calibration_dir / "capture_manifest.json"
    capture_report = validate_capture_evidence(args.config, args.calibration_dir)
    capture = _load_manifest(manifest_path)
    shard_paths = sorted(args.calibration_dir.glob("*.safetensors"))
    if len(shard_paths) != config.calibration_sequences:
        raise RuntimeError(
            f"expected {config.calibration_sequences} calibration shards, found {len(shard_paths)}"
        )
    if capture.get("sequences") != len(shard_paths):
        raise RuntimeError("capture manifest sequence count differs from the shard set")

    source = ModelSignature.from_dict(capture["source_signature"])
    target = ModelSignature.from_dict(capture["target_signature"])
    if source.fingerprint != capture.get("source_fingerprint"):
        raise RuntimeError("source signature fingerprint does not match the capture manifest")
    if target.fingerprint != capture.get("target_fingerprint"):
        raise RuntimeError("target signature fingerprint does not match the capture manifest")
    source.validate_pair(target, require_matched_kv=config.fit.require_matched_kv)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    mapper = fit_mapper(
        calibration_shard_factory(args.calibration_dir), source, target, config.fit
    )
    elapsed_seconds = time.perf_counter() - started
    raw_config = json.loads(args.config.read_text(encoding="utf-8"))
    storage_dtype = args.storage_dtype or raw_config.get("artifact_storage_dtype", "bfloat16")
    mapper.save(args.output_dir, storage_dtype=storage_dtype)
    calibration_bytes = sum(path.stat().st_size for path in shard_paths)
    fit_run = {
        "schema_version": 1,
        "evidence_tier": raw_config.get("evidence_tier", "T2"),
        "code_revision": code_revision(),
        "config_sha256": _sha256(args.config),
        "capture_manifest_sha256": capture_report["manifest_sha256"],
        "calibration_shards": len(shard_paths),
        "calibration_bytes": calibration_bytes,
        "calibration_data_passes": plan.calibration_data_passes,
        "estimated_calibration_bytes_read": calibration_bytes
        * plan.calibration_data_passes,
        "elapsed_seconds": elapsed_seconds,
        "artifact_storage_dtype": storage_dtype,
        "fit_key_r2_mean": sum(mapper.fit_key_r2) / len(mapper.fit_key_r2),
        "fit_value_r2_mean": sum(mapper.fit_value_r2) / len(mapper.fit_value_r2),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_peak_memory_bytes": (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            ),
            "packages": package_versions(("safetensors", "numpy")),
        },
    }
    atomic_write_text(
        args.output_dir / "fit_run.json",
        json.dumps(fit_run, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(fit_run, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
