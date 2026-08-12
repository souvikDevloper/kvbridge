"""Capture aligned real-model cache shards for a configured partner-lab run.

Dry-run is the default. Pass --execute only on a host sized for both models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import torch

from kvbridge.fit import CalibrationPair
from kvbridge.huggingface import capture_cache, model_signature, tokenizer_fingerprint
from kvbridge.io import atomic_write_text, save_calibration_shard
from kvbridge.planning import ExperimentConfig, build_scale_plan
from kvbridge.provenance import code_revision, package_versions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bfloat16 was requested but this CUDA device does not support it")
        return torch.bfloat16
    if name != "auto":
        raise ValueError("model dtype must be auto, float16, bfloat16, or float32")
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _assert_architecture(actual: Any, expected: Any, role: str) -> None:
    fields = ("num_layers", "num_kv_heads", "head_dim", "attention_kind")
    mismatches = {
        field: (getattr(expected, field), getattr(actual, field))
        for field in fields
        if getattr(expected, field) != getattr(actual, field)
    }
    if mismatches:
        raise RuntimeError(f"{role} model differs from config: {mismatches}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/calibration"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--model-dtype", choices=("auto", "float16", "bfloat16", "float32"), default=None
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    print(json.dumps(build_scale_plan(config).to_dict(), indent=2))
    if not args.execute:
        print("Dry-run only. Re-run with --execute on an appropriately sized, approved host.")
        return 0

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    raw_config = json.loads(args.config.read_text(encoding="utf-8"))
    dataset_id = raw_config["calibration"].get("dataset")
    dataset_revision = raw_config["calibration"].get("dataset_revision")
    if not dataset_id:
        raise ValueError("config calibration.dataset is required for real-model capture")
    if not dataset_revision:
        raise ValueError("config calibration.dataset_revision must pin the calibration corpus")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise RuntimeError("output directory must be empty to prevent mixed calibration runs")

    source_tokenizer = AutoTokenizer.from_pretrained(
        config.source.model_id, revision=config.source.revision
    )
    target_tokenizer = AutoTokenizer.from_pretrained(
        config.target.model_id, revision=config.target.revision
    )
    source_tokenizer_hash = tokenizer_fingerprint(source_tokenizer)
    target_tokenizer_hash = tokenizer_fingerprint(target_tokenizer)
    if source_tokenizer_hash != target_tokenizer_hash:
        raise RuntimeError("source and target tokenizer fingerprints differ")

    dtype_name = args.model_dtype or raw_config.get("model_dtype", "auto")
    dtype = _model_dtype(dtype_name)
    load_kwargs = {
        "device_map": args.device_map,
        "torch_dtype": dtype,
        "attn_implementation": args.attn_implementation,
        "low_cpu_mem_usage": True,
    }
    source_model = AutoModelForCausalLM.from_pretrained(
        config.source.model_id, revision=config.source.revision, **load_kwargs
    ).eval()
    target_model = AutoModelForCausalLM.from_pretrained(
        config.target.model_id, revision=config.target.revision, **load_kwargs
    ).eval()
    actual_source = model_signature(source_model, source_tokenizer, revision=config.source.revision)
    actual_target = model_signature(target_model, target_tokenizer, revision=config.target.revision)
    _assert_architecture(actual_source, config.source, "source")
    _assert_architecture(actual_target, config.target, "target")
    actual_source.validate_pair(actual_target, require_matched_kv=config.fit.require_matched_kv)

    dataset = load_dataset(
        dataset_id, revision=dataset_revision, split=args.split, streaming=True
    )
    captured = 0
    shard_records: list[dict[str, Any]] = []
    for row_index, row in enumerate(dataset):
        text = row.get(args.text_field)
        if not isinstance(text, str) or not text.strip():
            continue
        source_tokens = source_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=config.calibration_tokens,
            add_special_tokens=False,
        )["input_ids"]
        if source_tokens.shape[1] < config.calibration_tokens:
            continue
        target_tokens = target_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=config.calibration_tokens,
            add_special_tokens=False,
        )["input_ids"]
        if not torch.equal(source_tokens, target_tokens):
            raise RuntimeError(f"token ids diverged despite equal fingerprints at row {row_index}")
        pair = CalibrationPair(
            capture_cache(source_model, source_tokens).detach(),
            capture_cache(target_model, target_tokens).detach(),
        )
        sequence_id = f"{dataset_id}:{args.split}:{row_index}"
        shard_path = save_calibration_shard(
            args.output_dir / f"{captured:05}.safetensors", pair, sequence_id=sequence_id
        )
        shard_records.append(
            {
                "name": shard_path.name,
                "bytes": shard_path.stat().st_size,
                "sha256": _sha256(shard_path),
                "sequence_id": sequence_id,
            }
        )
        captured += 1
        print(f"captured {captured}/{config.calibration_sequences}")
        if captured >= config.calibration_sequences:
            break
    if captured != config.calibration_sequences:
        raise RuntimeError(f"dataset ended after {captured} usable sequences")
    manifest = {
        "schema_version": 1,
        "config": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "code_revision": code_revision(),
        "dataset": dataset_id,
        "dataset_revision": dataset_revision,
        "split": args.split,
        "sequences": captured,
        "tokens": config.calibration_tokens,
        "source_fingerprint": actual_source.fingerprint,
        "target_fingerprint": actual_target.fingerprint,
        "source_signature": actual_source.to_dict(),
        "target_signature": actual_target.to_dict(),
        "tokenizer_hash": source_tokenizer_hash,
        "shards": shard_records,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "model_dtype": str(dtype).removeprefix("torch."),
            "attention_implementation": args.attn_implementation,
            "packages": package_versions(
                ("transformers", "datasets", "accelerate", "safetensors", "numpy")
            ),
        },
    }
    atomic_write_text(
        args.output_dir / "capture_manifest.json", json.dumps(manifest, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
