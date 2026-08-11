"""Capture aligned real-model cache shards for a configured partner-lab run.

Dry-run is the default. Pass --execute only on a host sized for both models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from kvbridge.fit import CalibrationPair
from kvbridge.huggingface import capture_cache, model_signature, tokenizer_fingerprint
from kvbridge.io import save_calibration_shard
from kvbridge.planning import ExperimentConfig, build_scale_plan


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
    parser.add_argument("--attn-implementation", default="flash_attention_2")
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
    if not dataset_id:
        raise ValueError("config calibration.dataset is required for real-model capture")
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

    load_kwargs = {
        "device_map": args.device_map,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": args.attn_implementation,
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

    dataset = load_dataset(dataset_id, split=args.split, streaming=True)
    captured = 0
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
        save_calibration_shard(
            args.output_dir / f"{captured:05}.safetensors",
            pair,
            sequence_id=f"{dataset_id}:{args.split}:{row_index}",
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
        "dataset": dataset_id,
        "split": args.split,
        "sequences": captured,
        "tokens": config.calibration_tokens,
        "source_fingerprint": actual_source.fingerprint,
        "target_fingerprint": actual_target.fingerprint,
        "tokenizer_hash": source_tokenizer_hash,
    }
    (args.output_dir / "capture_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
