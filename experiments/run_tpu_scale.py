"""Sequential-residency Qwen3 cache calibration and fitting on TPU v3/v6e.

Dry-run is the default.  ``--execute`` requires a PyTorch/XLA TPU runtime.
The source and target never reside on the accelerator together.  Only
stride-sampled content-space caches are retained in host memory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from kvbridge.cache import KVCache
from kvbridge.evidence import calibration_contract_sha256
from kvbridge.fit import CalibrationPair, fit_mapper
from kvbridge.huggingface import capture_cache, model_signature, tokenizer_fingerprint
from kvbridge.io import atomic_write_text
from kvbridge.planning import ExperimentConfig, build_scale_plan
from kvbridge.provenance import code_revision, package_versions
from kvbridge.tpu_evidence import validate_tpu_fit
from kvbridge.xla import (
    XLAContext,
    initialize_xla,
    shard_batch,
    sync_xla,
    wrap_model_for_fsdp,
    xla_runtime_manifest,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_architecture(actual: Any, expected: Any, role: str) -> None:
    fields = ("num_layers", "num_kv_heads", "head_dim", "attention_kind")
    mismatches = {
        field: (getattr(expected, field), getattr(actual, field))
        for field in fields
        if getattr(expected, field) != getattr(actual, field)
    }
    if mismatches:
        raise RuntimeError(f"{role} model differs from config: {mismatches}")


def _host_bytes(caches: list[KVCache]) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for cache in caches
        for tensor in (*cache.keys, *cache.values)
    )


def _sample_to_host(cache: KVCache, *, stride: int, logical_batch: int) -> KVCache:
    sampled = cache.to_content_space().sample_tokens(stride)
    return KVCache(
        [tensor[:logical_batch].detach().to("cpu") for tensor in sampled.keys],
        [tensor[:logical_batch].detach().to("cpu") for tensor in sampled.values],
        keys_are_content=True,
    )


def _padded_batch(
    rows: list[torch.Tensor], start: int, batch_size: int
) -> tuple[torch.Tensor, int]:
    logical = min(batch_size, len(rows) - start)
    batch = rows[start : start + logical]
    if not batch:
        raise IndexError("batch start is outside the token rows")
    batch += [batch[-1]] * (batch_size - logical)
    return torch.stack(batch), logical


def _capture_model(
    *,
    model_id: str,
    revision: str,
    tokenizer: Any,
    rows: list[torch.Tensor],
    batch_size: int,
    stride: int,
    attention_implementation: str,
    context: XLAContext,
    role: str,
) -> tuple[list[KVCache], Any]:
    from transformers import AutoModel  # type: ignore[import-not-found, unused-ignore]

    captured: list[KVCache] = []
    with tempfile.TemporaryDirectory(prefix=f"kvbridge-{role}-hf-") as cache_dir:
        model = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation=attention_implementation,
            low_cpu_mem_usage=True,
        ).eval()
        actual = model_signature(model, tokenizer, revision=revision)
        model = wrap_model_for_fsdp(model, context).eval()
        for start in range(0, len(rows), batch_size):
            batch, logical = _padded_batch(rows, start, batch_size)
            mask = torch.ones_like(batch)
            sharded_ids = shard_batch(batch, context)
            sharded_mask = shard_batch(mask, context)
            cache = capture_cache(model, sharded_ids, attention_mask=sharded_mask)
            captured.append(_sample_to_host(cache, stride=stride, logical_batch=logical))
            sync_xla()
            print(f"{role} capture {min(start + logical, len(rows))}/{len(rows)}", flush=True)
        del model
    sync_xla()
    gc.collect()
    return captured, actual


def _collect_tokens(
    *,
    tokenizer: Any,
    dataset_id: str,
    dataset_revision: str,
    split: str,
    text_field: str,
    sequences: int,
    tokens: int,
) -> tuple[list[torch.Tensor], list[int]]:
    from datasets import load_dataset  # type: ignore[import-not-found, unused-ignore]

    dataset = load_dataset(dataset_id, revision=dataset_revision, split=split, streaming=True)
    rows: list[torch.Tensor] = []
    row_indices: list[int] = []
    for row_index, row in enumerate(dataset):
        text = row.get(text_field)
        if not isinstance(text, str) or not text.strip():
            continue
        input_ids = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=tokens,
            add_special_tokens=False,
        )["input_ids"][0]
        if input_ids.numel() < tokens:
            continue
        rows.append(input_ids.contiguous())
        row_indices.append(row_index)
        if len(rows) >= sequences:
            break
    if len(rows) != sequences:
        raise RuntimeError(f"dataset ended after {len(rows)} usable sequences")
    return rows, row_indices


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tpu-scale"))
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-xla-devices", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    plan = build_scale_plan(config)
    tpu_plan = {
        "scale_plan": plan.to_dict(),
        "execution": {
            "models_co_resident": False,
            "sampled_cache_storage": "host-memory",
            "batch_size": args.batch_size,
            "required_xla_devices": args.min_xla_devices,
            "checkpoint_granularity": "selection-block-and-target-layer",
        },
    }
    print(json.dumps(tpu_plan, indent=2))
    if not args.execute:
        print("Dry-run only. Select a TPU runtime and re-run with --execute.")
        return 0
    if config.fit.accumulation_device != "xla":
        raise ValueError("TPU scale config must set fit.accumulation_device to xla")
    if args.batch_size <= 0 or args.batch_size % args.min_xla_devices:
        raise ValueError("batch size must be positive and divisible by --min-xla-devices")

    calibration = raw["calibration"]
    dataset_id = calibration.get("dataset")
    dataset_revision = calibration.get("dataset_revision")
    split = calibration.get("split", "train")
    if not dataset_id or not dataset_revision:
        raise ValueError("calibration dataset and immutable revision are required")
    attention_implementation = raw.get("attention_implementation", "eager")
    storage_dtype = raw.get("artifact_storage_dtype", "bfloat16")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = args.output_dir / "artifact"
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        if not args.resume:
            raise RuntimeError("artifact output already exists and resume is disabled")
        report = validate_tpu_fit(args.config, args.output_dir)
        print("Reusing validated complete TPU fit:")
        print(json.dumps(report, indent=2))
        return 0

    context = initialize_xla(
        min_devices=args.min_xla_devices,
        compilation_cache=args.output_dir / "xla-compile-cache",
    )
    from transformers import AutoTokenizer  # type: ignore[import-not-found, unused-ignore]

    with tempfile.TemporaryDirectory(prefix="kvbridge-tokenizers-hf-") as tokenizer_cache:
        source_tokenizer = AutoTokenizer.from_pretrained(
            config.source.model_id,
            revision=config.source.revision,
            cache_dir=tokenizer_cache,
        )
        target_tokenizer = AutoTokenizer.from_pretrained(
            config.target.model_id,
            revision=config.target.revision,
            cache_dir=tokenizer_cache,
        )
    source_hash = tokenizer_fingerprint(source_tokenizer)
    target_hash = tokenizer_fingerprint(target_tokenizer)
    if source_hash != target_hash:
        raise RuntimeError("source and target tokenizer fingerprints differ")
    token_rows, row_indices = _collect_tokens(
        tokenizer=source_tokenizer,
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        split=split,
        text_field=args.text_field,
        sequences=config.calibration_sequences,
        tokens=config.calibration_tokens,
    )

    started = time.perf_counter()
    source_caches, actual_source = _capture_model(
        model_id=config.source.model_id,
        revision=config.source.revision,
        tokenizer=source_tokenizer,
        rows=token_rows,
        batch_size=args.batch_size,
        stride=config.token_stride,
        attention_implementation=attention_implementation,
        context=context,
        role="source",
    )
    _assert_architecture(actual_source, config.source, "source")
    target_caches, actual_target = _capture_model(
        model_id=config.target.model_id,
        revision=config.target.revision,
        tokenizer=target_tokenizer,
        rows=token_rows,
        batch_size=args.batch_size,
        stride=config.token_stride,
        attention_implementation=attention_implementation,
        context=context,
        role="target",
    )
    _assert_architecture(actual_target, config.target, "target")
    actual_source.validate_pair(
        actual_target, require_matched_kv=config.fit.require_matched_kv
    )
    if len(source_caches) != len(target_caches):
        raise RuntimeError("source and target capture batch counts differ")
    pairs = [
        CalibrationPair(source, target, config.token_stride)
        for source, target in zip(source_caches, target_caches, strict=True)
    ]
    observations = sum(pair.source.shape[1] * pair.source.shape[3] for pair in pairs)
    if observations != plan.observations:
        raise RuntimeError(
            f"captured {observations} observations but the plan requires {plan.observations}"
        )

    mapper = fit_mapper(
        pairs,
        actual_source,
        actual_target,
        config.fit,
        checkpoint_dir=args.output_dir / "checkpoints",
        resume=args.resume,
    )
    mapper.save(artifact_dir, storage_dtype=storage_dtype)
    elapsed_seconds = time.perf_counter() - started
    manifest = {
        "format": "kvbridge-tpu-scale-run",
        "schema_version": 1,
        "evidence_tier": raw.get("evidence_tier", "T3"),
        "status": "fit-complete-evaluation-pending",
        "config": str(args.config),
        "config_sha256": _sha256(args.config),
        "calibration_contract_sha256": calibration_contract_sha256(args.config),
        "artifact_manifest_sha256": _sha256(artifact_dir / "manifest.json"),
        "code_revision": code_revision(),
        "dataset": dataset_id,
        "dataset_revision": dataset_revision,
        "split": split,
        "dataset_row_indices": row_indices,
        "sequences": config.calibration_sequences,
        "tokens_per_sequence": config.calibration_tokens,
        "stride": config.token_stride,
        "observations": observations,
        "source_fingerprint": actual_source.fingerprint,
        "target_fingerprint": actual_target.fingerprint,
        "tokenizer_hash": source_hash,
        "source_sampled_cache_bytes": _host_bytes(source_caches),
        "target_sampled_cache_bytes": _host_bytes(target_caches),
        "elapsed_seconds": elapsed_seconds,
        "fit_key_r2_mean": sum(mapper.fit_key_r2) / len(mapper.fit_key_r2),
        "fit_value_r2_mean": sum(mapper.fit_value_r2) / len(mapper.fit_value_r2),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "xla": xla_runtime_manifest(context),
            "model_dtype": "bfloat16",
            "attention_implementation": attention_implementation,
            "packages": package_versions(
                ("transformers", "datasets", "safetensors", "numpy")
            ),
        },
    }
    atomic_write_text(
        args.output_dir / "fit_run.json",
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
