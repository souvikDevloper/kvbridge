"""Held-out TPU evaluation for a paper-scale KVBridge mapper.

Dry-run is the default. Source and target models are loaded sequentially. The
result records batched cache R², target-query attention cosine, short-suffix
logit KL, next-token agreement, and hardware-local timings.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from kvbridge.cache import KVCache
from kvbridge.huggingface import (
    capture_cache,
    capture_cache_with_queries,
    model_signature,
    suffix_logits_from_cache,
    tokenizer_fingerprint,
)
from kvbridge.io import atomic_write_text
from kvbridge.mapper import CrossModelKVMapper
from kvbridge.metrics import attention_output_cosine, logit_kl_divergence
from kvbridge.planning import ExperimentConfig, build_scale_plan
from kvbridge.provenance import code_revision, package_versions
from kvbridge.statistics import bootstrap_mean_interval
from kvbridge.synthetic import cache_r2
from kvbridge.xla import (
    XLAContext,
    initialize_xla,
    shard_batch,
    shard_cache,
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


def _timed(callable_: Any) -> tuple[Any, float]:
    sync_xla()
    started = time.perf_counter()
    result = callable_()
    sync_xla()
    return result, (time.perf_counter() - started) * 1000


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty list")
    return ordered[round((len(ordered) - 1) * fraction)]


def _collect_evaluation_rows(
    *,
    tokenizer: Any,
    dataset_id: str,
    dataset_revision: str,
    split: str,
    skip: int,
    sequences: int,
    tokens: int,
) -> tuple[list[torch.Tensor], list[int]]:
    from datasets import load_dataset  # type: ignore[import-not-found, unused-ignore]

    dataset = load_dataset(dataset_id, revision=dataset_revision, split=split, streaming=True)
    rows: list[torch.Tensor] = []
    indices: list[int] = []
    for row_index, row in enumerate(dataset):
        if row_index < skip:
            continue
        text = row.get("text")
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
        indices.append(row_index)
        if len(rows) == sequences:
            break
    if len(rows) != sequences:
        raise RuntimeError(f"dataset produced only {len(rows)} usable evaluation sequences")
    return rows, indices


def _capture_source_batches(
    *,
    config: ExperimentConfig,
    tokenizer: Any,
    prefixes: list[torch.Tensor],
    batch_size: int,
    attention_implementation: str,
    context: XLAContext,
) -> tuple[list[KVCache], Any]:
    from transformers import AutoModel  # type: ignore[import-not-found, unused-ignore]

    caches: list[KVCache] = []
    with tempfile.TemporaryDirectory(prefix="kvbridge-eval-source-hf-") as cache_dir:
        model = AutoModel.from_pretrained(
            config.source.model_id,
            revision=config.source.revision,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation=attention_implementation,
            low_cpu_mem_usage=True,
        ).eval()
        actual = model_signature(model, tokenizer, revision=config.source.revision)
        model = wrap_model_for_fsdp(model, context).eval()
        for start in range(0, len(prefixes), batch_size):
            batch = torch.stack(prefixes[start : start + batch_size])
            mask = torch.ones_like(batch)
            captured = capture_cache(
                model,
                shard_batch(batch, context),
                attention_mask=shard_batch(mask, context),
            ).to_content_space()
            caches.append(
                KVCache(
                    [tensor.detach().to("cpu") for tensor in captured.keys],
                    [tensor.detach().to("cpu") for tensor in captured.values],
                    keys_are_content=True,
                )
            )
            sync_xla()
            print(f"source eval capture {start + batch_size}/{len(prefixes)}", flush=True)
        del model
    sync_xla()
    gc.collect()
    return caches, actual


def _summary(cases: list[dict[str, Any]], sequences: int) -> dict[str, Any]:
    attention = [float(case["attention_cosine_mean"]) for case in cases]
    kl = [float(case["logit_kl"]) for case in cases]
    agreement = [float(case["next_token_agreement"]) for case in cases]
    transfer = [float(case["transfer_ms"]) for case in cases]
    prefill = [float(case["target_prefix_prefill_ms"]) for case in cases]
    return {
        "sequences": sequences,
        "batches": len(cases),
        "cache_r2_mean": sum(float(case["cache_r2"]) for case in cases) / len(cases),
        "attention_cosine_mean": sum(attention) / len(attention),
        "attention_cosine_min": min(float(case["attention_cosine_min"]) for case in cases),
        "logit_kl_mean": sum(kl) / len(kl),
        "logit_kl_p95": _percentile(kl, 0.95),
        "next_token_agreement": sum(agreement) / len(agreement),
        "transfer_ms_batch_median": _percentile(transfer, 0.50),
        "target_prefix_prefill_ms_batch_median": _percentile(prefill, 0.50),
        "prefill_to_transfer_ratio_batch_median": _percentile(
            [base / mapped for base, mapped in zip(prefill, transfer, strict=True)], 0.50
        ),
        "confidence_intervals": {
            "attention_cosine_batch_mean": bootstrap_mean_interval(
                attention, seed=202
            ).to_dict(),
            "logit_kl_batch_mean": bootstrap_mean_interval(kl, seed=203).to_dict(),
            "next_token_agreement_batch_mean": bootstrap_mean_interval(
                agreement, seed=204
            ).to_dict(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-xla-devices", type=int, default=8)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(build_scale_plan(config).to_dict(), indent=2))
    if not args.execute:
        print("Dry-run only. Re-run with --execute after the TPU mapper fit completes.")
        return 0
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite existing evaluation: {args.output}")
    evaluation = raw.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("config evaluation object is required")
    desired = int(evaluation["sequences"])
    if desired % args.batch_size:
        raise ValueError("evaluation sequences must be divisible by TPU batch size")
    if args.batch_size % args.min_xla_devices:
        raise ValueError("batch size must be divisible by the XLA device count")

    context = initialize_xla(
        min_devices=args.min_xla_devices,
        compilation_cache=args.output.parent / "xla-eval-compile-cache",
    )
    from transformers import (  # type: ignore[import-not-found, unused-ignore]
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    calibration = raw["calibration"]
    attention_implementation = raw.get("attention_implementation", "eager")
    with tempfile.TemporaryDirectory(prefix="kvbridge-eval-tokenizers-hf-") as cache_dir:
        source_tokenizer = AutoTokenizer.from_pretrained(
            config.source.model_id, revision=config.source.revision, cache_dir=cache_dir
        )
        target_tokenizer = AutoTokenizer.from_pretrained(
            config.target.model_id, revision=config.target.revision, cache_dir=cache_dir
        )
    if tokenizer_fingerprint(source_tokenizer) != tokenizer_fingerprint(target_tokenizer):
        raise RuntimeError("source and target tokenizer fingerprints differ")
    rows, row_indices = _collect_evaluation_rows(
        tokenizer=source_tokenizer,
        dataset_id=calibration["dataset"],
        dataset_revision=calibration["dataset_revision"],
        split=calibration.get("split", "train"),
        skip=int(evaluation.get("dataset_skip", 0)),
        sequences=desired,
        tokens=int(evaluation["tokens"]),
    )
    suffix_tokens = int(evaluation.get("max_suffix_tokens", 1))
    prefixes = [row[:-suffix_tokens] for row in rows]
    source_caches, actual_source = _capture_source_batches(
        config=config,
        tokenizer=source_tokenizer,
        prefixes=prefixes,
        batch_size=args.batch_size,
        attention_implementation=attention_implementation,
        context=context,
    )

    mapper = CrossModelKVMapper.load(args.artifact_dir)
    if actual_source.fingerprint != mapper.source_signature.fingerprint:
        raise RuntimeError("source model differs from the mapper artifact")
    mapper.to(context.device, dtype=torch.bfloat16)
    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="kvbridge-eval-target-hf-") as cache_dir:
        target = AutoModelForCausalLM.from_pretrained(
            config.target.model_id,
            revision=config.target.revision,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation=attention_implementation,
            low_cpu_mem_usage=True,
        ).eval()
        actual_target = model_signature(target, target_tokenizer, revision=config.target.revision)
        if actual_target.fingerprint != mapper.target_signature.fingerprint:
            raise RuntimeError("target model differs from the mapper artifact")
        target = wrap_model_for_fsdp(target, context).eval()

        def evaluate_batch(batch_index: int) -> dict[str, Any]:
            start = batch_index * args.batch_size
            full_ids = torch.stack(rows[start : start + args.batch_size])
            prefix_ids = full_ids[:, :-suffix_tokens]
            suffix_ids = full_ids[:, -suffix_tokens:]
            prefix_mask = torch.ones_like(prefix_ids)
            reference, target_prefill_ms = _timed(
                lambda: capture_cache_with_queries(
                    target,
                    shard_batch(prefix_ids, context),
                    attention_mask=shard_batch(prefix_mask, context),
                    retain_logits=False,
                )
            )
            source_cache = shard_cache(source_caches[batch_index], context)
            mapped, transfer_ms = _timed(
                lambda: mapper.map(source_cache, target_rotary=reference.cache.rotary)
            )
            attention = attention_output_cosine(
                reference.queries, mapped, reference.cache, causal=True
            )
            sharded_full = shard_batch(full_ids, context)
            full_mask = shard_batch(torch.ones_like(full_ids), context)
            reference_logits, full_prefill_ms = _timed(
                lambda: target(
                    input_ids=sharded_full,
                    attention_mask=full_mask,
                    use_cache=False,
                    return_dict=True,
                ).logits[:, -suffix_tokens:]
            )
            candidate_logits, suffix_decode_ms = _timed(
                lambda: suffix_logits_from_cache(
                    target,
                    mapped,
                    suffix_ids,
                    batch_sharder=lambda tensor: shard_batch(tensor, context),
                )
            )
            agreement = float(
                (
                    candidate_logits[:, -1].argmax(-1)
                    == reference_logits[:, -1].argmax(-1)
                )
                .float()
                .mean()
                .item()
            )
            return {
                "batch_index": batch_index,
                "sequence_ids": [
                    f"{calibration['dataset']}:{index}"
                    for index in row_indices[start : start + args.batch_size]
                ],
                "input_sha256": hashlib.sha256(full_ids.numpy().tobytes()).hexdigest(),
                "batch_size": args.batch_size,
                "tokens": int(full_ids.shape[1]),
                "cache_r2": cache_r2(mapped, reference.cache),
                "attention_cosine_mean": attention.mean,
                "attention_cosine_min": attention.minimum,
                "attention_cosine_per_layer": attention.per_layer,
                "logit_kl": logit_kl_divergence(candidate_logits, reference_logits),
                "next_token_agreement": agreement,
                "transfer_ms": transfer_ms,
                "target_prefix_prefill_ms": target_prefill_ms,
                "target_full_prefill_ms": full_prefill_ms,
                "suffix_decode_ms": suffix_decode_ms,
            }

        batch_count = desired // args.batch_size
        for _ in range(args.warmup_batches):
            evaluate_batch(0)
        for batch_index in range(batch_count):
            case = evaluate_batch(batch_index)
            numeric = [
                value
                for name, value in case.items()
                if name
                in {
                    "cache_r2",
                    "attention_cosine_mean",
                    "attention_cosine_min",
                    "logit_kl",
                    "next_token_agreement",
                    "transfer_ms",
                    "target_prefix_prefill_ms",
                    "target_full_prefill_ms",
                    "suffix_decode_ms",
                }
            ]
            if not all(math.isfinite(float(value)) for value in numeric):
                raise RuntimeError(f"non-finite metric in evaluation batch {batch_index}")
            cases.append(case)
            print(
                f"evaluated batch {batch_index + 1}/{batch_count}: "
                f"attention={case['attention_cosine_mean']:.6f}, "
                f"KL={case['logit_kl']:.6f}",
                flush=True,
            )
    summary = _summary(cases, desired)
    summary["attention_gate_passed"] = bool(
        summary["attention_cosine_min"] >= float(evaluation["attention_cosine_floor"])
    )
    summary["logit_kl_gate_passed"] = bool(
        summary["logit_kl_p95"] <= float(evaluation["logit_kl_ceiling"])
    )
    summary["all_quality_gates_passed"] = bool(
        summary["attention_gate_passed"] and summary["logit_kl_gate_passed"]
    )
    payload = {
        "format": "kvbridge-tpu-evaluation",
        "schema_version": 1,
        "evidence_tier": raw.get("evidence_tier", "T3"),
        "scope": "held-out batched TPU integration; downstream benchmark suite pending",
        "code_revision": code_revision(),
        "config_sha256": _sha256(args.config),
        "artifact_manifest_sha256": _sha256(args.artifact_dir / "manifest.json"),
        "dataset": calibration["dataset"],
        "dataset_revision": calibration["dataset_revision"],
        "source_fingerprint": mapper.source_signature.fingerprint,
        "target_fingerprint": mapper.target_signature.fingerprint,
        "timing_policy": {
            "warmup_batches": args.warmup_batches,
            "batch_size": args.batch_size,
            "cache_host_to_device_excluded": True,
            "hardware_local_only": True,
        },
        "thresholds": {
            "attention_cosine_floor": evaluation["attention_cosine_floor"],
            "logit_kl_ceiling": evaluation["logit_kl_ceiling"],
        },
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
        "summary": summary,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        args.output, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0 if summary["all_quality_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
