"""Evaluate a fitted mapper with real target queries and shadow-prefill logits.

Dry-run is the default. Execution writes a provenance-rich raw JSON artifact.
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

from kvbridge.huggingface import (
    capture_cache,
    capture_cache_with_queries,
    model_signature,
    suffix_logits_from_cache,
    tokenizer_fingerprint,
)
from kvbridge.mapper import CrossModelKVMapper
from kvbridge.metrics import attention_output_cosine, logit_kl_divergence
from kvbridge.planning import ExperimentConfig, build_scale_plan
from kvbridge.provenance import code_revision
from kvbridge.synthetic import cache_r2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(callable_: Any) -> tuple[Any, float]:
    _sync()
    started = time.perf_counter()
    result = callable_()
    _sync()
    return result, (time.perf_counter() - started) * 1000


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty list")
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _model_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("configured bfloat16 model dtype is unsupported on this GPU")
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError("model_dtype must be float16, bfloat16, or float32")


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    attention = [float(case["attention_cosine_mean"]) for case in cases]
    kl = [float(case["logit_kl"]) for case in cases]
    transfer = [float(case["transfer_ms"]) for case in cases]
    target_prefill = [float(case["target_prefix_prefill_ms"]) for case in cases]
    return {
        "sequences": len(cases),
        "cache_r2_mean": sum(float(case["cache_r2"]) for case in cases) / len(cases),
        "attention_cosine_mean": sum(attention) / len(attention),
        "attention_cosine_min": min(
            float(case["attention_cosine_min"]) for case in cases
        ),
        "logit_kl_mean": sum(kl) / len(kl),
        "logit_kl_p95": _percentile(kl, 0.95),
        "next_token_agreement": sum(bool(case["next_token_agreement"]) for case in cases)
        / len(cases),
        "transfer_ms_median": _percentile(transfer, 0.50),
        "transfer_ms_p95": _percentile(transfer, 0.95),
        "target_prefix_prefill_ms_median": _percentile(target_prefill, 0.50),
        "prefill_to_transfer_speed_ratio_median": _percentile(
            [baseline / mapped for baseline, mapped in zip(target_prefill, transfer, strict=True)],
            0.50,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/hf-mapper"))
    parser.add_argument("--output", type=Path, default=Path("results/t2_hf_evaluation.json"))
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    print(json.dumps(build_scale_plan(config).to_dict(), indent=2))
    if not args.execute:
        print("Dry-run only. Re-run with --execute after fitting the real-model artifact.")
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("T2 evaluation requires CUDA")
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite existing evaluation: {args.output}")

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    raw = json.loads(args.config.read_text(encoding="utf-8"))
    calibration = raw["calibration"]
    evaluation = raw["evaluation"]
    dtype = _model_dtype(raw.get("model_dtype", "float16"))
    mapper = CrossModelKVMapper.load(args.artifact_dir)
    source_tokenizer = AutoTokenizer.from_pretrained(
        config.source.model_id, revision=config.source.revision
    )
    target_tokenizer = AutoTokenizer.from_pretrained(
        config.target.model_id, revision=config.target.revision
    )
    if tokenizer_fingerprint(source_tokenizer) != tokenizer_fingerprint(target_tokenizer):
        raise RuntimeError("evaluation tokenizers differ")
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
    target_device = target_model.get_input_embeddings().weight.device
    mapper.to(target_device, dtype=torch.float32)
    actual_source = model_signature(
        source_model, source_tokenizer, revision=config.source.revision
    )
    actual_target = model_signature(
        target_model, target_tokenizer, revision=config.target.revision
    )
    if actual_source.fingerprint != mapper.source_signature.fingerprint:
        raise RuntimeError("loaded source model differs from the mapper artifact")
    if actual_target.fingerprint != mapper.target_signature.fingerprint:
        raise RuntimeError("loaded target model differs from the mapper artifact")

    dataset = load_dataset(
        calibration["dataset"],
        revision=calibration["dataset_revision"],
        split=calibration.get("split", "train"),
        streaming=True,
    )
    cases: list[dict[str, Any]] = []
    desired = int(evaluation["sequences"])
    max_tokens = int(evaluation["tokens"])
    suffix_tokens = int(evaluation.get("max_suffix_tokens", 1))
    dataset_skip = int(evaluation.get("dataset_skip", 0))
    for row_index, row in enumerate(dataset):
        if row_index < dataset_skip:
            continue
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        source_ids = source_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens,
            add_special_tokens=False,
        )["input_ids"]
        if source_ids.shape[1] < max_tokens or source_ids.shape[1] <= suffix_tokens:
            continue
        target_ids = target_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens,
            add_special_tokens=False,
        )["input_ids"]
        if not torch.equal(source_ids, target_ids):
            raise RuntimeError(f"token ids diverged at evaluation row {row_index}")
        prefix, suffix = source_ids[:, :-suffix_tokens], source_ids[:, -suffix_tokens:]
        source_cache, source_prefill_ms = _timed(
            lambda current_prefix=prefix: capture_cache(source_model, current_prefix)
        )
        reference, target_prefix_prefill_ms = _timed(
            lambda current_prefix=prefix: capture_cache_with_queries(
                target_model, current_prefix
            )
        )
        mapped, transfer_ms = _timed(
            lambda current_cache=source_cache, current_rotary=reference.cache.rotary: mapper.map(
                current_cache, target_rotary=current_rotary
            )
        )
        attention = attention_output_cosine(
            reference.queries, mapped, reference.cache, causal=True
        )
        reference_full, target_full_prefill_ms = _timed(
            lambda current_ids=source_ids: target_model(
                input_ids=current_ids.to(target_device),
                use_cache=False,
                return_dict=True,
            ).logits[:, -suffix_tokens:]
        )
        candidate_logits, suffix_decode_ms = _timed(
            lambda current_cache=mapped, current_suffix=suffix: suffix_logits_from_cache(
                target_model, current_cache, current_suffix
            )
        )
        kl = logit_kl_divergence(candidate_logits, reference_full)
        cases.append(
            {
                "sequence_id": f"{calibration['dataset']}:{row_index}",
                "input_sha256": hashlib.sha256(source_ids.numpy().tobytes()).hexdigest(),
                "tokens": int(source_ids.shape[1]),
                "cache_r2": cache_r2(mapped, reference.cache),
                "attention_cosine_mean": attention.mean,
                "attention_cosine_min": attention.minimum,
                "attention_cosine_per_layer": attention.per_layer,
                "logit_kl": kl,
                "next_token_agreement": bool(
                    torch.equal(
                        candidate_logits[:, -1].argmax(-1),
                        reference_full[:, -1].argmax(-1),
                    )
                ),
                "source_prefill_ms": source_prefill_ms,
                "transfer_ms": transfer_ms,
                "suffix_decode_ms": suffix_decode_ms,
                "target_prefix_prefill_ms": target_prefix_prefill_ms,
                "target_full_prefill_ms": target_full_prefill_ms,
            }
        )
        print(f"evaluated {len(cases)}/{desired}: attention={attention.mean:.6f}, KL={kl:.6f}")
        if len(cases) >= desired:
            break
    if len(cases) != desired:
        raise RuntimeError(f"dataset produced only {len(cases)} usable evaluation sequences")

    summary = _summary(cases)
    summary["attention_gate_passed"] = (
        summary["attention_cosine_min"] >= float(evaluation["attention_cosine_floor"])
    )
    summary["logit_kl_gate_passed"] = (
        summary["logit_kl_p95"] <= float(evaluation["logit_kl_ceiling"])
    )
    summary["all_quality_gates_passed"] = bool(
        summary["attention_gate_passed"] and summary["logit_kl_gate_passed"]
    )
    payload = {
        "schema_version": 1,
        "evidence_tier": raw.get("evidence_tier", "T2"),
        "scope": "real-model single-GPU integration; not a paper-scale reproduction",
        "code_revision": code_revision(),
        "config_sha256": _sha256(args.config),
        "artifact_manifest_sha256": _sha256(args.artifact_dir / "manifest.json"),
        "dataset": calibration["dataset"],
        "dataset_revision": calibration["dataset_revision"],
        "source_fingerprint": mapper.source_signature.fingerprint,
        "target_fingerprint": mapper.target_signature.fingerprint,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
            "cuda_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "cuda_peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "model_dtype": str(dtype).removeprefix("torch."),
            "attention_implementation": args.attn_implementation,
        },
        "thresholds": {
            "attention_cosine_floor": evaluation["attention_cosine_floor"],
            "logit_kl_ceiling": evaluation["logit_kl_ceiling"],
        },
        "summary": summary,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
