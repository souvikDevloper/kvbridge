"""Measure FP32 versus BF16 mapper storage on a deterministic CPU problem."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from kvbridge.mapper import CrossModelKVMapper
from kvbridge.metrics import attention_output_cosine
from kvbridge.synthetic import cache_r2, fit_demo, make_problem


@dataclass(frozen=True, slots=True)
class PrecisionResult:
    storage_dtype: str
    file_bytes: int
    tensor_bytes: int
    size_ratio_to_fp32: float
    cache_r2: float
    attention_output_cosine_mean: float
    attention_output_cosine_min: float
    attention_cosine_delta_from_fp32: float


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path, default=Path("results/artifact_precision.json"))
    parser.add_argument("--csv-output", type=Path, default=Path("results/artifact_precision.csv"))
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()

    problem = make_problem(
        seed=args.seed,
        calibration_pairs=4,
        tokens=128,
        source_layers=8,
        target_layers=6,
        heads=4,
        dim=16,
    )
    mapper = fit_demo(problem)
    generator = torch.Generator().manual_seed(args.seed + 1)
    queries = [
        torch.randn((1, 8, 128, 16), generator=generator)
        for _ in range(problem.target.num_layers)
    ]
    raw: list[dict[str, float | int | str]] = []
    with tempfile.TemporaryDirectory(prefix="kvbridge-precision-") as temporary:
        root = Path(temporary)
        for storage_dtype in ("float32", "bfloat16"):
            artifact = mapper.save(root / storage_dtype, storage_dtype=storage_dtype)
            loaded = CrossModelKVMapper.load(artifact)
            mapped = loaded.map(
                problem.evaluation.source,
                target_rotary=problem.evaluation.target.rotary,
            )
            attention = attention_output_cosine(
                queries,
                mapped,
                problem.evaluation.target,
                causal=True,
            )
            manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
            raw.append(
                {
                    "storage_dtype": storage_dtype,
                    "file_bytes": (artifact / "mapper.safetensors").stat().st_size,
                    "tensor_bytes": int(manifest["tensor_bytes"]),
                    "cache_r2": cache_r2(mapped, problem.evaluation.target),
                    "attention_output_cosine_mean": attention.mean,
                    "attention_output_cosine_min": attention.minimum,
                }
            )
    fp32 = raw[0]
    results = [
        PrecisionResult(
            storage_dtype=str(item["storage_dtype"]),
            file_bytes=int(item["file_bytes"]),
            tensor_bytes=int(item["tensor_bytes"]),
            size_ratio_to_fp32=float(item["file_bytes"]) / float(fp32["file_bytes"]),
            cache_r2=float(item["cache_r2"]),
            attention_output_cosine_mean=float(item["attention_output_cosine_mean"]),
            attention_output_cosine_min=float(item["attention_output_cosine_min"]),
            attention_cosine_delta_from_fp32=float(item["attention_output_cosine_mean"])
            - float(fp32["attention_output_cosine_mean"]),
        )
        for item in raw
    ]
    payload = {
        "schema_version": 1,
        "scope": (
            "CPU synthetic artifact-precision validation; not a real-model quality or GPU claim"
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "cuda_available": torch.cuda.is_available(),
        },
        "problem": {
            "source_layers": 8,
            "target_layers": 6,
            "kv_heads": 4,
            "query_heads": 8,
            "head_dim": 16,
            "tokens": 128,
            "calibration_pairs": 4,
        },
        "variants": [asdict(result) for result in results],
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with args.csv_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
