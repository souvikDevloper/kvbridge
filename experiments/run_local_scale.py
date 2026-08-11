"""Run reproducible CPU-only scale sweeps; no model downloads required."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

from kvbridge.config import FitConfig
from kvbridge.fit import fit_mapper
from kvbridge.synthetic import cache_r2, make_problem

CASES = [
    {
        "name": "micro",
        "source_layers": 3,
        "target_layers": 2,
        "heads": 2,
        "dim": 4,
        "tokens": 64,
        "pairs": 4,
    },
    {
        "name": "small",
        "source_layers": 5,
        "target_layers": 4,
        "heads": 2,
        "dim": 8,
        "tokens": 128,
        "pairs": 4,
    },
    {
        "name": "medium",
        "source_layers": 8,
        "target_layers": 6,
        "heads": 4,
        "dim": 16,
        "tokens": 256,
        "pairs": 3,
    },
]


def run_case(case: dict[str, int | str], repeats: int, warmup: int) -> dict[str, object]:
    problem = make_problem(
        seed=7,
        calibration_pairs=int(case["pairs"]),
        tokens=int(case["tokens"]),
        source_layers=int(case["source_layers"]),
        target_layers=int(case["target_layers"]),
        heads=int(case["heads"]),
        dim=int(case["dim"]),
    )
    started = time.perf_counter()
    mapper = fit_mapper(
        problem.calibration,
        problem.source,
        problem.target,
        FitConfig(
            top_k=1,
            ridge_alpha=0.01,
            accumulation_dtype="float64",
            target_layer_block_size=1,
            selection_target_layer_block_size=2,
        ),
    )
    fit_seconds = time.perf_counter() - started
    timings = []
    mapped = None
    for _ in range(warmup):
        mapper.map(problem.evaluation.source, target_rotary=problem.evaluation.target.rotary)
    for _ in range(repeats):
        begin = time.perf_counter()
        mapped = mapper.map(
            problem.evaluation.source, target_rotary=problem.evaluation.target.rotary
        )
        timings.append((time.perf_counter() - begin) * 1000)
    assert mapped is not None
    parameters = sum(
        tensor.numel()
        for tensor in (
            *mapper.key_weights,
            *mapper.value_weights,
            *mapper.key_biases,
            *mapper.value_biases,
        )
    )
    return {
        **case,
        "mapper_parameters": parameters,
        "fit_seconds": fit_seconds,
        "warmup_iterations": warmup,
        "timed_iterations": repeats,
        "map_ms_median": statistics.median(timings),
        "map_ms_p95": sorted(timings)[max(0, int(0.95 * len(timings)) - 1)],
        "tokens_per_second": int(case["tokens"]) / (statistics.median(timings) / 1000),
        "evaluation_r2": cache_r2(mapped, problem.evaluation.target),
        "selection_exact": mapper.selected_layers == [[layer] for layer in problem.true_layers],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [run_case(case, args.repeats, args.warmup) for case in CASES]
    document = {
        "schema_version": 1,
        "scope": "CPU synthetic structural validation; not a real-model quality claim",
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "cuda_available": torch.cuda.is_available(),
            "torch_threads": torch.get_num_threads(),
        },
        "cases": results,
    }
    json_path = args.output_dir / "local_scale_results.json"
    csv_path = args.output_dir / "local_scale_results.csv"
    json_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(document, indent=2))
    return (
        0
        if all(
            bool(case["selection_exact"]) and float(case["evaluation_r2"]) > 0.999
            for case in results
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
