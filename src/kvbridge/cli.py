"""Dependency-light command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from kvbridge.mapper import CrossModelKVMapper
from kvbridge.metrics import attention_output_cosine
from kvbridge.planning import ExperimentConfig, build_scale_plan
from kvbridge.synthetic import cache_r2, fit_demo, make_problem


def _demo(args: argparse.Namespace) -> int:
    problem = make_problem(
        seed=args.seed, calibration_pairs=args.calibration_pairs, tokens=args.tokens
    )
    mapper = fit_demo(problem)
    mapped, report = mapper.transfer(
        problem.evaluation.source,
        target_rotary=problem.evaluation.target.rotary,
    )
    score = cache_r2(mapped, problem.evaluation.target)
    generator = torch.Generator().manual_seed(args.seed + 1)
    queries = [
        torch.randn(
            (1, problem.target.num_kv_heads, args.tokens, problem.target.head_dim),
            generator=generator,
        )
        for _ in range(problem.target.num_layers)
    ]
    attention = attention_output_cosine(
        queries,
        mapped,
        problem.evaluation.target,
        causal=True,
    )
    if args.output is not None:
        mapper.save(
            args.output,
            overwrite=args.overwrite,
            storage_dtype=args.storage_dtype,
        )
    payload = {
        "status": "ok" if score >= args.min_r2 else "failed",
        "evaluation_r2": round(score, 8),
        "attention_output_cosine_mean": round(attention.mean, 8),
        "attention_output_cosine_min": round(attention.minimum, 8),
        "selected_layers": mapper.selected_layers,
        "expected_layers": problem.true_layers,
        "fit_key_r2": [round(value, 8) for value in mapper.fit_key_r2],
        "fit_value_r2": [round(value, 8) for value in mapper.fit_value_r2],
        "tokens": report.tokens,
        "transfer_ms": round(report.elapsed_ms, 3),
        "artifact": str(Path(args.output).resolve()) if args.output else None,
        "artifact_storage_dtype": args.storage_dtype if args.output else None,
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "ok" else 1


def _inspect(args: argparse.Namespace) -> int:
    mapper = CrossModelKVMapper.load(args.artifact)
    payload = {
        "source": mapper.source_signature.to_dict(),
        "target": mapper.target_signature.to_dict(),
        "config": mapper.config.to_dict(),
        "selected_layers": mapper.selected_layers,
        "fit_key_r2": mapper.fit_key_r2,
        "fit_value_r2": mapper.fit_value_r2,
        "storage_dtype": mapper.storage_dtype,
    }
    print(json.dumps(payload, indent=2))
    return 0


def _plan(args: argparse.Namespace) -> int:
    config = ExperimentConfig.load(args.config)
    print(json.dumps(build_scale_plan(config).to_dict(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kvbridge", description="Cross-model KV-cache transfer via closed-form ridge mapping"
    )
    parser.add_argument("--version", action="version", version="kvbridge 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="fit and validate a deterministic no-download demo")
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--calibration-pairs", type=int, default=6)
    demo.add_argument("--tokens", type=int, default=24)
    demo.add_argument("--min-r2", type=float, default=0.99)
    demo.add_argument("--output", type=Path)
    demo.add_argument("--overwrite", action="store_true")
    demo.add_argument("--storage-dtype", choices=("float32", "bfloat16"), default="float32")
    demo.set_defaults(handler=_demo)
    inspect = commands.add_parser("inspect", help="verify and display a mapper artifact")
    inspect.add_argument("artifact", type=Path)
    inspect.set_defaults(handler=_inspect)
    plan = commands.add_parser("plan", help="estimate mapper, cache, and fit memory from a config")
    plan.add_argument("config", type=Path)
    plan.set_defaults(handler=_plan)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as error:
        print(
            json.dumps({"status": "error", "error": f"{type(error).__name__}: {error}"}),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
