"""Dependency-light command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kvbridge.mapper import CrossModelKVMapper
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
    if args.output is not None:
        mapper.save(args.output, overwrite=args.overwrite)
    payload = {
        "status": "ok" if score >= args.min_r2 else "failed",
        "evaluation_r2": round(score, 8),
        "selected_layers": mapper.selected_layers,
        "expected_layers": problem.true_layers,
        "fit_key_r2": [round(value, 8) for value in mapper.fit_key_r2],
        "fit_value_r2": [round(value, 8) for value in mapper.fit_value_r2],
        "tokens": report.tokens,
        "transfer_ms": round(report.elapsed_ms, 3),
        "artifact": str(Path(args.output).resolve()) if args.output else None,
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
