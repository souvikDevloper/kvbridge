"""Validate TPU mapper/evaluation artifacts without loading either model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kvbridge.tpu_evidence import validate_tpu_evaluation, validate_tpu_fit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    report = (
        validate_tpu_evaluation(args.config, args.run_dir, args.result)
        if args.result is not None
        else validate_tpu_fit(args.config, args.run_dir)
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
