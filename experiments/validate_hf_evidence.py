"""Validate a real-model capture, mapper artifact, or evaluation result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kvbridge.evidence import (
    validate_capture_evidence,
    validate_mapper_evidence,
    validate_result_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--allow-legacy-shards", action="store_true")
    args = parser.parse_args()

    if args.result is not None:
        if args.artifact_dir is None:
            parser.error("--result requires --artifact-dir")
        report = validate_result_evidence(
            args.config, args.calibration_dir, args.artifact_dir, args.result
        )
    elif args.artifact_dir is not None:
        report = validate_mapper_evidence(args.config, args.calibration_dir, args.artifact_dir)
    else:
        report = validate_capture_evidence(
            args.config,
            args.calibration_dir,
            require_shard_hashes=not args.allow_legacy_shards,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
