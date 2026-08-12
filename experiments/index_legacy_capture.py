"""Recoverably add shard hashes to an exact-config legacy capture manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kvbridge.evidence import index_legacy_capture_evidence
from kvbridge.provenance import code_revision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--calibration-dir", type=Path, default=Path("data/calibration"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            "Dry-run only. Re-run with --execute to hash every shard, preserve the original "
            "manifest, and write an indexed manifest."
        )
        return 0
    report = index_legacy_capture_evidence(
        args.config,
        args.calibration_dir,
        index_code_revision=code_revision(),
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
