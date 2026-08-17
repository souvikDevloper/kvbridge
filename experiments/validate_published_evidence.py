"""Validate lightweight, checked-in real-model evidence and its hash chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kvbridge.evidence import validate_published_result_evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("evidence_dir", type=Path)
    args = parser.parse_args()
    report = validate_published_result_evidence(args.config, args.evidence_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
