#!/usr/bin/env bash
set -euo pipefail

CONFIG="${KVBRIDGE_T2_CONFIG:-configs/qwen3_0.6b_to_1.7b.t2-smoke.json}"
CALIBRATION_DIR="${KVBRIDGE_CALIBRATION_DIR:-data/calibration}"
ARTIFACT_DIR="${KVBRIDGE_ARTIFACT_DIR:-artifacts/qwen3-0.6b-to-1.7b}"
RESULT_PATH="${KVBRIDGE_RESULT_PATH:-results/qwen3-0.6b-to-1.7b.t2.json}"

python -m pip install -e ".[hf]"
python -m kvbridge plan "$CONFIG"
nvidia-smi

python experiments/capture_hf_calibration.py \
  "$CONFIG" \
  --output-dir "$CALIBRATION_DIR" \
  --execute

python experiments/fit_hf_mapper.py \
  "$CONFIG" \
  --calibration-dir "$CALIBRATION_DIR" \
  --output-dir "$ARTIFACT_DIR" \
  --execute

python experiments/evaluate_hf_mapper.py \
  "$CONFIG" \
  --artifact-dir "$ARTIFACT_DIR" \
  --output "$RESULT_PATH" \
  --execute

KVBRIDGE_RESULT_PATH="$RESULT_PATH" python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["KVBRIDGE_RESULT_PATH"])
payload = json.loads(path.read_text(encoding="utf-8"))
print(json.dumps(payload["summary"], indent=2))
if not payload["summary"]["all_quality_gates_passed"]:
    raise SystemExit("T2 completed but failed its preregistered quality gates")
PY
