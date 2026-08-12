#!/usr/bin/env bash
set -euo pipefail

CONFIG="${KVBRIDGE_T2_CONFIG:-configs/qwen3_0.6b_to_1.7b.t2-smoke.json}"
CALIBRATION_DIR="${KVBRIDGE_CALIBRATION_DIR:-data/calibration}"
ARTIFACT_DIR="${KVBRIDGE_ARTIFACT_DIR:-artifacts/qwen3-0.6b-to-1.7b}"
RESULT_PATH="${KVBRIDGE_RESULT_PATH:-results/qwen3-0.6b-to-1.7b.t2.json}"
RESUME="${KVBRIDGE_RESUME:-1}"

python -m pip install -e ".[hf]"
python -m kvbridge plan "$CONFIG"
nvidia-smi

if [[ "$RESUME" == "1" && -f "$CALIBRATION_DIR/capture_manifest.json" ]]; then
  python experiments/validate_hf_evidence.py \
    "$CONFIG" \
    --calibration-dir "$CALIBRATION_DIR"
  echo "Reusing validated calibration evidence: $CALIBRATION_DIR"
else
  python experiments/capture_hf_calibration.py \
    "$CONFIG" \
    --output-dir "$CALIBRATION_DIR" \
    --execute
fi

if [[ "$RESUME" == "1" && -f "$ARTIFACT_DIR/manifest.json" && -f "$ARTIFACT_DIR/fit_run.json" ]]; then
  python experiments/validate_hf_evidence.py \
    "$CONFIG" \
    --calibration-dir "$CALIBRATION_DIR" \
    --artifact-dir "$ARTIFACT_DIR"
  echo "Reusing validated mapper evidence: $ARTIFACT_DIR"
else
  python experiments/fit_hf_mapper.py \
    "$CONFIG" \
    --calibration-dir "$CALIBRATION_DIR" \
    --output-dir "$ARTIFACT_DIR" \
    --execute
fi

if [[ "$RESUME" == "1" && -f "$RESULT_PATH" ]]; then
  python experiments/validate_hf_evidence.py \
    "$CONFIG" \
    --calibration-dir "$CALIBRATION_DIR" \
    --artifact-dir "$ARTIFACT_DIR" \
    --result "$RESULT_PATH"
  echo "Reusing validated evaluation evidence: $RESULT_PATH"
else
  set +e
  python experiments/evaluate_hf_mapper.py \
    "$CONFIG" \
    --artifact-dir "$ARTIFACT_DIR" \
    --output "$RESULT_PATH" \
    --execute
  EVALUATION_EXIT=$?
  set -e
  if ! python experiments/validate_hf_evidence.py \
    "$CONFIG" \
    --calibration-dir "$CALIBRATION_DIR" \
    --artifact-dir "$ARTIFACT_DIR" \
    --result "$RESULT_PATH"; then
    if [[ "$EVALUATION_EXIT" -ne 0 ]]; then
      exit "$EVALUATION_EXIT"
    fi
    exit 1
  fi
  if [[ "$EVALUATION_EXIT" -ne 0 ]]; then
    echo "Evaluator exited $EVALUATION_EXIT after committing valid evidence; continuing."
  fi
fi

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
