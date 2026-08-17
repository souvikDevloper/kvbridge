#!/usr/bin/env bash
set -euo pipefail

export KVBRIDGE_T2_CONFIG="configs/qwen3_8b_to_14b.t2-h100-pilot.json"
export KVBRIDGE_CALIBRATION_DIR="data/calibration-qwen3-8b-to-14b"
export KVBRIDGE_ARTIFACT_DIR="artifacts/qwen3-8b-to-14b"
export KVBRIDGE_RESULT_PATH="results/qwen3-8b-to-14b.t2-h100-pilot.json"

bash scripts/kaggle_t2_smoke.sh
