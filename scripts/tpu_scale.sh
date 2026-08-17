#!/usr/bin/env bash
set -euo pipefail

export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

CONFIG="${KVBRIDGE_TPU_CONFIG:-configs/qwen3_14b_to_32b.tpu-128k.json}"
OUTPUT_DIR="${KVBRIDGE_TPU_OUTPUT_DIR:-runs/qwen3-14b-to-32b-tpu-128k}"
RESULT_PATH="${KVBRIDGE_TPU_RESULT_PATH:-$OUTPUT_DIR/evaluation.json}"
RESUME="${KVBRIDGE_RESUME:-1}"
EVALUATE="${KVBRIDGE_TPU_EVALUATE:-1}"

python -m pip install -e ".[hf]"
python -m kvbridge plan "$CONFIG"
python -c "import torch_xla, torch_xla.runtime as xr; print({'torch_xla': torch_xla.__version__, 'devices': xr.global_runtime_device_count()})"

RUN_ARGS=("$CONFIG" --output-dir "$OUTPUT_DIR" --execute)
if [[ "$RESUME" != "1" ]]; then
  RUN_ARGS+=(--no-resume)
fi
python experiments/run_tpu_scale.py "${RUN_ARGS[@]}"

if [[ "$EVALUATE" == "1" ]]; then
  if [[ "$RESUME" == "1" && -f "$RESULT_PATH" ]]; then
    python experiments/validate_tpu_evidence.py \
      "$CONFIG" --run-dir "$OUTPUT_DIR" --result "$RESULT_PATH"
    echo "Reusing validated TPU evaluation: $RESULT_PATH"
  else
    set +e
    python experiments/evaluate_tpu_mapper.py \
      "$CONFIG" \
      --artifact-dir "$OUTPUT_DIR/artifact" \
      --output "$RESULT_PATH" \
      --execute
    EVALUATION_EXIT=$?
    set -e
    python experiments/validate_tpu_evidence.py \
      "$CONFIG" --run-dir "$OUTPUT_DIR" --result "$RESULT_PATH"
    if [[ "$EVALUATION_EXIT" -ne 0 ]]; then
      exit "$EVALUATION_EXIT"
    fi
  fi
fi
