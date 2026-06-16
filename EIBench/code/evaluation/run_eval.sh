#!/bin/bash
# =============================================================================
# EIBench single-model evaluation.
#
# Usage:
#   bash evaluation/run_eval.sh
#
# Fill in the two credential sets below (or export them as env vars first):
#   1) SIMULATOR  - the user-simulator that drives the dialogue and scores it
#   2) TEST MODEL - the model you want to evaluate
# Both are called through an OpenAI-compatible /chat/completions endpoint.
# =============================================================================
set -e

# ===== Locate project root =====
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ============================================================================
# 1) SIMULATOR  (model_name + api_key + base_url)
# ============================================================================
export SIM_MODEL="${SIM_MODEL:-qwen3-max}"
export SIM_API_KEY="${SIM_API_KEY:-YOUR_SIM_API_KEY}"
export SIM_BASE_URL="${SIM_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"

# ============================================================================
# 2) TEST MODEL  (model_name + api_key + base_url) -- the model under evaluation
# ============================================================================
export MODEL_NAME="${MODEL_NAME:-gpt-5.4}"
export MODEL_API_KEY="${MODEL_API_KEY:-YOUR_MODEL_API_KEY}"
export MODEL_BASE_URL="${MODEL_BASE_URL:-https://api.openai.com/v1}"

# ============================================================================

# ===== Eval config =====
DATA_DIR="${EIBENCH_DATA_DIR:-$PROJECT_ROOT/data/test}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results}"
MAX_ROUNDS="${MAX_ROUNDS:-8}"
MAX_WORKERS="${MAX_WORKERS:-0}"   # 0 = default (5)

echo "=================================================================="
echo "EIBench single-model evaluation"
echo "=================================================================="
echo "  simulator  : $SIM_MODEL"
echo "  test model : $MODEL_NAME"
echo "  data dir   : $DATA_DIR"
echo "  output dir : $OUTPUT_DIR"
echo

# ===== Pre-flight checks =====
if [ "$SIM_API_KEY" = "YOUR_SIM_API_KEY" ]; then
    echo "ERROR: please set SIM_API_KEY (simulator API key)"
    exit 1
fi
if [ "$MODEL_API_KEY" = "YOUR_MODEL_API_KEY" ]; then
    echo "ERROR: please set MODEL_API_KEY (test-model API key)"
    exit 1
fi
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: data directory not found: $DATA_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

python3 "$SCRIPT_DIR/run_eval.py" \
    --model_name "$MODEL_NAME" \
    --sim_model "$SIM_MODEL" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --max_rounds "$MAX_ROUNDS" \
    --max_workers "$MAX_WORKERS" \
    --resume

echo
echo "=================================================================="
echo "Done. Results saved to: $OUTPUT_DIR"
echo "=================================================================="
