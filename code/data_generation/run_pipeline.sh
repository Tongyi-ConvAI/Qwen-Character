#!/bin/bash
# =============================================================================
# EIBench data generation pipeline: generate -> format-clean.
#
# Prerequisite: implement call_llm() in BOTH generate_scenarios.py and
# clean_scenarios.py with your own LLM API (the paper uses Gemini-3.1-Pro).
#
# Usage:
#   bash data_generation/run_pipeline.sh
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ===== Config =====
SCENE="${SCENE:-charm}"                                  # support | defense | repair | charm
SEED_FILE="${SEED_FILE:-data/seed_profiles.jsonl}"        # existing scenarios used as few-shot
OUT_FILE="${OUT_FILE:-out/${SCENE}_generated.jsonl}"
N="${N:-50}"                                              # number of scenarios to generate
MODEL="${MODEL:-gemini-3.1-pro}"
MAX_WORKERS="${MAX_WORKERS:-20}"

echo "=================================================================="
echo "EIBench data pipeline  (scene=$SCENE, n=$N, model=$MODEL)"
echo "=================================================================="

# ----- Step 1: generate scenarios (profiles + 3 anchors) -----
echo "[1/2] generating scenarios -> $OUT_FILE"
python3 "$SCRIPT_DIR/generate_scenarios.py" \
    --scene_type "$SCENE" \
    --input  "$SEED_FILE" \
    --output "$OUT_FILE" \
    --n "$N" \
    --model_gen   "$MODEL" \
    --model_score "$MODEL" \
    --model_worst "$MODEL" \
    --max_workers "$MAX_WORKERS"

# ----- Step 2: format-clean (rule check + LLM re-clean in place) -----
echo "[2/2] format-cleaning $OUT_FILE"
python3 "$SCRIPT_DIR/clean_scenarios.py" \
    --input "$OUT_FILE" \
    --model "$MODEL" \
    --workers "$MAX_WORKERS"

echo
echo "Done. Cleaned scenarios in: $OUT_FILE"
