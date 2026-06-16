# EIBench Evaluation

Evaluate a single model on EIBench. The model is dropped into a multi-turn
dialogue with an LLM **simulator** that plays the user, tracks an
(anger, trust) state after every turn, and scores the final state against
per-scenario anchors. Both the simulator and the model under test are called
through an OpenAI-compatible `/chat/completions` endpoint.

## Quick start

1. Open `run_eval.sh` and fill in the two credential sets (or export them as
   environment variables):

   ```bash
   # Simulator (plays the user, does the scoring)
   export SIM_MODEL="qwen3-max"
   export SIM_API_KEY="..."
   export SIM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

   # Test model (the one being evaluated)
   export MODEL_NAME="gpt-5.4"
   export MODEL_API_KEY="..."
   export MODEL_BASE_URL="https://api.openai.com/v1"
   ```

2. Run:

   ```bash
   bash evaluation/run_eval.sh
   ```

## Output

Results are written to `results/<model_name>/`:

- `results.jsonl` — one line per scenario (final state, reward, full dialogue).
  Written incrementally, so an interrupted run resumes where it stopped.
- `summary.json` — per-scene and overall averages.

The overall and per-scene reward is also printed at the end. Reward is in
`[-1, +1]`: `+1` means the final state reached the success anchor, `-1` the
failure anchor.

## Notes

- `--n_per_scene N` evaluates only N scenarios per scene (default: all).
- `--max_rounds` sets the max dialogue turns per scenario (default: 8).
- The dialogue/scoring core lives in `../src/simulator_v2.py`; this folder only
  wires up the two API endpoints and aggregates results.
