# EIBench Data Generation

Two steps:

1. **`generate_scenarios.py`** — generate scenarios. For each one it samples a
   theme + scenario keyword + one option from each modifier pool
   (relationship / place / intensity / personality), draws a few random
   same-scene few-shot examples, and asks an LLM to expand it into a full
   scenario: the `aggressor_profile` (simulated user) and `defender_profile`
   (model under test), an `opening_line`, and the three state anchors
   (`initial_calibration` / `rub_goals` / `worst_case`). Anchors are then
   rounded to multiples of 5 and reordered so that
   `target < initial < worst` on the emotion axis and `worst < initial < target`
   on the relation axis.
2. **`clean_scenarios.py`** — format-clean. Rule-checks that each profile's
   `### sections` match the required layout and order; any record that does not
   is sent to an LLM that re-orders / fills / merges the sections back into the
   target layout without changing the meaning.

The seed pools (`SCENE_KEYWORDS`, `SEED_DIMENSIONS`) and the target section
layout live at the top of the two files — edit them there.

## Step 0 — implement the LLM call (required)

Both scripts leave the actual API call empty so you can use any provider. In the
paper, scenarios are generated with **Gemini-3.1-Pro**. Open each file and
implement `call_llm(messages, model, temperature, max_retry)` — it just needs to
send OpenAI-style `messages` to your model and return the text reply. A
ready-to-paste OpenAI-compatible example is in `generate_scenarios.py`'s
`call_llm` docstring; paste the same body into `clean_scenarios.py`.

## One-click: generate + clean

```bash
SCENE=charm SEED_FILE=data/seed_profiles.jsonl N=50 MODEL=gemini-3.1-pro \
    bash data_generation/run_pipeline.sh
```

`SEED_FILE` must contain at least 2 scenarios of the chosen `SCENE`
(`scene_tag` field) for the few-shot examples. Generate one scene type at a
time. Output is append-only.

## Or run the two steps manually

```bash
# 1) generate
python generate_scenarios.py \
    --scene_type charm \
    --input  data/seed_profiles.jsonl \
    --output out/charm_generated.jsonl \
    --n 50 --model_gen gemini-3.1-pro --max_workers 20

# 2) format-clean (in place; globs allowed)
python clean_scenarios.py --input "out/*.jsonl" --model gemini-3.1-pro --workers 20
# add --check-only to just report format issues without calling the LLM
```

## Concurrency (max_workers)

Both scripts take a concurrency argument that drives a `ThreadPoolExecutor`:

- `generate_scenarios.py --max_workers N` (default 20) — set in `main()`.
- `clean_scenarios.py --workers N` (default 20) — set in `main()`.
- `run_pipeline.sh` passes `MAX_WORKERS` (default 20) to both.

Set it to `1` to run serially (debugging / strict rate limits); raise it to send
more concurrent requests — tune to your LLM endpoint's rate limit.

## Notes

- Generation can use different models/temperatures per stage via
  `--model_*` / `--temperature_*` (generation runs hotter for diversity,
  anchor scoring runs cooler for stability).
- After generation + cleaning, deduplicate against the test set before use
  (keyword-combo filtering + embedding retrieval), as described in the paper.
