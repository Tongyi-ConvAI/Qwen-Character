# EIBench & CTC-GRPO

EIBench is a multi-turn benchmark for **emotion management**: a model talks to an
LLM **simulator** that plays the user across several turns, and is scored by how
the user's `(anger, trust)` state moves toward per-scenario anchors. Scenarios
span four scene types — **Support / Defense / Repair / Charm**.

**CTC-GRPO** (Centered Turn-Credit GRPO) trains on EIBench by reusing the
simulator's per-turn state changes as a dense process reward, adding a centered
turn-level credit term to the GRPO advantage:
`A_τ = A_trace + α · (r_proc_τ − mean_τ r_proc)`.

## Repository layout

```
data/
  test/        213 held-out scenarios (the benchmark); {charm,defense,repair,support}_final.jsonl
  train/       ~2000 training scenarios, same schema
code/
  src/
    simulator_v2.py        the user-simulator + anchor-based scoring (shared core)
  evaluation/
    run_eval.py / run_eval.sh   evaluate one model on EIBench -> per-scene & overall reward
    README.md
  data_generation/
    generate_scenarios.py  sample seed pools -> LLM-expand into full scenarios + anchors
    clean_scenarios.py     rule-based section-format check + LLM re-clean
    run_pipeline.sh        one-click: generate -> clean
    README.md
  training/
    verl/                  RL framework (FSDP + vLLM) with the CTC-GRPO additions
    scripts/
      train_8b.sh          Qwen3-8B run (paper hyperparameters)
      train_32b.sh         Qwen3-32B run (paper hyperparameters)
    requirements.txt       pinned, mutually-compatible dependency versions
```

Each scenario carries two role profiles (`aggressor_profile` = simulated user,
`defender_profile` = model under test), an opening line, and three state anchors
(`initial_calibration` / `rub_goals` / `worst_case`).

## Environment setup

The simulator and the generation/cleaning steps call an **OpenAI-compatible**
LLM endpoint over HTTP; no model weights are needed for evaluation or data work.
Training additionally needs GPUs, FSDP, and vLLM.

```bash
# Python 3.10+; a CUDA-matched torch/vLLM stack is required for training.
cd code/training
pip install -r requirements.txt
```

`requirements.txt` holds the exact, mutually-compatible versions we trained with
(torch 2.8 / vLLM 0.11 / transformers 4.57 / ray 2.54 / flash-attn 2.8 ...). The
torch / vLLM / transformers / flash-attn versions are tightly coupled — install a
set that matches your CUDA, following the verl install guide if you deviate.

## Reproduce

### 1. Evaluate a model on EIBench

Drop any model into the simulator and score it on the 213-scenario test set.
Set the two credential sets (simulator + model under test), then:

```bash
cd code
bash evaluation/run_eval.sh
```

See `code/evaluation/README.md` for the env vars and output format
(`results/<model>/summary.json` with per-scene and overall reward).

### 2. (Optional) Generate more training scenarios

Implement the empty `call_llm()` in the two scripts (the paper uses
Gemini-3.1-Pro), then:

```bash
cd code
SCENE=charm N=50 bash data_generation/run_pipeline.sh   # generate -> format-clean
```

Details in `code/data_generation/README.md`.

### 3. Train with CTC-GRPO

Fill in `MODEL_PATH`, `SIMULATOR_API_KEYS`, `SIMULATOR_BASE_URL` at the top of the
script, then:

```bash
cd code/training

# full runs (paper hyperparameters)
bash scripts/train_8b.sh     
bash scripts/train_32b.sh    
```

Each script starts a local Ray head and launches `verl.trainer.main_ppo` with the
CTC-GRPO settings (`turn_credit_alpha`, `grpo_std_min`, the simulator rollout
environment). Checkpoints and `train.log` are written under the script's output
dir. Set `WANDB_MODE=online` + `WANDB_API_KEY` to log to Weights & Biases.

The simulator is reached over HTTP via `SIMULATOR_BASE_URL` / `SIMULATOR_API_KEYS`
using `SIMULATOR_MODEL` (default `qwen3-max`).

## License

- **Dataset** (`data/`): [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — see `data/README.md` for details.
- **Code** (`code/`): [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
