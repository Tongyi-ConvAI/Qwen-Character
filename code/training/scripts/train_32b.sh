#!/usr/bin/env bash
# =============================================================================
# CTC-GRPO training: Qwen3-32B on EIBench (FSDP + vLLM, rollout TP=8).
#
# Reproduces the 32B run in the paper: same recipe as 8B (sigma_min=0.1,
# entropy=0.001, centered turn-credit alpha=15, Qwen3-Max simulator) with the
# 32B-scale sequence lengths and tensor parallelism.
#
# 32B needs a lot of memory: one 8xGPU node is the minimum (rollout TP=8). For
# multi-node, set NNODES>1, start this script's Ray head on node 0, and join
# the other nodes to it (ray start --address=<head>:6379) before training.
#
# Fill in the 3 starred vars, then:  bash scripts/train_32b.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CODE_DIR="$(cd "$TRAIN_DIR/.." && pwd)"
REPO_ROOT="$(cd "$CODE_DIR/.." && pwd)"
SRC_DIR="${CODE_DIR}/src"

RL_VENV="${RL_VENV:-}"
[ -n "$RL_VENV" ] && [ -f "$RL_VENV" ] && source "$RL_VENV"
export PYTHONPATH="${TRAIN_DIR}:${SRC_DIR}:${PYTHONPATH:-}"

# ============================================================================
# ★★★ FILL THESE IN ★★★
# ============================================================================
MODEL_PATH="${MODEL_PATH:-/path/to/Qwen3-32B}"                         # ★ local HF model dir
export SIMULATOR_API_KEYS="${SIMULATOR_API_KEYS:-YOUR_SIM_API_KEYS}"   # ★ comma-separated keys
export SIMULATOR_BASE_URL="${SIMULATOR_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions}"  # ★ full chat/completions URL
SIMULATOR_MODEL="${SIMULATOR_MODEL:-qwen3-max}"

# ============================================================================
# Hyperparameters (paper 32B run)
# ============================================================================
ACTOR_LR="1e-6"; LR_WARMUP_RATIO="0.03"; MIN_LR_RATIO="0.1"; CLIP_RATIO="0.2"; ENTROPY_COEFF="0.001"
PROMPT_LEN="16384"; RESPONSE_LEN="8192"
TRAIN_BATCH_SIZE="16"; VAL_BATCH_SIZE="32"; ROLLOUT_N="8"
PPO_MINI_BATCH_SIZE="32"; PPO_MICRO_BATCH_SIZE_PER_GPU="2"
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU="24576"; ROLLOUT_TP="8"
TURN_CREDIT_ALPHA="15.0"; GRPO_STD_MIN="0.1"          # CTC-GRPO core
TOTAL_STEPS="${TOTAL_STEPS:-120}"; MAX_TURNS="8"
N_GPUS_PER_NODE="8"; NNODES="${NNODES:-1}"

# ============================================================================
# Data
# ============================================================================
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/train}"
VAL_DIR="${VAL_DIR:-$REPO_ROOT/data/test}"
TRAIN_FILES="${DATA_DIR}/charm_final.jsonl,${DATA_DIR}/defense_final.jsonl,${DATA_DIR}/repair_final.jsonl,${DATA_DIR}/support_final.jsonl"
VAL_FILES="${VAL_DIR}/charm_final.jsonl,${VAL_DIR}/defense_final.jsonl,${VAL_DIR}/repair_final.jsonl,${VAL_DIR}/support_final.jsonl"
OUT_DIR="${OUT_DIR:-$TRAIN_DIR/ckpt_32b}"
mkdir -p "$OUT_DIR"

# ============================================================================
# Simulator / reward env
# ============================================================================
export SIMULATOR_MODEL
export SIMULATOR_MAX_TURNS="${MAX_TURNS}"
export SIMULATOR_HTTP_TIMEOUT="150"
export SIMULATOR_MAX_RETRY="8"
export SIMULATOR_BATCH_MAX_WAIT="2.0"
export REWARD_MODE="simulator_only"
export THINK_PENALTY_MODE="ratio"; export THINK_MISSING_PENALTY="0.2"
export PARSE_PENALTY_MODE="any_bad_turn"; export PARSE_ERROR_PENALTY="0.05"
export ENABLE_THINKING="True"
export ROLLOUT_TEMPERATURE="0.6"; export ROLLOUT_TOP_P="0.95"; export ROLLOUT_TOP_K="20"

# ---- runtime ----
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 VLLM_USE_V1=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_MODE="${WANDB_MODE:-offline}"

# ---- preflight ----
[ "$SIMULATOR_API_KEYS" = "YOUR_SIM_API_KEYS" ] && { echo "ERROR: set SIMULATOR_API_KEYS"; exit 1; }
[ -d "$MODEL_PATH" ] || { echo "ERROR: MODEL_PATH not found: $MODEL_PATH"; exit 1; }
[ -f "${DATA_DIR}/charm_final.jsonl" ] || { echo "ERROR: train data not found under $DATA_DIR"; exit 1; }
(( PPO_MINI_BATCH_SIZE % PPO_MICRO_BATCH_SIZE_PER_GPU != 0 )) && { echo "ERROR: mini % micro != 0"; exit 1; }

# ---- start Ray head on this node (for NNODES>1, join workers before training) ----
ray stop --force 2>/dev/null || true
ray start --head --port=6379 --disable-usage-stats --num-gpus=${N_GPUS_PER_NODE} --temp-dir=/tmp/ray_eibench
if (( NNODES > 1 )); then
  echo ">>> NNODES=${NNODES}: now run 'ray start --address=<this-node-ip>:6379' on the other ${NNODES} nodes, then press Enter."
  read -r _
fi

echo "=== CTC-GRPO 32B: steps=${TOTAL_STEPS}, alpha=${TURN_CREDIT_ALPHA}, sigma_min=${GRPO_STD_MIN}, TP=${ROLLOUT_TP}, nodes=${NNODES} ==="

python3 -m verl.trainer.main_ppo \
  +ray_kwargs.ray_init.address=auto \
  data.prompt_key=prompt \
  data.train_files="[${TRAIN_FILES}]" \
  data.val_files="[${VAL_FILES}]" \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.shuffle=True \
  +data.stratified_batch=True \
  +data.stratified_key=scene_tag \
  data.val_batch_size=${VAL_BATCH_SIZE} \
  data.max_prompt_length=${PROMPT_LEN} \
  data.max_response_length=${RESPONSE_LEN} \
  data.return_raw_chat=True \
  +data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING} \
  algorithm.adv_estimator=grpo \
  +algorithm.grpo_grouping=prompt_trace \
  +algorithm.grpo_scene_key=scene_tag \
  +algorithm.grpo_trace_key=trace_uid \
  algorithm.norm_adv_by_std_in_grpo=True \
  +algorithm.grpo_std_min=${GRPO_STD_MIN} \
  algorithm.rollout_correction.bypass_mode=True \
  +algorithm.turn_credit_alpha=${TURN_CREDIT_ALPHA} \
  algorithm.kl_ctrl.kl_coef=0 \
  +actor_rollout_ref.thinking=${ENABLE_THINKING} \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${LR_WARMUP_RATIO} \
  actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
  actor_rollout_ref.actor.optim.min_lr_ratio=${MIN_LR_RATIO} \
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU} \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU} \
  actor_rollout_ref.actor.clip_ratio=${CLIP_RATIO} \
  actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO} \
  actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO} \
  actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
  +actor_rollout_ref.rollout.environment.name=simulator_v2_env_strip_think \
  +actor_rollout_ref.rollout.environment.per_turn_length=1024 \
  +actor_rollout_ref.rollout.environment.max_turns=${MAX_TURNS} \
  +actor_rollout_ref.rollout.environment.model_sim="${SIMULATOR_MODEL}" \
  +actor_rollout_ref.rollout.environment.simulator_batch_size=64 \
  +actor_rollout_ref.rollout.environment.reward_mode="${REWARD_MODE}" \
  +actor_rollout_ref.rollout.environment.think_penalty_mode="${THINK_PENALTY_MODE}" \
  +actor_rollout_ref.rollout.environment.think_missing_penalty="${THINK_MISSING_PENALTY}" \
  +actor_rollout_ref.rollout.environment.parse_penalty_mode="${PARSE_PENALTY_MODE}" \
  +actor_rollout_ref.rollout.environment.parse_error_penalty="${PARSE_ERROR_PENALTY}" \
  +actor_rollout_ref.rollout.environment.training_method=naive_grpo \
  actor_rollout_ref.rollout.prompt_length=${PROMPT_LEN} \
  actor_rollout_ref.rollout.response_length=${RESPONSE_LEN} \
  actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE} \
  actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P} \
  actor_rollout_ref.rollout.top_k=${ROLLOUT_TOP_K} \
  actor_rollout_ref.rollout.n=${ROLLOUT_N} \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
  actor_rollout_ref.rollout.disable_log_stats=True \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  trainer.project_name="eibench_ctc_grpo" \
  trainer.experiment_name="ctc_grpo_32b" \
  trainer.default_local_dir="${OUT_DIR}" \
  'trainer.logger=["console","wandb"]' \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
  trainer.nnodes=${NNODES} \
  trainer.save_freq=10 \
  trainer.test_freq=30 \
  trainer.total_epochs=100 \
  trainer.total_training_steps=${TOTAL_STEPS} \
  trainer.resume_mode=auto \
  2>&1 | tee "${OUT_DIR}/train.log"

ray stop --force 2>/dev/null || true
echo "=== done. checkpoints in ${OUT_DIR} ==="
