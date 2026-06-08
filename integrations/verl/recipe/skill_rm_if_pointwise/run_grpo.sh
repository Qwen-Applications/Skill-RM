#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${VERL_ROOT}"

: "${MODEL_PATH:?Set MODEL_PATH to the policy initialization model path.}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the processed VerInstruct train parquet.}"
: "${VAL_FILE:?Set VAL_FILE to the processed VerInstruct validation parquet.}"

export VLLM_USE_V1="${VLLM_USE_V1:-1}"

export SKILL_RM_BACKEND="${SKILL_RM_BACKEND:-openai}"
export MODEL_NAME="${MODEL_NAME:-Qwen3.5-27B}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export ENABLE_THINKING="${ENABLE_THINKING:-false}"
export SEND_THINKING_FIELD="${SEND_THINKING_FIELD:-true}"
export MODEL_TEMPERATURE="${MODEL_TEMPERATURE:-0.0}"
export MODEL_TOP_P="${MODEL_TOP_P:-1.0}"
export MODEL_RETRIES="${MODEL_RETRIES:-1}"
export MODEL_TIMEOUT="${MODEL_TIMEOUT:-300}"
export DEFAULT_CONCURRENCY="${DEFAULT_CONCURRENCY:-128}"
export HTTP_POOL_CONNECTIONS="${HTTP_POOL_CONNECTIONS:-128}"
export HTTP_POOL_MAXSIZE="${HTTP_POOL_MAXSIZE:-512}"
export MAX_TOKENS="${MAX_TOKENS:-4096}"
export MAX_AGENT_STEPS="${MAX_AGENT_STEPS:-5}"
export MAX_RESOURCES_PER_SAMPLE="${MAX_RESOURCES_PER_SAMPLE:-5}"
export FORCED_FINALIZATION_MAX_TOKENS="${FORCED_FINALIZATION_MAX_TOKENS:-512}"
export PYTHON_SANDBOX_TIMEOUT="${PYTHON_SANDBOX_TIMEOUT:-3.0}"
export PYTHON_SANDBOX_MAX_CODE_CHARS="${PYTHON_SANDBOX_MAX_CODE_CHARS:-6000}"
export PYTHON_SANDBOX_MAX_OUTPUT_CHARS="${PYTHON_SANDBOX_MAX_OUTPUT_CHARS:-4000}"
export MAX_PYTHON_SANDBOX_CALLS="${MAX_PYTHON_SANDBOX_CALLS:-2}"
export SKILL_RM_LOG_LEVEL="${SKILL_RM_LOG_LEVEL:-INFO}"

NUM_GPUS="${NUM_GPUS:-8}"
NNODES="${NNODES:-1}"
PROJECT_NAME="${PROJECT_NAME:-verl_skillrm_if_pointwise}"
EXP_NAME="${EXP_NAME:-skillrm_if_pointwise_mounted_verifier_plus_code}"
CKPT_DIR="${CKPT_DIR:-checkpoints/${PROJECT_NAME}/${EXP_NAME}}"

TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-32}"
VAL_PROMPT_BSZ="${VAL_PROMPT_BSZ:-32}"
N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-4}"
PPO_MINI_BSZ="${PPO_MINI_BSZ:-32}"
PPO_MICRO_BSZ_PER_GPU="${PPO_MICRO_BSZ_PER_GPU:-4}"
LOGPROB_MICRO_BSZ_PER_GPU="${LOGPROB_MICRO_BSZ_PER_GPU:-4}"
REWARD_NUM_WORKERS="${REWARD_NUM_WORKERS:-48}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
ROLLOUT_TP="${ROLLOUT_TP:-2}"
ROLLOUT_GPU_UTIL="${ROLLOUT_GPU_UTIL:-0.6}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-20}"
TRAINER_LOGGER="${TRAINER_LOGGER:-'[\"console\",\"tensorboard\"]'}"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size="${TRAIN_PROMPT_BSZ}" \
  data.val_batch_size="${VAL_PROMPT_BSZ}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BSZ}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BSZ_PER_GPU}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOGPROB_MICRO_BSZ_PER_GPU}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_UTIL}" \
  actor_rollout_ref.rollout.n="${N_RESP_PER_PROMPT}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOGPROB_MICRO_BSZ_PER_GPU}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  reward.reward_model.enable=False \
  reward.reward_manager.name=naive \
  reward.num_workers="${REWARD_NUM_WORKERS}" \
  reward.custom_reward_function.path=recipe/skill_rm_if_pointwise/reward_fn_pointwise.py \
  reward.custom_reward_function.name=compute_score \
  trainer.critic_warmup=0 \
  +reward.custom_reward_function.reward_kwargs.variant=skill_mounted_verifier_plus_code \
  trainer.logger="${TRAINER_LOGGER}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.n_gpus_per_node="${NUM_GPUS}" \
  trainer.nnodes="${NNODES}" \
  trainer.val_before_train=False \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.default_local_dir="${CKPT_DIR}" \
  actor_rollout_ref.actor.checkpoint.save_contents="['hf_model']" \
  "$@"
