#!/usr/bin/env bash

set -euo pipefail
set -x

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

ENGINE="${ENGINE:-vllm}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
DATA_ROOT="${DATA_ROOT:-${HOME}/data/verl-agent/text}"

TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-16}"
VAL_DATA_SIZE="${VAL_DATA_SIZE:-128}"
GROUP_SIZE="${GROUP_SIZE:-8}"
NUM_GPUS="${NUM_GPUS:-2}"
NUM_CPUS_PER_ENV_WORKER="${NUM_CPUS_PER_ENV_WORKER:-0.1}"

SDAR_COEF="${SDAR_COEF:-0.01}"
GATE_BETA="${GATE_BETA:-5.0}"
SKILL_ALL="${SKILL_ALL:-false}"
WEBSHOP_USE_SMALL="${WEBSHOP_USE_SMALL:-false}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-sdar_qwen2.5_3b_webshop_coef${SDAR_COEF}_beta${GATE_BETA}_skillall${SKILL_ALL}}"

WEBSHOP_DATA_DIR="${REPO_ROOT}/agent_system/environments/env_package/webshop/webshop/data"
if [[ "${WEBSHOP_USE_SMALL}" == "true" ]]; then
    REQUIRED_WEBSHOP_FILES=(items_shuffle_1000.json items_ins_v2_1000.json items_human_ins.json)
else
    REQUIRED_WEBSHOP_FILES=(items_shuffle.json items_ins_v2.json items_human_ins.json)
fi
for required_file in "${REQUIRED_WEBSHOP_FILES[@]}"; do
    if [[ ! -s "${WEBSHOP_DATA_DIR}/${required_file}" ]]; then
        echo "Missing WebShop data file: ${WEBSHOP_DATA_DIR}/${required_file}" >&2
        echo "Run the WebShop installation/download steps before training." >&2
        exit 1
    fi
done

if [[ ! -d "${REPO_ROOT}/agent_system/environments/env_package/webshop/webshop/search_engine/indexes" ]]; then
    echo "Missing WebShop search index. Run webshop/setup.sh -d all first." >&2
    exit 1
fi

mkdir -p "${DATA_ROOT}"
if [[ ! -s "${DATA_ROOT}/train.parquet" || ! -s "${DATA_ROOT}/test.parquet" ]]; then
    python3 -m examples.data_preprocess.prepare \
        --mode text \
        --local_dir "$(dirname "${DATA_ROOT}")" \
        --train_data_size "${TRAIN_DATA_SIZE}" \
        --val_data_size "${VAL_DATA_SIZE}"
fi

LOGGER="['console']"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    LOGGER="['console','wandb']"
fi

python3 -m verl.trainer.main_sdar \
    algorithm.adv_estimator=grpo \
    data.train_files="${DATA_ROOT}/train.parquet" \
    data.val_files="${DATA_ROOT}/test.parquet" \
    data.train_batch_size="${TRAIN_DATA_SIZE}" \
    data.val_batch_size="${VAL_DATA_SIZE}" \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name="${ENGINE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    +algorithm.sdar.sdar_coef="${SDAR_COEF}" \
    +algorithm.sdar.gate_beta="${GATE_BETA}" \
    +algorithm.sdar.skills_dir="${REPO_ROOT}/skills/webshop" \
    +algorithm.sdar.skill_all="${SKILL_ALL}" \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n="${GROUP_SIZE}" \
    env.webshop.use_small="${WEBSHOP_USE_SMALL}" \
    env.resources_per_worker.num_cpus="${NUM_CPUS_PER_ENV_WORKER}" \
    trainer.critic_warmup=0 \
    trainer.logger="${LOGGER}" \
    trainer.project_name=verl_agent_webshopv1 \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    "$@"
