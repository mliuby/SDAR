#!/usr/bin/env bash
set -euo pipefail
set -x

# Reproducible local/NAS launcher for Qwen3-1.7B + WebShop-small OVCSD.
# This script prepares one run; use a persistent tmux session when launching it.
source /data/minghao/anaconda3/etc/profile.d/conda.sh
conda activate verl-agent-webshop

REPO_ROOT=/data/minghao/SDAR
STORAGE=/data/minghao/nas2-d6/SDAR-storage
MODEL_PATH="$STORAGE/models/Qwen3-1.7B"
TRAIN_FILE="$STORAGE/data/verl-agent/text/train.parquet"
VAL_FILE="$STORAGE/data/verl-agent/text/test.parquet"
WEBSHOP_ROOT="$REPO_ROOT/agent_system/environments/env_package/webshop/webshop"
SKILLS_DIR="$REPO_ROOT/skills/webshop"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$STORAGE/checkpoints/webshop/qwen3-1.7b-ovcsd}"

export HF_HOME="$STORAGE/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
unset TRANSFORMERS_CACHE
export JAVA_HOME="$CONDA_PREFIX"
export JVM_PATH="$CONDA_PREFIX/lib/jvm/lib/server/libjvm.so"
export PATH="$JAVA_HOME/bin:$PATH"
export _JAVA_OPTIONS="-XX:+UseSerialGC -Xss512k -Xmx1g"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export DATASETS_MAX_WORKERS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
ulimit -u 65536

for required in \
    "$MODEL_PATH/config.json" \
    "$MODEL_PATH/model.safetensors.index.json" \
    "$MODEL_PATH/model-00001-of-00002.safetensors" \
    "$MODEL_PATH/model-00002-of-00002.safetensors" \
    "$TRAIN_FILE" \
    "$VAL_FILE" \
    "$WEBSHOP_ROOT/data/items_shuffle_1000.json" \
    "$WEBSHOP_ROOT/search_engine/indexes/segments_1" \
    "$SKILLS_DIR/skill_mapping.json" \
    "$SKILLS_DIR/general_skills.md"; do
    if [[ ! -e "$required" ]]; then
        echo "Missing required file: $required" >&2
        exit 1
    fi
done

mkdir -p "$CHECKPOINT_DIR"
cd "$REPO_ROOT"

ENGINE="${ENGINE:-vllm}"
NUM_GPUS="${NUM_GPUS:-2}"
TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-16}"
VAL_DATA_SIZE="${VAL_DATA_SIZE:-128}"
GROUP_SIZE="${GROUP_SIZE:-8}"
SUFFIX_COEF="${SUFFIX_COEF:-1.0}"
TOPK="${TOPK:-16}"
NUM_TEACHER_CONTINUATIONS="${NUM_TEACHER_CONTINUATIONS:-2}"
MAX_NODES_PER_GROUP="${MAX_NODES_PER_GROUP:-3}"
SKILL_ALL="${SKILL_ALL:-false}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-5}"

python3 -m verl.trainer.main_ovcsd \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size="$TRAIN_DATA_SIZE" \
    data.val_batch_size="$VAL_DATA_SIZE" \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="$MODEL_PATH" \
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
    actor_rollout_ref.rollout.name="$ENGINE" \
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
    +algorithm.ovcsd.suffix_coef="$SUFFIX_COEF" \
    +algorithm.ovcsd.topk="$TOPK" \
    +algorithm.ovcsd.num_teacher_continuations="$NUM_TEACHER_CONTINUATIONS" \
    +algorithm.ovcsd.max_nodes_per_group="$MAX_NODES_PER_GROUP" \
    +algorithm.ovcsd.r_succ=10.0 \
    +algorithm.ovcsd.eps_r=1e-6 \
    +algorithm.ovcsd.min_support=2 \
    +algorithm.ovcsd.fallback_max_depth=4 \
    +algorithm.ovcsd.cover_all_branches=False \
    +algorithm.ovcsd.max_intervene_groups=-1 \
    +algorithm.ovcsd.zero_fail_group_adv=True \
    +algorithm.ovcsd.token_scope=action \
    +algorithm.ovcsd.teacher_do_sample=True \
    +algorithm.ovcsd.branch_pool_size=-1 \
    +algorithm.ovcsd.skills_dir="$SKILLS_DIR" \
    +algorithm.ovcsd.skill_all="$SKILL_ALL" \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n="$GROUP_SIZE" \
    env.resources_per_worker.num_cpus=0.1 \
    env.webshop.use_small=True \
    trainer.critic_warmup=0 \
    "trainer.logger=['console']" \
    trainer.project_name=verl_agent_webshopv1 \
    trainer.experiment_name=ovcsd_qwen3_1.7b_small \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    +ray_init.include_dashboard=False \
    "$@"
