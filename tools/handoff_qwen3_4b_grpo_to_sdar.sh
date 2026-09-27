#!/usr/bin/env bash
set -euo pipefail

GRPO_LOG=/data/minghao/nas2-d6/SDAR-storage/runs/webshop_qwen3_4b_grpo_20260920_111419/train.log
REPO_ROOT=/data/minghao/SDAR
RUNS_ROOT=/data/minghao/nas2-d6/SDAR-storage/runs
SDAR_SESSION=webshop_qwen3_4b_sdar
STATE_FILE="$RUNS_ROOT/qwen3_4b_sdar_handoff.state"

while true; do
    if rg -q 'Traceback|Error executing job|CUDA out of memory|ncclUnhandledCudaError' "$GRPO_LOG"; then
        printf 'status=grpo_failed\ntime=%s\n' "$(date -u +%FT%TZ)" > "$STATE_FILE"
        exit 1
    fi

    if rg -q 'Training Progress: *100%.*150/150' "$GRPO_LOG"; then
        mapfile -t used_memory < <(
            nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
                awk -F, '$1 >= 1 && $1 <= 4 {gsub(/ /, "", $2); print $2}'
        )
        gpu_released=true
        for used in "${used_memory[@]}"; do
            if (( used > 1024 )); then
                gpu_released=false
                break
            fi
        done

        if [[ "$gpu_released" == true ]]; then
            if tmux has-session -t "$SDAR_SESSION" 2>/dev/null; then
                printf 'status=sdar_session_already_exists\ntime=%s\n' "$(date -u +%FT%TZ)" > "$STATE_FILE"
                exit 1
            fi

            timestamp=$(date -u +%Y%m%d_%H%M%S)
            run_dir="$RUNS_ROOT/webshop_qwen3_4b_sdar_${timestamp}"
            mkdir -p "$run_dir"
            tmux new-session -d -s "$SDAR_SESSION"
            tmux set-option -t "$SDAR_SESSION" remain-on-exit on
            command="cd $REPO_ROOT && CUDA_VISIBLE_DEVICES=1,2,3,4 NUM_GPUS=4 NCCL_NVLS_ENABLE=0 $REPO_ROOT/examples/sdar_trainer/run_webshop_qwen3_4b_nas.sh > $run_dir/train.log 2>&1"
            tmux send-keys -t "$SDAR_SESSION" -l "$command"
            tmux send-keys -t "$SDAR_SESSION" Enter
            printf 'status=sdar_started\ntime=%s\nsession=%s\nrun_dir=%s\nlog=%s\n' \
                "$(date -u +%FT%TZ)" "$SDAR_SESSION" "$run_dir" "$run_dir/train.log" > "$STATE_FILE"
            exit 0
        fi
    fi

    printf 'status=waiting_for_grpo\ntime=%s\n' "$(date -u +%FT%TZ)" > "$STATE_FILE"
    sleep 300
done
