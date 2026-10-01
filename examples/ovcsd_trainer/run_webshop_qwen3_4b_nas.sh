#!/usr/bin/env bash
set -euo pipefail

# Qwen3-4B variant of the validated WebShop-small OVCSD launcher.
REPO_ROOT=/data/minghao/SDAR
STORAGE=/data/minghao/nas2-d6/SDAR-storage
MODEL_PATH="${MODEL_PATH:-$STORAGE/models/Qwen3-4B}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$STORAGE/checkpoints/webshop/qwen3-4b-ovcsd}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-ovcsd_qwen3_4b_small}"

python3 - "$MODEL_PATH" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
required = [root / "config.json", root / "model.safetensors.index.json"]
if required[1].is_file():
    index = json.loads(required[1].read_text())
    required.extend(root / name for name in set(index["weight_map"].values()))
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit("Missing Qwen3-4B model files:\n  " + "\n  ".join(missing))
PY

mkdir -p "$CHECKPOINT_DIR"
CHECKPOINT_DIR="$CHECKPOINT_DIR" \
NUM_GPUS="${NUM_GPUS:-4}" \
exec "$REPO_ROOT/examples/ovcsd_trainer/run_webshop_qwen3_1b_nas.sh" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.save_freq=10 \
    "$@"
