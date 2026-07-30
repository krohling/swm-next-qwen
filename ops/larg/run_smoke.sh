#!/bin/bash
set -uo pipefail
B=/scratch/cluster/krohling/swm-followup
cd $B/swm-next-qwen; source $B/.venv/bin/activate; set -a; source $B/.env; set +a; export WANDB_ENTITY=kevin_ai
export XDG_CACHE_HOME=$B/.xdg_cache TMPDIR=$B/.tmp
export HF_HOME=$B/hf_cache HF_HUB_OFFLINE=1
export WANDB_DIR=$B/.wandb_lora
export CUDA_VISIBLE_DEVICES=$1 PYTHONUNBUFFERED=1
mkdir -p $WANDB_DIR
nice -n 10 python train.py --config configs/smoke.yaml
