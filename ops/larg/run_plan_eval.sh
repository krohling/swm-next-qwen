#!/bin/bash
set -uo pipefail
B=/scratch/cluster/krohling/swm-followup
CKPT=$1 TASK=$2 GPU=$3 OUT=$4 SEEDS=${5:-25} SEED_START=${6:-6000}
cd $B/swm-next-qwen; source $B/.venv/bin/activate
export XDG_CACHE_HOME=$B/.xdg_cache TMPDIR=$B/.tmp MPLCONFIGDIR=$B/.mpl_cache
export CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl
export HF_HOME=$B/hf_cache HF_HUB_OFFLINE=1 TF_USE_LEGACY_KERAS=1
export SWMS_REPO=$B/swms
export PYTHONPATH=$B/swm-next-qwen:$B/swms:$B/language-table
nice -n 15 python scripts/evaluate_planning.py --ckpt $CKPT --task-idx $TASK \
  --seed-start $SEED_START --num-seeds $SEEDS --out $OUT
