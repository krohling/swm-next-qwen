#!/bin/bash
set -uo pipefail
B=/scratch/cluster/krohling/swm-followup
GPU=$1 OUT=$2; shift 2
cd $B/swm-next-qwen; source $B/.venv/bin/activate
export XDG_CACHE_HOME=$B/.xdg_cache TMPDIR=$B/.tmp MPLCONFIGDIR=$B/.mpl_cache
export CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl
export HF_HOME=$B/hf_cache HF_HUB_OFFLINE=1 TF_USE_LEGACY_KERAS=1
export SWMS_REPO=$B/swms PYTHONPATH=$B/swm-next-qwen:$B/swms:$B/language-table
nice -n 15 python scripts/evaluate_planning.py --out $OUT "$@"
