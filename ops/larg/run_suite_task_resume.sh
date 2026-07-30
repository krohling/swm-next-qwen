#!/bin/bash
set -uo pipefail
B=/scratch/cluster/krohling/swm-followup
CKPT=$1 TASK=$2 GPU=$3 OUT=$4
cd $B/dino_wm; source $B/.venv/bin/activate
export XDG_CACHE_HOME=$B/.xdg_cache XDG_DATA_HOME=$B/.xdg_data XDG_CONFIG_HOME=$B/.xdg_config
export TMPDIR=$B/.tmp MPLCONFIGDIR=$B/.mpl_cache
export CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl
export HF_HOME=$B/hf_cache HF_HUB_OFFLINE=1 TF_USE_LEGACY_KERAS=1
export SWMS_REPO=$B/swms PREDICTOR_CKPT=$CKPT DINO_WM_QWEN_EVAL_ROOT=$OUT
export PYTHONPATH=$B/dino_wm:$B/swms:$B/language-table
mkdir -p $OUT
echo START task$TASK $(date -Iseconds)
nice -n 15 python scripts/evaluate_qwen_lt.py --config-name eval_qwen_ogb \
  +task_idx=$TASK num_seeds=25 +use_deepstack_for_predictions=false 2>&1
echo "RESULT: $(grep -a success $OUT/results.txt 2>/dev/null | tail -1)"
