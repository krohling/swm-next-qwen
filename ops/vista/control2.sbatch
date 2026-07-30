#!/bin/bash
#SBATCH -J control2-swm-runner
#SBATCH -o /work/11599/krohling/vista/swm-followup/logs/control2-%j.out
#SBATCH -e /work/11599/krohling/vista/swm-followup/logs/control2-%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -A ASC25021
set -uo pipefail
W=$WORK/swm-followup
source $W/swms/.venv/bin/activate
cd $W/swm-next-qwen && git pull -q
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export SWMS_REPO=$W/swms PYTHONPATH=$W/swm-next-qwen:$W/swms:$W/language-table
mkdir -p $W/runs
python -u scripts/evaluate_planning.py --swm-ckpt $W/swms/ckpts/paligemma_wm_ogbench \
  --task-idx 0 --num-seeds 25 --out $W/runs/control_swm
