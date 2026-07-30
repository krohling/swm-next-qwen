#!/bin/bash
#SBATCH -J swm-ref-val
#SBATCH -o /work/11599/krohling/vista/swm-followup/logs/swmref-%j.out
#SBATCH -e /work/11599/krohling/vista/swm-followup/logs/swmref-%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -A ASC25021
set -uo pipefail
W=$WORK/swm-followup
source $W/swms/.venv/bin/activate
cd $W/swm-next-qwen && git pull -q
export HF_HOME=$WORK/hf_cache SWMS_REPO=$W/swms PYTHONPATH=$W/swm-next-qwen:$W/swms
python -u scripts/val_swm_reference.py --config configs/control4b_ce.yaml
