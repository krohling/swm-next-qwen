#!/bin/bash
#SBATCH -J control4-paligemma
#SBATCH -o /work/11599/krohling/vista/swm-followup/logs/control4-%j.out
#SBATCH -e /work/11599/krohling/vista/swm-followup/logs/control4-%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 30:00:00
#SBATCH -A ASC25021
#SBATCH --mail-user=kevin@coniferlabs.com
#SBATCH --mail-type=END,FAIL
set -uo pipefail
W=$WORK/swm-followup
source $W/swms/.venv/bin/activate
source $W/.wandb_env
cd $W/swm-next-qwen && git pull -q
export HF_HOME=$WORK/hf_cache WANDB_ENTITY=kevin_ai WANDB_DIR=$W/.wandb
export SWMS_REPO=$W/swms PYTHONPATH=$W/swm-next-qwen:$W/swms
python -u train_paligemma.py --config configs/control4_paligemma.yaml
