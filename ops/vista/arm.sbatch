#!/bin/bash
#SBATCH -J swm-lora-arm
#SBATCH -o /work/11599/krohling/vista/swm-followup/logs/arm-%x-%j.out
#SBATCH -e /work/11599/krohling/vista/swm-followup/logs/arm-%x-%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -A ASC25021
#SBATCH --mail-user=kevin@coniferlabs.com
#SBATCH --mail-type=END,FAIL
set -uo pipefail
ARM=$1   # arm_a_oracle | arm_b_teacher
W=$WORK/swm-followup
source $W/swms/.venv/bin/activate
source $W/.wandb_env
export HF_HOME=$WORK/hf_cache
export WANDB_ENTITY=kevin_ai WANDB_DIR=$W/.wandb
export PYTHONPATH=$W/swm-next-qwen
mkdir -p $W/.wandb $W/qwen_lora_runs
cd $W/swm-next-qwen
python -u train.py --config configs/$ARM.yaml \
  data_path=$W/world-model-data/ogb-recipe-balanced \
  out_dir=$W/qwen_lora_runs/$ARM \
  micro_batch=8 grad_accum=8 workers=8
