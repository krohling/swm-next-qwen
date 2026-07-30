#!/bin/bash
#SBATCH -J lora-setup-probe
#SBATCH -o /work/11599/krohling/vista/swm-followup/logs/probe-%j.out
#SBATCH -e /work/11599/krohling/vista/swm-followup/logs/probe-%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A ASC25021
set -uo pipefail
W=$WORK/swm-followup
for V in "$W/labeler-eval/.venv" "$W/swms/.venv"; do
  [ -f "$V/bin/activate" ] || continue
  if "$V/bin/python" -W ignore -c "from transformers import Qwen3VLForConditionalGeneration" 2>/dev/null; then
    VENV=$V; break
  fi
done
[ -z "${VENV:-}" ] && { echo "FATAL: no Qwen3VL-capable venv"; exit 1; }
echo "CHOSEN_VENV=$VENV"
source $VENV/bin/activate
python -c "import peft" 2>/dev/null || python -m pip install -q peft pyyaml wandb || { python -m ensurepip && python -m pip install -q peft pyyaml wandb; }
python -W ignore -c "import transformers, torch, peft; print(\"VERSIONS\", transformers.__version__, torch.__version__, peft.__version__)"
cd $W
[ -d swm-next-qwen ] && (cd swm-next-qwen && git pull -q) || git clone -q https://github.com/krohling/swm-next-qwen.git
echo "REPO $(cd swm-next-qwen && git log --oneline -1)"
export HF_HOME=$WORK/hf_cache
cd $W/swm-next-qwen; export PYTHONPATH=$W/swm-next-qwen
python -u scripts/probe_batch.py --sizes 2,4,8,16,24,32,48
echo "PROBE_DONE"
