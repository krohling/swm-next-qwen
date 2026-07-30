#!/bin/bash
set -uo pipefail
B=/scratch/cluster/krohling/swm-followup
pick_idle() { nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '$2+0<1000 && $3+0<10 {print $1; exit}'; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | awk '{print $1+0}'; }
for T in 2 4; do
  until GPU=$(pick_idle); [ -n "$GPU" ]; do sleep 300; done
  bash $B/run_control.sh $GPU $B/runs/auto_eval/arm_a_e4_temp$T --ckpt $B/qwen_lora_runs/arm_a_epoch4.pt --task-idx 0 --num-seeds 25 --temperature $T > $B/logs/arm_a_e4_temp$T.log 2>&1 &
  echo "launched T=$T on GPU $GPU"
  until [ "$(gpu_used $GPU)" -gt 10000 ]; do sleep 30; done
done
wait
echo TEMP_SWEEP_DONE
