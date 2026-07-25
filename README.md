# swm-next-qwen

Action-conditioned LoRA fine-tuning of Qwen3-VL-8B as a future-state QA model
(SWM recipe, no separate predictor). Two supervision arms: oracle labels vs
self-labels (frozen Qwen's own answers on real future frames).

See the design spec in the SWM-Next Notion. Companion repo:
[krohling/dino_wm](https://github.com/krohling/dino_wm/tree/qwen3-vl-encoder)
(predictor-based arms + planning/eval harness).

```
python train.py --config configs/smoke.yaml          # 1-GPU smoke
python train.py --config configs/arm_a_oracle.yaml   # oracle arm
python train.py --config configs/arm_b_teacher.yaml  # self-label arm
```
