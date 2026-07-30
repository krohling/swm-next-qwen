import os, sys, pickle
from pathlib import Path
sys.path.insert(0, "/scratch/cluster/krohling/swm-followup/swms"); sys.path.insert(0, "/scratch/cluster/krohling/swm-followup/swm-next-qwen")
import torch
try: import ogbench
except ImportError: pass
from swm.evaluation import eval as swm_eval
import swm.planning_algos as PA
from PIL import Image as _Img
import numpy as _np
def _hm_stub(env, obs, actions, rewards, *a, **k):
    arr = _np.asarray(obs, dtype=_np.uint8)
    if arr.ndim != 3: arr = _np.zeros((64,64,3), dtype=_np.uint8)
    return [_Img.fromarray(arr)]
PA.get_heat_map = _hm_stub
from swm_next_qwen.planning_adapter import QwenLoraJudge
JUDGE = QwenLoraJudge(ckpt_path=f"{chr(47)}scratch{chr(47)}cluster{chr(47)}krohling{chr(47)}swm-followup{chr(47)}qwen_lora_runs{chr(47)}arm_a_epoch4.pt")
B = "/scratch/cluster/krohling/swm-followup"
combo = ("blue_cube", "green_cube")
out = Path(f"{B}/runs/auto_eval/base_seedmatch_task0"); out.mkdir(parents=True, exist_ok=True)
succ = 0
for seed in range(6000, 6025):
    sd = out / str(seed)
    if (sd/"reward.pkl").exists():
        succ += bool(pickle.load(open(sd/"reward.pkl","rb"))[0]); print(f"{seed}: cached"); continue
    JUDGE.reset_episode()
    s, t = swm_eval(seed=seed, reward_type="stack_blocks", env_type="ogbench", device="cuda",
        output_dir=str(sd), ckpt_path="", processor_path="", model=JUDGE,
        diffusion_path=f"{B}/swms/ckpts/ogbench_base_diffusion/{combo[0]}_{combo[1]}.pt",
        diffusion=True, mppi=False, gradient=False, expert_diffusion=False,
        precision=torch.bfloat16, action_skip=8, model_batch_size=8,
        reward_kwargs={"block_combo": combo, "ood": False}, action_dim=5,
        num_steps=50, num_actions_executed=4, pred_horizon=16, num_samples=1,
        num_planning_iters=20, gradient_lr=0.2, gradient_clipping_value=10.0,
        mppi_temperature=1.0, intermediate_hm=False)
    succ += bool(s); print(f"{seed}: success={bool(s)} running {succ}", flush=True)
print(f"BASE FINAL: {succ}/25")
