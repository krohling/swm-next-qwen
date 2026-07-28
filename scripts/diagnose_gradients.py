"""Instrumented gradient planning for the LoRA judge (ports dino_wm's
diagnose_planning_gradients): records per-iteration grad norms, reward climb,
and action displacement per replan. Compare against SWM's signature
(grads decay 10-30x, reward climbs +0.02..0.10) and the frozen-era signature
(no climb)."""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
THIS = Path(__file__).resolve().parent.parent
SWMS = Path(os.environ["SWMS_REPO"]).resolve()
sys.path.insert(0, str(THIS)); sys.path.insert(0, str(SWMS))

import torch
try: import ogbench  # noqa
except ImportError: pass
import swm.planning_algos as PA
from swm.evaluation import eval as swm_eval
from swm_next_qwen.planning_adapter import QwenLoraJudge

RECORDS = []

def instrumented(diffusion_model, get_rewards_fn, pln_cfg, ret_intermediate=False, action_samp=None):
    if action_samp is None:
        action_samp = PA.sample_initial_actions(diffusion_model=diffusion_model, pln_cfg=pln_cfg)
    orig = action_samp.clone()
    action_samp.requires_grad = True
    opt = torch.optim.SGD([action_samp], lr=pln_cfg.gradient_lr)
    rec = {"grad_pre": [], "reward": []}
    hist_a, hist_r = [], []
    for _ in range(pln_cfg.n_planning_itrs):
        opt.zero_grad()
        rewards, weighted, rsum = get_rewards_fn(action_samp, action_skip=pln_cfg.action_skip, gradient=True)
        if ret_intermediate:
            hist_a.append(action_samp.detach().cpu().numpy().copy()); hist_r.append(weighted.copy())
        (-rsum).backward()
        pre = torch.nn.utils.clip_grad_norm_([action_samp], max_norm=pln_cfg.gradient_clipping_value)
        opt.step()
        with torch.no_grad():
            action_samp.clamp_(-pln_cfg.max_action_value, pln_cfg.max_action_value)
        rec["grad_pre"].append(float(pre)); rec["reward"].append(float(rsum))
    d = action_samp.detach() - orig
    rec.update({"disp_l2": float(d.norm()), "disp_max": float(d.abs().max())})
    RECORDS.append(rec)
    out = action_samp.detach().numpy()
    return (out, rewards, weighted, hist_a, hist_r) if ret_intermediate else (out, rewards, weighted)

PA.plan_model_gradient = instrumented
TASKS = [("blue_cube","green_cube"),("blue_cube","yellow_cube"),("yellow_cube","red_cube")]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task-idx", type=int, default=0)
    ap.add_argument("--num-seeds", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    combo = TASKS[args.task_idx]
    model = QwenLoraJudge(ckpt_path=args.ckpt)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in range(6000, 6000 + args.num_seeds):
        RECORDS.clear(); model.reset_episode()
        s, t = swm_eval(seed=seed, reward_type="stack_blocks", env_type="ogbench", device="cuda",
            output_dir=str(out.parent / out.stem / f"ep_{seed}"), ckpt_path="", processor_path="",
            model=model, diffusion_path=str(SWMS / "ckpts" / "ogbench_base_diffusion" / f"{combo[0]}_{combo[1]}.pt"),
            diffusion=True, mppi=False, gradient=True, expert_diffusion=False,
            precision=torch.bfloat16, action_skip=8, model_batch_size=8,
            reward_kwargs={"block_combo": combo, "ood": False}, action_dim=5,
            num_steps=50, num_actions_executed=4, pred_horizon=16, num_samples=1,
            num_planning_iters=20, gradient_lr=0.2, gradient_clipping_value=10.0,
            mppi_temperature=1.0, intermediate_hm=False)
        results.append({"seed": seed, "success": bool(s), "replans": list(RECORDS)})
        import statistics as st
        if RECORDS:
            g0 = st.mean(r["grad_pre"][0] for r in RECORDS); gl = st.mean(r["grad_pre"][-1] for r in RECORDS)
            climb = st.mean(r["reward"][-1] - r["reward"][0] for r in RECORDS)
            disp = st.mean(r["disp_l2"] for r in RECORDS)
            print(f"seed {seed}: success={bool(s)} grad0={g0:.4g} gradN={gl:.4g} climb={climb:+.4f} disp={disp:.4f}", flush=True)
        json.dump(results, open(out, "w"))
    print("DIAG_DONE", flush=True)

if __name__ == "__main__":
    main()
