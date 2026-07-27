"""Closed-loop planning eval for the (fine-tuned or epoch-0) LoRA judge.

Mirrors dino_wm's evaluate path: builds the adapter, then calls SWM's
swm_eval per seed with the standard OGB planner config. Requires SWMS_REPO.

Usage:
    python scripts/evaluate_planning.py --ckpt .../epoch0.pt --task-idx 0 \
        --seed-start 6000 --num-seeds 25 --out .../epoch0_suite_task0
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

THIS_REPO = Path(__file__).resolve().parent.parent
SWMS_REPO = Path(os.environ["SWMS_REPO"]).resolve()
sys.path.insert(0, str(THIS_REPO))
sys.path.insert(0, str(SWMS_REPO))

import torch

try:
    import ogbench  # noqa: F401
except ImportError:
    pass

from swm.evaluation import eval as swm_eval

from swm_next_qwen.planning_adapter import QwenLoraJudge

OGB_TASKS = [
    ("blue_cube", "green_cube"),
    ("blue_cube", "yellow_cube"),
    ("yellow_cube", "red_cube"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--task-idx", type=int, default=0)
    ap.add_argument("--seed-start", type=int, default=6000)
    ap.add_argument("--num-seeds", type=int, default=25)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gradient-lr", type=float, default=0.2)
    ap.add_argument("--swm-ckpt", default=None,
                    help="eval SWM's PaliGemma instead of our adapter (control)")
    args = ap.parse_args()

    combo = OGB_TASKS[args.task_idx]
    diffusion_path = str(SWMS_REPO / "ckpts" / "ogbench_base_diffusion"
                         / f"{combo[0]}_{combo[1]}.pt")
    model = None if args.swm_ckpt else QwenLoraJudge(ckpt_path=args.ckpt)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_succ = n_done = 0
    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        seed_dir = out / f"{seed}"
        if (seed_dir / "reward.pkl").exists():
            import pickle
            r = pickle.load(open(seed_dir / "reward.pkl", "rb"))
            n_succ += bool(r[0]); n_done += 1
            print(f"seed {seed}: cached ({bool(r[0])})", flush=True)
            continue
        if model is not None:
            model.reset_episode()
        success, t_min = swm_eval(
            seed=seed, reward_type="stack_blocks", env_type="ogbench",
            device="cuda", output_dir=str(seed_dir),
            ckpt_path=args.swm_ckpt or "", processor_path=args.swm_ckpt or "",
            model=model,
            diffusion_path=diffusion_path,
            diffusion=True, mppi=False, gradient=True, expert_diffusion=False,
            precision=torch.bfloat16, action_skip=8, model_batch_size=8,
            reward_kwargs={"block_combo": combo, "ood": False},
            action_dim=5, num_steps=50, num_actions_executed=4,
            pred_horizon=16, num_samples=1, num_planning_iters=20,
            gradient_lr=args.gradient_lr, gradient_clipping_value=10.0,
            mppi_temperature=1.0, intermediate_hm=False,
        )
        n_succ += bool(success); n_done += 1
        print(f"seed {seed}: success={bool(success)} ({t_min:.1f} min) "
              f"running {n_succ}/{n_done}", flush=True)
    print(f"FINAL task{args.task_idx}: {n_succ}/{n_done}", flush=True)


if __name__ == "__main__":
    main()
