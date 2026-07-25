"""Save an epoch-0 checkpoint: freshly initialized adapter, no training.

LoRA B-matrices init to zero, so the LLM is EXACTLY base Qwen; only the
action projection is random. This is the 'before fine-tuning' planning
control: gradients through a random P carry no action information, so
closed-loop SR should match the base diffusion policy up to noise.
"""
import argparse
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from swm_next_qwen.data import FutureQADataset  # noqa: E402
from swm_next_qwen.model import ActionConditionedQwen  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds = FutureQADataset(args.data_path, label_source="teacher", samples_per_epoch=1)
    rows = [torch.as_tensor(ds._episode(i)["actions"], dtype=torch.float32)
            for i in range(min(200, len(ds.episodes)))]
    a = torch.cat(rows)
    amean, astd = a.mean(0), a.std(0).clamp_min(1e-6)

    torch.manual_seed(0)
    model = ActionConditionedQwen(device="cpu")
    cfg = {"action_dim": 5, "lora_r": 16, "lora_alpha": 32, "epoch": 0}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.trainable_state_dict(), "epoch": 0, "gstep": 0,
                "cfg": cfg, "action_mean": amean, "action_std": astd}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
