"""Reference row: SWM's released fine-tuned PaliGemmaWM checkpoint evaluated
with OUR validation harness on OUR balanced-600 val split -- the like-for-like
target for control-4b's curves, and a calibration of the val methodology
against a known-good judge.

Usage: python scripts/val_swm_reference.py --config configs/control4b_ce.yaml
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import yaml
from torch.utils.data import DataLoader

from swm_next_qwen.data import FutureQADataset, collate
from train import action_stats, teacher_oracle_agreement
from train_paligemma import PaliGemmaWMTrainable, validate

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
args = ap.parse_args()
cfg = yaml.safe_load(open(args.config))

man_eps = [e["id"] for e in (lambda m: m["episodes"] if isinstance(m, dict) else m)(
    json.load(open(Path(cfg["data_path"]) / "manifest.json")))]
val_ids = man_eps[-int(cfg.get("val_episodes", 30)):]
vds = FutureQADataset(cfg["data_path"], episode_ids=val_ids, label_source="oracle",
                      samples_per_epoch=int(cfg.get("val_samples", 2000)),
                      seed=1234, fixed_eval=True)
vl = DataLoader(vds, batch_size=16, collate_fn=collate, num_workers=4)

tds = FutureQADataset(cfg["data_path"], label_source="oracle", samples_per_epoch=1)
rows = [torch.as_tensor(tds._episode(i)["actions"], dtype=torch.float32)
        for i in range(min(200, len(tds.episodes)))]
a = torch.cat(rows); amean, astd = a.mean(0), a.std(0).clamp_min(1e-6)

print(f"frozen-teacher ref on this split: {teacher_oracle_agreement(vds):.4f}", flush=True)
# Load their FINE-TUNED ckpt wholesale (weights + action projector + config)
m = PaliGemmaWMTrainable(cfg["processor_path"], cfg["processor_path"])
vm = validate(m, vl, amean, astd, "cuda")
print("SWM-REFERENCE:", {k: round(v, 4) for k, v in vm.items()}, flush=True)
