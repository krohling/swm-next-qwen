"""LoRA fine-tune Qwen3-VL as an action-conditioned future-state QA model.

Usage:
    python train.py --config configs/smoke.yaml [key=value overrides]

Per-epoch validation reports future-QA accuracy vs oracle AND vs the teacher
on a fixed held-out sample set -- the leading indicator for planning lift
(SWM's fine-tuned judge ~97% vs our frozen teacher's ~87% oracle agreement).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from swm_next_qwen.data import FutureQADataset, collate
from swm_next_qwen.model import ActionConditionedQwen


def load_cfg():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    for ov in args.overrides:
        k, v = ov.split("=", 1)
        d = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            d = d[p]
        old = d.get(parts[-1])
        d[parts[-1]] = type(old)(v) if old is not None and not isinstance(old, bool) \
            else (v.lower() == "true" if isinstance(old, bool) else v)
    return cfg


def action_stats(ds: FutureQADataset, n: int = 200):
    rows = []
    for i in range(0, min(n, len(ds.episodes))):
        rows.append(torch.as_tensor(ds._episode(i)["actions"], dtype=torch.float32))
    a = torch.cat(rows)
    return a.mean(0), a.std(0).clamp_min(1e-6)


@torch.no_grad()
def validate(model, loader, amean, astd, device):
    model.eval()
    n = bce = 0
    hits_orc = n_orc = hits_tch = 0
    for batch in loader:
        acts = [(a - amean) / astd for a in batch["actions"]]
        logit = model(batch["images"], acts, batch["question"])
        tgt = batch["target"].to(device)
        bce += torch.nn.functional.binary_cross_entropy_with_logits(
            logit, tgt, reduction="sum").item()
        pred = (logit > 0).float().cpu()
        orc = batch["oracle"]
        has = orc >= 0
        hits_orc += (pred[has] == orc[has]).sum().item()
        n_orc += int(has.sum())
        hits_tch += (pred == (batch["teacher_p_yes"] >= 0.5).float()).sum().item()
        n += len(pred)
    model.train()
    return {"val/bce": bce / max(n, 1),
            "val/qa_acc_oracle": hits_orc / max(n_orc, 1),
            "val/qa_acc_teacher": hits_tch / max(n, 1),
            "val/n": n, "val/n_oracle": n_orc}


def main():
    cfg = load_cfg()
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    torch.manual_seed(cfg.get("seed", 0))

    import wandb
    wid = cfg.get("wandb_id")
    kw = {"resume": "allow", "id": wid} if wid else {}
    run = wandb.init(project=cfg.get("wandb_project", "swm-next-qwen"),
                     name=cfg["run_name"], config=cfg, **kw)

    # ---- data: last val_episodes episodes held out
    man_eps = [e["id"] for e in (lambda m: m["episodes"] if isinstance(m, dict) else m)(
        json.load(open(Path(cfg["data_path"]) / "manifest.json")))]
    n_val = int(cfg.get("val_episodes", 30))
    train_ids, val_ids = man_eps[:-n_val], man_eps[-n_val:]
    if cfg.get("max_train_episodes"):
        train_ids = train_ids[: int(cfg["max_train_episodes"])]

    tds = FutureQADataset(cfg["data_path"], episode_ids=train_ids,
                          label_source=cfg["label_source"],
                          samples_per_epoch=int(cfg["samples_per_epoch"]),
                          seed=cfg.get("seed", 0))
    vds = FutureQADataset(cfg["data_path"], episode_ids=val_ids,
                          label_source=cfg["label_source"],
                          samples_per_epoch=int(cfg.get("val_samples", 500)),
                          seed=1234, fixed_eval=True)
    tl = DataLoader(tds, batch_size=int(cfg["micro_batch"]), collate_fn=collate,
                    num_workers=int(cfg.get("workers", 4)), shuffle=False,
                    persistent_workers=False)
    vl = DataLoader(vds, batch_size=int(cfg["micro_batch"]), collate_fn=collate,
                    num_workers=2)

    amean, astd = action_stats(tds)
    torch.save({"action_mean": amean, "action_std": astd}, out_dir / "action_stats.pt")

    # ---- model + optimizer
    model = ActionConditionedQwen(
        model_id=cfg.get("model_id", "Qwen/Qwen3-VL-8B-Instruct"),
        action_dim=int(cfg.get("action_dim", 5)),
        lora_r=int(cfg.get("lora_r", 16)), lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)), device=device)
    lora_p, act_p = model.trainable_parameters()
    opt = torch.optim.AdamW([
        {"params": lora_p, "lr": float(cfg.get("lr_lora", 1e-4))},
        {"params": act_p, "lr": float(cfg.get("lr_action", 3e-4))},
    ], weight_decay=0.01)

    accum = int(cfg["grad_accum"])
    epochs = int(cfg["epochs"])
    steps_per_epoch = math.ceil(len(tds) / (int(cfg["micro_batch"]) * accum))
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(0.03 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(
        (s + 1) / warmup, 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))))

    start_epoch, gstep = 0, 0
    ck_resume = out_dir / "last.pt"
    if ck_resume.exists():
        sd = torch.load(ck_resume, map_location="cpu", weights_only=False)
        model.load_trainable_state_dict(sd["model"])
        opt.load_state_dict(sd["opt"]); sched.load_state_dict(sd["sched"])
        start_epoch, gstep = sd["epoch"], sd["gstep"]
        print(f"resumed from {ck_resume} at epoch {start_epoch}, gstep {gstep}", flush=True)

    yes_w = float(cfg.get("yes_weight", 3.0))
    model.train()
    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        run_loss = n_seen = 0
        opt.zero_grad()
        for i, batch in enumerate(tl):
            acts = [(a - amean) / astd for a in batch["actions"]]
            logit = model(batch["images"], acts, batch["question"])
            tgt = batch["target"].to(device)
            w = 1.0 + (yes_w - 1.0) * tgt  # soft-target-aware yes upweighting
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                logit, tgt, reduction="none") * w).mean()
            (loss / accum).backward()
            run_loss += loss.item() * len(tgt); n_seen += len(tgt)
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(lora_p + act_p, 1.0)
                opt.step(); sched.step(); opt.zero_grad(); gstep += 1
                if gstep % 10 == 0:
                    wandb.log({"train/loss": run_loss / n_seen,
                               "train/lr": sched.get_last_lr()[0],
                               "train/samples_per_s": n_seen / (time.time() - t0),
                               "gstep": gstep, "epoch": epoch + i / len(tl)})
                    run_loss = n_seen = 0; t0 = time.time()

        vm = validate(model, vl, amean, astd, device)
        vm["epoch"] = epoch + 1
        wandb.log(vm)
        print(f"epoch {epoch+1}: {vm}", flush=True)
        torch.save({"model": model.trainable_state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch + 1, "gstep": gstep,
                    "cfg": cfg, "action_mean": amean, "action_std": astd},
                   out_dir / f"epoch{epoch+1}.pt")
        torch.save(torch.load(out_dir / f"epoch{epoch+1}.pt", weights_only=False),
                   ck_resume)

    print("TRAINING_DONE", flush=True)
    run.finish()


if __name__ == "__main__":
    main()
