"""Control-4c: literal transformers.Trainer replication of SWM's training.

Jacob (first author): "I just used the hugging face trainer class." This runs
exactly that -- their PaliGemmaWM model, processor-built suffix-CE labels,
transformers.Trainer with DEFAULT TrainingArguments (lr 5e-5 linear decay,
per-device batch 8, 3 epochs, weight_decay 0, max_grad_norm 1.0) -- with our
validation suite attached as a TrainerCallback for per-epoch curves.

Usage:
    python train_paligemma_hf.py --config configs/control4c_hf.yaml
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.environ.get("SWMS_REPO", str(Path(__file__).parent.parent / "swms")))

from swm_next_qwen.data import FutureQADataset, collate
from train import action_stats, teacher_oracle_agreement, load_cfg
from train_paligemma import PaliGemmaWMTrainable, validate


class HFWrapDataset(Dataset):
    """FutureQADataset -> processor-ready dict per sample (tokenization happens
    in the collator so padding is batch-local, matching Trainer conventions)."""

    def __init__(self, base: FutureQADataset, amean, astd):
        self.base, self.amean, self.astd = base, amean, astd

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        s = self.base[i]
        return {
            "image": s["images"][1],                       # single frame, their convention
            "action": (s["actions"] - self.amean) / self.astd,
            "question": s["question"],
            "answer": "yes" if float(s["target"]) >= 0.5 else "no",
        }


def make_collator(processor):
    def collate_hf(batch):
        inputs = processor(
            text=[b["question"] for b in batch],
            images=[b["image"] for b in batch],
            actions=[b["action"] for b in batch],
            suffix=[b["answer"] for b in batch],
            return_tensors="pt", padding="longest",
        )
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)
        if "action_values" in inputs:
            inputs["action_values"] = inputs["action_values"].to(torch.float32)
        return dict(inputs)
    return collate_hf


class ValCallback:
    """Per-epoch: run our validation suite and log through the Trainer."""

    def __init__(self, wrap, vloader, amean, astd, toa, out_dir):
        self.wrap, self.vl = wrap, vloader
        self.amean, self.astd, self.toa = amean, astd, toa
        self.out_dir = Path(out_dir)

    # transformers Callback protocol (duck-typed via TrainerCallback subclass below)


def main():
    from transformers import Trainer, TrainingArguments, TrainerCallback

    cfg = load_cfg()
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.get("seed", 0))

    import wandb
    wandb.init(project=cfg.get("wandb_project", "swm-next-qwen"),
               name=cfg["run_name"], config=cfg)

    man_eps = [e["id"] for e in (lambda m: m["episodes"] if isinstance(m, dict) else m)(
        json.load(open(Path(cfg["data_path"]) / "manifest.json")))]
    n_val = int(cfg.get("val_episodes", 30))
    train_ids, val_ids = man_eps[:-n_val], man_eps[-n_val:]

    tds_base = FutureQADataset(cfg["data_path"], episode_ids=train_ids,
                               label_source=cfg.get("label_source", "oracle"),
                               samples_per_epoch=int(cfg["samples_per_epoch"]),
                               seed=cfg.get("seed", 0))
    vds = FutureQADataset(cfg["data_path"], episode_ids=val_ids,
                          label_source=cfg.get("label_source", "oracle"),
                          samples_per_epoch=int(cfg.get("val_samples", 2000)),
                          seed=1234, fixed_eval=True)
    amean, astd = action_stats(tds_base)
    torch.save({"action_mean": amean, "action_std": astd}, out_dir / "action_stats.pt")
    toa = teacher_oracle_agreement(vds)
    print(f"frozen-teacher oracle agreement on val split: {toa:.4f}", flush=True)

    wrap = PaliGemmaWMTrainable(cfg["processor_path"],
                                cfg.get("base_model", "google/paligemma-3b-pt-224"))
    vl = DataLoader(vds, batch_size=8, collate_fn=collate, num_workers=2)

    class EpochVal(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kw):
            vm = validate(wrap, vl, amean, astd, "cuda")
            vm["epoch"] = state.epoch
            vm["val/teacher_oracle_agreement"] = toa
            wandb.log(vm)
            print(f"epoch {state.epoch}: {vm}", flush=True)
            sd = {"model": wrap.model.state_dict(), "epoch": int(round(state.epoch)),
                  "cfg": cfg, "action_mean": amean, "action_std": astd}
            torch.save(sd, out_dir / f"epoch{int(round(state.epoch))}.pt")

    # LITERAL DEFAULTS: only output_dir, epochs=3, logging, and disabling the
    # unused eval/save machinery are set. lr 5e-5 linear, batch 8, wd 0.0,
    # clip 1.0, AdamW -- all transformers defaults.
    targs = TrainingArguments(
        output_dir=str(out_dir / "hf_out"),
        num_train_epochs=float(cfg.get("epochs", 3)),
        logging_steps=25,
        save_strategy="no",
        report_to=["wandb"],
        bf16=True,
        dataloader_num_workers=int(cfg.get("workers", 6)),
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=wrap.model,
        args=targs,
        train_dataset=HFWrapDataset(tds_base, amean, astd),
        data_collator=make_collator(wrap.processor),
        callbacks=[EpochVal()],
    )
    trainer.train()
    print("TRAINING_DONE", flush=True)


if __name__ == "__main__":
    main()
