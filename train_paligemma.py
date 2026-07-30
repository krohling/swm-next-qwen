"""Control-4: full fine-tune of SWM's own PaliGemmaWM architecture with OUR
training loop and OUR balanced-600 data.

Model = swms' released PaliGemmaWMForConditionalGeneration (action conditioning
built in), initialized from google/paligemma-3b-pt-224 with a fresh action
projector; processor/tokenizer from their OGB checkpoint dir (has the action
token). Loss/metrics/protocol identical to the Qwen arms: binary yes/no logit
BCE with yes-weight, per-epoch future-QA validation vs oracle + teacher,
collapse detectors, per-horizon accuracy.

If this reproduces SWM-class planning (~71% suite), our loop + data are
validated end-to-end and the Qwen gap is a modeling-choice question.

Usage:
    python train_paligemma.py --config configs/control4_paligemma.yaml [k=v ...]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.environ.get("SWMS_REPO", str(Path(__file__).parent.parent / "swms")))

from swm_next_qwen.data import FutureQADataset, collate
from train import action_stats, teacher_oracle_agreement, load_cfg  # reuse


class PaliGemmaWMTrainable:
    """Thin wrapper: batch assembly via their processor, binary yes/no logit."""

    def __init__(self, processor_path: str, base_model: str, device: str = "cuda"):
        from swm.paligemma_wm.processing_paligemma_wm import PaliGemmaWMProcessor
        from swm.paligemma_wm.modeling_paligemma_wm import PaliGemmaWMForConditionalGeneration
        from swm.paligemma_wm.configuration_paligemma_wm import PaliGemmaWMConfig

        self.processor = PaliGemmaWMProcessor.from_pretrained(processor_path)
        cfg = PaliGemmaWMConfig.from_pretrained(processor_path)
        # Base weights from google's release; action projector is a fresh init
        # (reported in missing_keys). Their tokenizer reuses a reserved token id
        # for actions, so no embedding resize is needed -- asserted below.
        self.model = PaliGemmaWMForConditionalGeneration.from_pretrained(
            base_model, config=cfg, torch_dtype=torch.float32,
            low_cpu_mem_usage=True, ignore_mismatched_sizes=True,
        )
        assert cfg.action_token_index < self.model.get_input_embeddings().num_embeddings
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False
        self.model = self.model.to(device)
        self.device = device

        tok = self.processor.tokenizer
        def one(s):
            ids = tok.encode(s)
            ids = [i for i in ids if i not in (tok.bos_token_id,)]
            assert len(ids) == 1, (s, ids)
            return ids[0]
        # SWM's ANSWER_OPTIONS convention (swm.constants): capitalized single tokens
        from swm.constants import ANSWER_OPTIONS
        self.yes_id = one(ANSWER_OPTIONS[0])
        self.no_id = one(ANSWER_OPTIONS[1])

    def forward_loss_ce(self, images, actions, questions, answers):
        """SWM's objective: teacher-forced suffix CE (PaliGemma convention).
        answers: list of "yes"/"no" strings."""
        acts = [a if isinstance(a, torch.Tensor) else torch.as_tensor(a, dtype=torch.float32) for a in actions]
        inputs = self.processor(text=list(questions), images=list(images), actions=acts,
                                suffix=list(answers), return_tensors="pt", padding="longest")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        for k in ("pixel_values", "action_values"):
            if k in inputs:
                inputs[k] = inputs[k].to(torch.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model(**inputs)
        return out.loss

    def forward_logit(self, images, actions, questions):
        """images: list of PIL (single frame, their convention);
        actions: list of (H_i, 5) fp32 z-scored; questions: list[str].
        Returns (B,) binary yes-vs-no logit."""
        acts = [a if isinstance(a, torch.Tensor) else torch.as_tensor(a, dtype=torch.float32) for a in actions]
        inputs = self.processor(text=list(questions), images=list(images), actions=acts,
                                return_tensors="pt", padding="longest",
                                tokenize_newline_separately=False)
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)
        if "action_values" in inputs:
            inputs["action_values"] = inputs["action_values"].to(torch.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model(**inputs)
        # Their tokenizer LEFT-pads: the last real token is index -1 for every
        # row (matches their get_scores). Assert so a config change can't
        # silently reintroduce the wrong-position readout.
        assert self.processor.tokenizer.padding_side == "left"
        logits = out.logits[:, -1, :].float()
        return logits[:, self.yes_id] - logits[:, self.no_id]


@torch.no_grad()
def validate(m: PaliGemmaWMTrainable, loader, amean, astd, device):
    m.model.eval()
    n = bce = 0
    hits_orc = n_orc = hits_tch = pred_yes = 0
    tp = fn = tn = fp = 0
    h_hits, h_n = {}, {}
    for batch in loader:
        acts = [(a - amean) / astd for a in batch["actions"]]
        imgs = [pair[1] for pair in batch["images"]]  # single current frame
        logit = m.forward_logit(imgs, acts, batch["question"])
        tgt = batch["target"].to(device)
        bce += torch.nn.functional.binary_cross_entropy_with_logits(
            logit, tgt, reduction="sum").item()
        pred = (logit > 0).float().cpu()
        pred_yes += int(pred.sum())
        orc = batch["oracle"]; has = orc >= 0
        hits_orc += (pred[has] == orc[has]).sum().item(); n_orc += int(has.sum())
        tp += int(((pred == 1) & (orc == 1)).sum()); fn += int(((pred == 0) & (orc == 1)).sum())
        tn += int(((pred == 0) & (orc == 0)).sum()); fp += int(((pred == 1) & (orc == 0)).sum())
        for j, a in enumerate(batch["actions"]):
            if orc[j] < 0: continue
            hb = min([1, 2, 4, 8, 16], key=lambda b: abs(b - a.shape[0]))
            h_hits[hb] = h_hits.get(hb, 0) + int(pred[j] == orc[j]); h_n[hb] = h_n.get(hb, 0) + 1
        hits_tch += (pred == (batch["teacher_p_yes"] >= 0.5).float()).sum().item()
        n += len(pred)
    m.model.train()
    yes_rec = tp / max(tp + fn, 1); no_rec = tn / max(tn + fp, 1)
    out = {"val/bce": bce / max(n, 1), "val/qa_acc_oracle": hits_orc / max(n_orc, 1),
           "val/qa_acc_teacher": hits_tch / max(n, 1), "val/yes_recall": yes_rec,
           "val/no_recall": no_rec, "val/balanced_acc": (yes_rec + no_rec) / 2,
           "val/pred_yes_rate": pred_yes / max(n, 1), "val/n": n}
    for b in sorted(h_hits):
        out[f"val/qa_acc_h{b}"] = h_hits[b] / h_n[b]
    return out


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

    man_eps = [e["id"] for e in (lambda m: m["episodes"] if isinstance(m, dict) else m)(
        json.load(open(Path(cfg["data_path"]) / "manifest.json")))]
    n_val = int(cfg.get("val_episodes", 30))
    train_ids, val_ids = man_eps[:-n_val], man_eps[-n_val:]
    if cfg.get("max_train_episodes"):
        train_ids = train_ids[: int(cfg["max_train_episodes"])]

    tds = FutureQADataset(cfg["data_path"], episode_ids=train_ids,
                          label_source=cfg["label_source"],
                          samples_per_epoch=int(cfg["samples_per_epoch"]), seed=cfg.get("seed", 0))
    vds = FutureQADataset(cfg["data_path"], episode_ids=val_ids,
                          label_source=cfg["label_source"],
                          samples_per_epoch=int(cfg.get("val_samples", 500)),
                          seed=1234, fixed_eval=True)
    tl = DataLoader(tds, batch_size=int(cfg["micro_batch"]), collate_fn=collate,
                    num_workers=int(cfg.get("workers", 6)), shuffle=False)
    vl = DataLoader(vds, batch_size=int(cfg["micro_batch"]), collate_fn=collate, num_workers=2)

    amean, astd = action_stats(tds)
    torch.save({"action_mean": amean, "action_std": astd}, out_dir / "action_stats.pt")
    toa = teacher_oracle_agreement(vds)
    print(f"frozen-teacher oracle agreement on val split: {toa:.4f}", flush=True)

    m = PaliGemmaWMTrainable(cfg["processor_path"], cfg.get("base_model", "google/paligemma-3b-pt-224"), device)
    params = [p for p in m.model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    print(f"trainable params: {n_params:,} (full FT)", flush=True)
    opt = torch.optim.AdamW(params, lr=float(cfg.get("lr", 1e-5)), weight_decay=0.01)

    accum = int(cfg["grad_accum"]); epochs = int(cfg["epochs"])
    steps_per_epoch = math.ceil(len(tds) / (int(cfg["micro_batch"]) * accum))
    total = steps_per_epoch * epochs; warmup = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(
        (s + 1) / warmup, 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))

    start_epoch, gstep = 0, 0
    ck = out_dir / "last.pt"
    if ck.exists():
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        m.model.load_state_dict(sd["model"]); opt.load_state_dict(sd["opt"])
        sched.load_state_dict(sd["sched"]); start_epoch, gstep = sd["epoch"], sd["gstep"]
        print(f"resumed at epoch {start_epoch}", flush=True)

    yes_w = float(cfg.get("yes_weight", 3.0))
    objective = str(cfg.get("objective", "suffix_ce"))
    print(f"objective: {objective}", flush=True)
    m.model.train(); t0 = time.time()
    for epoch in range(start_epoch, epochs):
        run_loss = n_seen = 0
        opt.zero_grad()
        for i, batch in enumerate(tl):
            acts = [(a - amean) / astd for a in batch["actions"]]
            imgs = [pair[1] for pair in batch["images"]]
            tgt = batch["target"].to(device)
            if objective == "suffix_ce":
                answers = ["yes" if v >= 0.5 else "no" for v in batch["target"]]
                loss = m.forward_loss_ce(imgs, acts, batch["question"], answers)
            else:
                logit = m.forward_logit(imgs, acts, batch["question"])
                w = 1.0 + (yes_w - 1.0) * tgt
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                    logit, tgt, reduction="none") * w).mean()
            (loss / accum).backward()
            run_loss += loss.item() * len(tgt); n_seen += len(tgt)
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(); gstep += 1
                if gstep % 10 == 0:
                    wandb.log({"train/loss": run_loss / n_seen, "train/lr": sched.get_last_lr()[0],
                               "train/samples_per_s": n_seen / (time.time() - t0),
                               "gstep": gstep, "epoch": epoch + i / len(tl)})
                    run_loss = n_seen = 0; t0 = time.time()
        vm = validate(m, vl, amean, astd, device)
        vm["epoch"] = epoch + 1; vm["val/teacher_oracle_agreement"] = toa
        wandb.log(vm); print(f"epoch {epoch+1}: {vm}", flush=True)
        torch.save({"model": m.model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch + 1, "gstep": gstep,
                    "cfg": cfg, "action_mean": amean, "action_std": astd},
                   out_dir / f"epoch{epoch+1}.pt")
        os.replace(out_dir / f"epoch{epoch+1}.pt", out_dir / "last.pt")
        torch.save({"model": m.model.state_dict(), "epoch": epoch + 1, "cfg": cfg,
                    "action_mean": amean, "action_std": astd}, out_dir / f"epoch{epoch+1}.pt")
    print("TRAINING_DONE", flush=True)
    run.finish()


if __name__ == "__main__":
    main()
