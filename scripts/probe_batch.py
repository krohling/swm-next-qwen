"""Micro-batch / gradient-checkpointing throughput probe.

Times forward+backward on worst-case synthetic batches (H=16 action tokens,
two 448px images, longest question) across a (micro_batch x grad_ckpt) grid.
Reports peak memory and samples/s per config; catches OOM and moves on.
Effective batch stays 64 in training -- this probe only picks the fastest
(micro_batch, grad_ckpt, grad_accum=64/micro_batch) combination.

Usage:  python scripts/probe_batch.py [--sizes 2,4,8,16,24,32]
"""
from __future__ import annotations

import argparse
import time

import torch
from PIL import Image

from swm_next_qwen.model import ActionConditionedQwen

QUESTION = "Is the blue cube stacked directly on top of the green cube?"


def make_batch(bs: int, image_size: int, action_dim: int, H: int = 16):
    img = Image.new("RGB", (image_size, image_size), (120, 120, 120))
    images = [[img, img] for _ in range(bs)]
    actions = [torch.randn(H, action_dim) for _ in range(bs)]
    questions = [QUESTION] * bs
    return images, actions, questions


def probe(model, bs: int, steps: int = 5):
    images, actions, questions = make_batch(bs, model.image_size, model.action_dim)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    # warmup
    logit = model(images, actions, questions)
    logit.sum().backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        logit = model(images, actions, questions)
        logit.sum().backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    peak = torch.cuda.max_memory_allocated() / 2**30
    return bs / dt, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="2,4,8,16,24,32")
    ap.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]

    results = []
    for ckpt in (True, False):
        model = ActionConditionedQwen(model_id=args.model_id, device="cuda")
        base = model._base
        if not ckpt:
            base.gradient_checkpointing_disable()
        for bs in sizes:
            try:
                sps, peak = probe(model, bs)
                results.append((ckpt, bs, sps, peak))
                print(f"grad_ckpt={ckpt}  micro_batch={bs:3d}  "
                      f"{sps:6.2f} samples/s  peak {peak:5.1f} GiB", flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"grad_ckpt={ckpt}  micro_batch={bs:3d}  OOM", flush=True)
                torch.cuda.empty_cache()
                break
        del model
        torch.cuda.empty_cache()

    best = max(results, key=lambda r: r[2])
    print(f"\nBEST: grad_ckpt={best[0]} micro_batch={best[1]} "
          f"({best[2]:.2f} samples/s, {best[3]:.1f} GiB) "
          f"-> grad_accum={max(1, 64 // best[1])}", flush=True)


if __name__ == "__main__":
    main()
