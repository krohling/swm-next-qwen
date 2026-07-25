"""Planning adapter: the fine-tuned (or epoch-0) judge as SWM's reward model.

Implements the SWMModel API consumed by swm.evaluation.eval:
  - get_probabilistic_rewards_wm(action_seq, image, pred_horizon, questions,
        batch_size, action_skip, gradient)
  - get_scores(images, actions=None, questions=None)   # real-frame VQA
  - reset_episode()

No latent prediction anywhere: rewards come from a single forward of the
action-conditioned VLM per (candidate, horizon, question) row. Gradients flow
from P(yes) back to the planner's action leaf through the action projection.
"""
from __future__ import annotations

import contextlib
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from .model import ActionConditionedQwen


class QwenLoraJudge:
    def __init__(
        self,
        ckpt_path: str,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda",
        image_size: int = 448,
        max_action_horizon: int = 16,
    ):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ck.get("cfg", {})
        self.model = ActionConditionedQwen(
            model_id=model_id,
            action_dim=int(cfg.get("action_dim", 5)),
            max_horizon=max_action_horizon,
            lora_r=int(cfg.get("lora_r", 16)),
            lora_alpha=int(cfg.get("lora_alpha", 32)),
            lora_dropout=0.0,
            device=device,
            image_size=image_size,
        )
        self.model.load_trainable_state_dict(ck["model"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        # Planning backprops through the frozen weights to the actions;
        # gradient checkpointing keeps that affordable at batch>1.
        self.device = device
        self.action_dim = self.model.action_dim
        self.max_action_horizon = max_action_horizon
        self.action_mean = ck["action_mean"].float().to(device)
        self.action_std = ck["action_std"].float().to(device)
        self.reset_episode()

    def reset_episode(self):
        self._prev_image = None
        self._cur_key = None
        self._cur = None
        self._prev_for_cur = None

    def _history(self, image: Image.Image) -> list[Image.Image]:
        """Rolling 2-frame history (matches training: (frame_{t-1}, frame_t));
        duplicate at episode start. Keyed on content: the MPC loop calls us
        many times per outer step with the same frame."""
        key = hash(image.tobytes())
        if key != self._cur_key:
            self._prev_for_cur = self._cur if self._cur_key is not None else image
            self._cur = image
            self._cur_key = key
        return [self._prev_for_cur, self._cur]

    def get_probabilistic_rewards_wm(
        self,
        action_seq: torch.Tensor | np.ndarray,
        image: Image.Image,
        pred_horizon: int,
        questions: List[Tuple[str, str, float]],
        batch_size: int = 8,
        action_skip: int = 1,
        gradient: bool = False,
    ):
        if isinstance(action_seq, np.ndarray):
            action_seq = torch.from_numpy(action_seq)
        action_seq = action_seq.to(self.device, dtype=torch.float32)
        N, T_full, A = action_seq.shape
        assert A == self.action_dim, (A, self.action_dim)
        assert T_full <= self.max_action_horizon

        hist = self._history(image)
        h_steps = list(range(action_skip, pred_horizon + action_skip, action_skip))
        a_norm = (action_seq - self.action_mean.view(1, 1, -1)) / self.action_std.view(1, 1, -1)

        # rows = (question, candidate, h_step); keep autograd views of a_norm
        rows_img, rows_act, rows_q, meta = [], [], [], []
        for q_idx, (q_text, desired, weight) in enumerate(questions):
            for a_idx in range(N):
                for h in h_steps:
                    rows_img.append(hist)
                    rows_act.append(a_norm[a_idx, :h])
                    rows_q.append(q_text)
                    meta.append((q_idx, a_idx, h, desired, weight))

        rewards = np.zeros((len(questions), N, pred_horizon), dtype=np.float32)
        grad_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        ctx = contextlib.nullcontext() if gradient else torch.no_grad()
        with ctx:
            for s in range(0, len(meta), batch_size):
                e = min(s + batch_size, len(meta))
                logit = self.model(rows_img[s:e], rows_act[s:e], rows_q[s:e])
                p_yes = torch.sigmoid(logit.float())
                for k, (q_idx, a_idx, h, desired, weight) in enumerate(meta[s:e]):
                    p = p_yes[k] if str(desired).lower().startswith("y") else 1.0 - p_yes[k]
                    rewards[q_idx, a_idx, h - action_skip : h] = float(p.detach().cpu())
                    if gradient:
                        grad_sum = grad_sum + p * weight

        weighted = rewards.copy()
        for q_idx, (_, _, weight) in enumerate(questions):
            weighted[q_idx] *= weight
        if gradient:
            return rewards, weighted, grad_sum
        return rewards, weighted

    @torch.no_grad()
    def get_scores(self, images, actions=None, questions=None):
        """Real-frame VQA for subgoal tracking. Single image, no actions
        (H=0 -> the prompt simply has no action segment)."""
        if isinstance(questions, str):
            questions = [questions] * len(images)
        pil = []
        for img in images:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.asarray(img, dtype=np.uint8))
            pil.append(img)
        p_yes_all = []
        for s in range(0, len(pil), 8):
            e = min(s + 8, len(pil))
            imgs = [[im, im] for im in pil[s:e]]  # 2-slot layout, duplicated frame
            acts = [torch.zeros(0, self.action_dim) for _ in range(e - s)]
            logit = self.model(imgs, acts, [str(q) for q in questions[s:e]])
            p_yes_all.append(torch.sigmoid(logit.float()).cpu())
        p_yes = torch.cat(p_yes_all)
        return p_yes, 1.0 - p_yes
