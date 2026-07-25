"""Future-QA dataset for action-conditioned VLM fine-tuning.

A sample is (frame_{t-1}, frame_t, actions_{t:t+H-1}, question, label) with
H ~ U{1..16} and the label read at frame t+H:
  label_source="teacher": frozen Qwen3-VL's P(yes) on the real future frame (soft)
  label_source="oracle":  simulator ground truth (hard 0/1)

Both label sources come from the teacher sidecar JSONs
(data_path/teacher/<episode_id>.json: per-frame lists of
{"q": str, "p_yes": float, "oracle": bool|None}), the same artifacts used by
dino_wm's DistillDataset. Yes/no question sampling is balanced per sample
(labels are ~77% "no"; unbalanced draws collapse training to constant-no).
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class TeacherStore:
    def __init__(self, data_path: str):
        self.dir = Path(data_path) / "teacher"
        if not self.dir.is_dir():
            raise RuntimeError(f"{self.dir} missing (teacher sidecars required)")
        self._cache: dict[str, list] = {}

    def entries(self, episode_id: str, frame_idx: int) -> list:
        if episode_id not in self._cache:
            p = self.dir / f"{episode_id}.json"
            self._cache[episode_id] = json.load(open(p)) if p.exists() else []
        frames = self._cache[episode_id]
        if not frames or frame_idx >= len(frames):
            return []
        return [e for e in frames[frame_idx] if e]


def _to_pil(fr) -> Image.Image:
    if isinstance(fr, (bytes, bytearray)):
        return Image.open(io.BytesIO(fr)).convert("RGB")
    t = torch.as_tensor(fr)
    if t.ndim == 3 and t.shape[0] in (1, 3):
        t = t.permute(1, 2, 0)
    return Image.fromarray(t.numpy().astype("uint8"))


class FutureQADataset(Dataset):
    def __init__(
        self,
        data_path: str,
        episode_ids: list[str] | None = None,
        label_source: str = "teacher",
        max_horizon: int = 16,
        obs_horizon: int = 2,
        samples_per_epoch: int | None = None,
        seed: int = 0,
        fixed_eval: bool = False,
    ):
        assert label_source in ("teacher", "oracle")
        self.data_path = Path(data_path)
        self.label_source = label_source
        self.max_horizon = max_horizon
        self.obs_horizon = obs_horizon
        self.teacher = TeacherStore(data_path)
        self.fixed_eval = fixed_eval
        self._rng = np.random.default_rng(seed)
        self._ep_cache: dict[str, dict] = {}

        man = json.load(open(self.data_path / "manifest.json"))
        eps = man["episodes"] if isinstance(man, dict) else man
        if episode_ids is not None:
            keep = set(episode_ids)
            eps = [e for e in eps if e["id"] in keep]
        self.episodes = eps

        # Enumerate valid (ep, t) starts: need t-1 >= 0 and t+H <= length-1.
        self.starts = []
        for i, e in enumerate(eps):
            L = int(e["length"])
            for t in range(1, L - 1):
                self.starts.append((i, t))
        if fixed_eval:
            # Deterministic eval set: fixed (start, H, entry) triples.
            rng = np.random.default_rng(seed)
            idxs = rng.permutation(len(self.starts))
            self.eval_samples = []
            for j in idxs:
                ep_idx, t = self.starts[j]
                L = int(eps[ep_idx]["length"])
                H = int(rng.integers(1, min(max_horizon, L - 1 - t) + 1))
                entries = self._entries(ep_idx, t + H)
                if not entries:
                    continue
                e = entries[int(rng.integers(len(entries)))]
                self.eval_samples.append((ep_idx, t, H, e))
                if samples_per_epoch and len(self.eval_samples) >= samples_per_epoch:
                    break
            self.n = len(self.eval_samples)
        else:
            self.n = samples_per_epoch or len(self.starts)

    def _entries(self, ep_idx: int, frame: int) -> list:
        entries = self.teacher.entries(self.episodes[ep_idx]["id"], frame)
        if self.label_source == "oracle":
            entries = [e for e in entries if e.get("oracle") is not None]
        return entries

    def _target_of(self, e) -> float:
        if self.label_source == "teacher":
            return float(e["p_yes"])
        return 1.0 if e.get("oracle") else 0.0

    def _episode(self, ep_idx: int) -> dict:
        eid = self.episodes[ep_idx]["id"]
        if eid not in self._ep_cache:
            if len(self._ep_cache) > 24:  # bound worker RSS
                self._ep_cache.pop(next(iter(self._ep_cache)))
            fn = self.episodes[ep_idx]["filename"]
            self._ep_cache[eid] = torch.load(
                self.data_path / fn, map_location="cpu", weights_only=False
            )
        return self._ep_cache[eid]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.fixed_eval:
            ep_idx, t, H, e = self.eval_samples[idx]
        else:
            # Rejection-sample a start whose target frame has questions.
            for _ in range(50):
                ep_idx, t = self.starts[self._rng.integers(len(self.starts))]
                L = int(self.episodes[ep_idx]["length"])
                H = int(self._rng.integers(1, min(self.max_horizon, L - 1 - t) + 1))
                entries = self._entries(ep_idx, t + H)
                if entries:
                    break
            else:
                raise RuntimeError("no labeled samples found in 50 draws")
            # Balanced yes/no draw (see module docstring).
            yes = [x for x in entries if self._target_of(x) >= 0.5]
            no = [x for x in entries if self._target_of(x) < 0.5]
            pool = yes if yes and (not no or self._rng.random() < 0.5) else (no or yes)
            e = pool[self._rng.integers(len(pool))]

        d = self._episode(ep_idx)
        frames, actions = d["frames"], torch.as_tensor(d["actions"], dtype=torch.float32)
        return {
            "images": [_to_pil(frames[t - 1]), _to_pil(frames[t])],
            "actions": actions[t : t + H],          # (H, action_dim)
            "question": e["q"],
            "target": torch.tensor(self._target_of(e), dtype=torch.float32),
            "oracle": torch.tensor(
                -1.0 if e.get("oracle") is None else float(bool(e["oracle"]))
            ),
            "teacher_p_yes": torch.tensor(float(e["p_yes"])),
        }


def collate(batch):
    """Variable-H, variable-length prompts: keep python lists; model batches."""
    return {
        "images": [b["images"] for b in batch],
        "actions": [b["actions"] for b in batch],
        "question": [b["question"] for b in batch],
        "target": torch.stack([b["target"] for b in batch]),
        "oracle": torch.stack([b["oracle"] for b in batch]),
        "teacher_p_yes": torch.stack([b["teacher_p_yes"] for b in batch]),
    }
