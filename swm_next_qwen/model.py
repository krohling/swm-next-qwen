"""Action-conditioned Qwen3-VL for future-state QA (SWM recipe, LoRA).

Sequence layout per sample:
    [chat preamble | image tokens (frame t-1) | image tokens (frame t) |
     "Actions:" | H action tokens | question "... Answer with one word: yes or no." | gen prompt]

Action tokens ride on placeholder text tokens: the prompt contains exactly H
copies of a reserved token; after the embedding lookup, a forward hook on
embed_tokens swaps those rows for projected action embeddings. This keeps
Qwen's own image splicing, DeepStack injection, and M-RoPE position math
completely stock (positions come from input_ids, which never change).
Variable horizon = variable prompt length; no mid-sequence attention masking.

Trainable: LoRA adapters on the LLM decoder layers + the action projection
(linear + LayerNorm + per-slot positional embedding). Vision tower, merger,
embeddings, and LM head stay frozen.
"""
from __future__ import annotations

import torch
import torch.nn as nn

PLACEHOLDER_CANDIDATES = ["<|fim_pad|>", "<|fim_prefix|>", "<|box_start|>"]


class ActionConditionedQwen(nn.Module):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        action_dim: int = 5,
        max_horizon: int = 16,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        device: str = "cuda",
        precision=torch.bfloat16,
        image_size: int = 448,
        lora: bool = True,
    ):
        super().__init__()
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.device = device
        self.precision = precision
        self.image_size = image_size
        self.action_dim = action_dim
        self.max_horizon = max_horizon

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, dtype=precision, low_cpu_mem_usage=True
        )
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

        if lora:
            from peft import LoraConfig, get_peft_model

            lcfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                # Regex pins adapters to the LLM decoder layers only -- the
                # vision tower and merger must stay stock.
                target_modules=r"model\.language_model\.layers\.\d+\."
                               r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
                               r"|mlp\.(gate_proj|up_proj|down_proj))",
            )
            model = get_peft_model(model, lcfg)
            model.print_trainable_parameters()
        self.model = model.to(device)

        # ---- action conditioning (new params, trained in full, fp32 master)
        d_llm = self._base.config.text_config.hidden_size
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, d_llm), nn.LayerNorm(d_llm)
        ).to(device, torch.float32)
        self.action_pos = nn.Parameter(
            torch.zeros(max_horizon, d_llm, device=device, dtype=torch.float32)
        )
        nn.init.trunc_normal_(self.action_pos, std=0.02)

        # ---- placeholder token for action slots
        self.placeholder_id = None
        for cand in PLACEHOLDER_CANDIDATES:
            ids = self.tokenizer(cand, add_special_tokens=False).input_ids
            if len(ids) == 1:
                self.placeholder_id, self.placeholder_str = ids[0], cand
                break
        assert self.placeholder_id is not None, "no single-token placeholder found"

        # ---- yes/no readout ids
        def singles(cands):
            out = []
            for c in cands:
                ids = self.tokenizer(c, add_special_tokens=False).input_ids
                if len(ids) == 1:
                    out.append(ids[0])
            return out

        self.yes_ids = singles([" Yes", " yes", "Yes", "yes"])
        self.no_ids = singles([" No", " no", "No", "no"])
        assert self.yes_ids and self.no_ids

        # ---- embed hook: swap placeholder rows for action embeddings
        self._pending_action_embeds: torch.Tensor | None = None  # (B, Hmax, d) padded
        self._pending_input_ids: torch.Tensor | None = None
        emb = self._base.get_input_embeddings()
        emb.register_forward_hook(self._inject_actions)

    @property
    def _base(self):
        m = self.model
        return m.get_base_model() if hasattr(m, "get_base_model") else m

    def _inject_actions(self, module, inp, out):
        if self._pending_action_embeds is None:
            return out
        ids = self._pending_input_ids
        mask = ids == self.placeholder_id  # (B, T)
        out = out.clone()
        act = self._pending_action_embeds.to(out.dtype)
        for b in range(ids.shape[0]):
            pos = mask[b].nonzero(as_tuple=False).squeeze(-1)
            out[b, pos] = act[b, : pos.numel()]
        return out

    def _build_batch(self, images, actions, questions):
        """images: list of [PIL, PIL]; actions: list of (H_i, action_dim) fp32
        (already z-scored by caller); questions: list of str."""
        texts = []
        for act, q in zip(actions, questions):
            H = act.shape[0]
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text",
                     "text": "Actions: " + self.placeholder_str * H
                             + f"\n{q} Answer with one word: yes or no."},
                ],
            }]
            texts.append(self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False))
        imgs = [[im.resize((self.image_size, self.image_size)) if im.size != (self.image_size, self.image_size) else im
                 for im in pair] for pair in images]
        inputs = self.processor(text=texts, images=imgs, return_tensors="pt", padding=True)
        return inputs.to(self.device)

    def forward(self, images, actions, questions):
        """Returns (B,) yes-vs-no binary logits (grad flows to LoRA + action params)."""
        B = len(questions)
        inputs = self._build_batch(images, actions, questions)

        # Project actions -> (B, Hmax, d_llm), padded; hook consumes per-row counts.
        d_llm = self.action_pos.shape[1]
        Hmax = max(a.shape[0] for a in actions)
        act_emb = torch.zeros(B, Hmax, d_llm, device=self.device)
        for b, a in enumerate(actions):
            h = a.shape[0]
            act_emb[b, :h] = self.action_proj(a.to(self.device, torch.float32)) \
                + self.action_pos[:h]
        self._pending_action_embeds = act_emb
        self._pending_input_ids = inputs["input_ids"]
        try:
            out = self.model(**inputs)
        finally:
            self._pending_action_embeds = None
            self._pending_input_ids = None

        # Last non-pad position per row -> restricted yes/no readout.
        last = inputs["attention_mask"].sum(dim=1) - 1  # (B,)
        logits = out.logits[torch.arange(B, device=self.device), last].float()
        yes = torch.logsumexp(logits[:, self.yes_ids], dim=-1)
        no = torch.logsumexp(logits[:, self.no_ids], dim=-1)
        return yes - no  # binary logit; sigmoid() = P(yes | {yes,no})

    # ---- persistence: adapters + action params only (MB-scale)
    def trainable_state_dict(self):
        from peft import get_peft_model_state_dict
        return {
            "lora": get_peft_model_state_dict(self.model),
            "action_proj": self.action_proj.state_dict(),
            "action_pos": self.action_pos.detach().cpu(),
        }

    def load_trainable_state_dict(self, sd):
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(self.model, sd["lora"])
        self.action_proj.load_state_dict(sd["action_proj"])
        with torch.no_grad():
            self.action_pos.copy_(sd["action_pos"].to(self.action_pos.device))

    def trainable_parameters(self):
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        action_params = list(self.action_proj.parameters()) + [self.action_pos]
        return lora_params, action_params
