"""Hugging Face transformers backend: forward hooks on each MoE layer's router."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch
from torch import nn

from ..adapters.base import Adapter
from ..adapters.registry import adapter_for
from ..prompts import Prompt
from ..trace.schema import RouterEvent
from .base import ModelInfo


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class RouterRecorder:
    """Hooks every router; each forward pass appends one (ids, weights, logit_max) per layer.

    Weights are recomputed as a softmax over the chosen experts' logits, so every family
    reports the same quantity regardless of its own renormalization convention.
    """

    def __init__(self, adapter: Adapter, routers: list[tuple[str, nn.Module]]):
        self.adapter = adapter
        self.routers = routers
        self.calls: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._handles: list = []

    def _hook(self, layer: int):
        def hook(module, inputs, out):
            logits, indices = self.adapter.unpack(out)
            logits = logits.detach().float()
            weights = torch.softmax(logits.gather(-1, indices), dim=-1)
            weights, order = weights.sort(dim=-1, descending=True)
            self.calls.append((layer, indices.gather(-1, order), weights, logits.max(dim=-1).values))

        return hook

    def __enter__(self) -> RouterRecorder:
        self._handles = [m.register_forward_hook(self._hook(i)) for i, (_, m) in enumerate(self.routers)]
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def drain(self) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
        """Move everything captured since the last drain to CPU in one transfer."""
        if not self.calls:
            return []
        layers = [c[0] for c in self.calls]
        sizes = [c[1].shape[0] for c in self.calls]
        ids = torch.cat([c[1] for c in self.calls]).cpu().numpy()
        weights = torch.cat([c[2] for c in self.calls]).cpu().numpy()
        lmax = torch.cat([c[3] for c in self.calls]).cpu().numpy()
        self.calls = []
        out, start = [], 0
        for layer, n in zip(layers, sizes):
            out.append((layer, ids[start : start + n], weights[start : start + n], lmax[start : start + n]))
            start += n
        return out


class HFBackend:
    name = "hf"

    def __init__(
        self,
        device: str | None = None,
        dtype: str | None = None,
        max_new_tokens: int = 128,
        chat_template: bool = True,
    ):
        self.device = device or default_device()
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.chat_template = chat_template
        self.model = self.tokenizer = self.adapter = self.info = None

    # ---- loading --------------------------------------------------------------------------
    def load(self, model_id: str) -> ModelInfo:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        config = AutoConfig.from_pretrained(model_id)
        adapter_for(config)  # fail fast, before downloading weights
        dtype = getattr(torch, self.dtype) if self.dtype else "auto"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return self.attach(model, tokenizer, model_id)

    def attach(self, model: nn.Module, tokenizer, model_id: str) -> ModelInfo:
        """Use an already-constructed model (also how tests inject tiny random models)."""
        self.adapter = adapter_for(model.config)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.routers = self.adapter.find_routers(self.model)
        self.info = ModelInfo(
            model_id=model_id,
            family=self.adapter.family,
            num_layers=len(self.routers),
            num_experts=self.adapter.num_experts,
            top_k=self.adapter.top_k,
            shared_experts=self.adapter.shared_experts,
            expert_bytes=self.adapter.expert_bytes(self.model),
            dtype=str(next(self.model.parameters()).dtype).removeprefix("torch."),
        )
        return self.info

    @torch.no_grad()
    def self_test(self, num_tokens: int = 8) -> None:
        """Check the hooks against the model's own router logits before trusting a trace."""
        vocab = self.model.config.vocab_size
        ids = torch.arange(1, num_tokens + 1, device=self.device)[None] % vocab
        raw_logits: list[torch.Tensor] = []
        raw = [m.register_forward_hook(lambda m, i, o: raw_logits.append(self.adapter.unpack(o)[0])) for _, m in self.routers]
        try:
            with RouterRecorder(self.adapter, self.routers) as rec:
                out = self.model(ids, output_router_logits=True)
                calls = rec.calls
        finally:
            for h in raw:
                h.remove()
        if len(calls) != len(self.routers):
            raise RuntimeError(f"self-test: {len(calls)} router hooks fired, expected {len(self.routers)}")
        # Prefer the model's own recorded router logits; fall back to the router outputs.
        reference = getattr(out, "router_logits", None) or raw_logits
        for (layer, hook_ids, _, _), logits in zip(calls, reference):
            ref_ids, _ = self.adapter.select(logits.reshape(-1, logits.shape[-1]), torch.float32)
            if not torch.equal(ref_ids.sort(-1).values, hook_ids.sort(-1).values):
                raise RuntimeError(
                    f"self-test: layer {layer} hook top-k disagrees with the model's router logits; "
                    f"the {self.adapter.family} adapter does not match this transformers version"
                )

    # ---- tracing ----------------------------------------------------------------------------
    def encode(self, text: str) -> torch.Tensor:
        tok = self.tokenizer
        if self.chat_template and getattr(tok, "chat_template", None):
            text = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
            ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids
        else:
            ids = tok(text, return_tensors="pt").input_ids
        return ids.to(self.device)

    def eos_ids(self) -> set[int]:
        ids = set()
        for source in (getattr(self.model, "generation_config", None), self.tokenizer):
            eos = getattr(source, "eos_token_id", None)
            ids.update(eos if isinstance(eos, list) else [eos] if eos is not None else [])
        return ids

    def run(self, prompts: list[Prompt]) -> Iterator[RouterEvent]:
        for p in prompts:
            yield from self.trace_ids(self.encode(p.prompt), p.id, p.category)

    @torch.no_grad()
    def trace_ids(self, input_ids: torch.Tensor, prompt_id: int = 0, category: str = "default") -> Iterator[RouterEvent]:
        """Prefill once, then greedy-decode up to max_new_tokens, one event per layer per step."""
        eos = self.eos_ids()
        n = input_ids.shape[1]
        with RouterRecorder(self.adapter, self.routers) as rec:
            out = self.model(input_ids, use_cache=True, logits_to_keep=1)
            yield from self._events(rec, prompt_id, category, "prefill", start=0)
            past = out.past_key_values
            next_id = out.logits[:, -1].argmax(-1, keepdim=True)
            for step in range(self.max_new_tokens):
                if int(next_id) in eos:
                    break
                out = self.model(next_id, past_key_values=past, use_cache=True)
                yield from self._events(rec, prompt_id, category, "decode", start=n + step)
                past = out.past_key_values
                next_id = out.logits[:, -1].argmax(-1, keepdim=True)

    def _events(self, rec: RouterRecorder, prompt_id, category, phase, start) -> Iterator[RouterEvent]:
        for layer, ids, weights, lmax in rec.drain():
            yield RouterEvent(
                prompt_id=prompt_id,
                category=category,
                phase=phase,
                layer=layer,
                token_pos=np.arange(start, start + len(ids), dtype=np.int32),
                expert_ids=ids,
                weights=weights,
                logit_max=lmax,
            )
