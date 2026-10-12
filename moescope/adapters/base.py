"""Model adapters: where the router lives, what it returns, and how it picks experts.

Each architecture family gets one small subclass. Everything else (expert count, top-k,
expert size) is read from the loaded model at runtime, never hardcoded.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import nn


class Adapter:
    family: ClassVar[str]
    model_types: ClassVar[tuple[str, ...]]
    router_classes: ClassVar[tuple[str, ...]]
    # Positions of each tensor in the router module's output tuple.
    logits_at: ClassVar[int]
    weights_at: ClassVar[int]
    indices_at: ClassVar[int]

    def __init__(self, config: Any):
        self.config = config

    @classmethod
    def matches(cls, config: Any) -> bool:
        return getattr(config, "model_type", None) in cls.model_types

    # ---- static facts, read from the config ---------------------------------------------
    @property
    def num_experts(self) -> int:
        for attr in ("num_local_experts", "num_experts", "n_routed_experts"):
            value = getattr(self.config, attr, None)
            if isinstance(value, int):
                return value
        raise ValueError(f"{self.family}: config has no expert count field")

    @property
    def top_k(self) -> int:
        value = getattr(self.config, "num_experts_per_tok", None)
        if not isinstance(value, int):
            raise ValueError(f"{self.family}: config has no num_experts_per_tok")
        return value

    @property
    def shared_experts(self) -> list[int]:
        """Shared experts inside the routed index space. Families whose shared expert is a
        separate always-on MLP (Qwen2-MoE, GraniteMoeShared) return [] here: that MLP is never
        routed, so it is always resident and never a pruning candidate anyway."""
        return []

    # ---- module discovery ---------------------------------------------------------------
    def find_routers(self, model: nn.Module) -> list[tuple[str, nn.Module]]:
        """Router modules in layer order. Index in this list is the MoE layer index."""
        routers = [(n, m) for n, m in model.named_modules() if type(m).__name__ in self.router_classes]
        if not routers:
            found = sorted({type(m).__name__ for m in model.modules() if "Router" in type(m).__name__ or "Gate" in type(m).__name__})
            raise RuntimeError(
                f"{self.family} adapter found no router modules of type {self.router_classes}. "
                f"Router-like modules present: {found or 'none'}. The installed transformers version "
                "may have renamed them; update the adapter."
            )
        return routers

    def expert_bytes(self, model: nn.Module) -> int:
        """Bytes of one routed expert in one layer, measured from the first MoE block."""
        name, _ = self.find_routers(model)[0]
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        experts = getattr(parent, "experts", None)
        if experts is None:
            raise RuntimeError(f"{self.family}: no `experts` module next to router {name}")
        total = sum(p.numel() * p.element_size() for p in experts.parameters())
        return total // self.num_experts

    # ---- router outputs -----------------------------------------------------------------
    def unpack(self, out: tuple) -> tuple[torch.Tensor, torch.Tensor]:
        """(logits (T, E), indices (T, k)) from the router's output tuple."""
        return out[self.logits_at], out[self.indices_at]

    def select(self, logits: torch.Tensor, ref_weights_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Reproduce the router's choice from (possibly masked) logits: (indices, weights)."""
        raise NotImplementedError

    def repack(self, out: tuple, logits: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> tuple:
        new = list(out)
        new[self.logits_at], new[self.indices_at], new[self.weights_at] = logits, indices, weights
        return tuple(new)


class LogitsTopKAdapter(Adapter):
    """Granite style: top-k on raw logits, then softmax over the chosen k."""

    logits_at, weights_at, indices_at = 2, 1, 0

    def select(self, logits, ref_weights_dtype):
        top_logits, indices = logits.float().topk(self.top_k, dim=-1)
        return indices, torch.softmax(top_logits, dim=-1).to(ref_weights_dtype)


class SoftmaxTopKAdapter(Adapter):
    """Mixtral / Qwen / OLMoE style: softmax over all experts, top-k, optional renormalize."""

    logits_at, weights_at, indices_at = 0, 1, 2
    always_renormalize: ClassVar[bool] = False

    @property
    def renormalize(self) -> bool:
        return self.always_renormalize or bool(getattr(self.config, "norm_topk_prob", False))

    def select(self, logits, ref_weights_dtype):
        probs = torch.softmax(logits.float(), dim=-1)
        top, indices = probs.topk(self.top_k, dim=-1)
        if self.renormalize:
            top = top / top.sum(dim=-1, keepdim=True)
        return indices, top.to(ref_weights_dtype)
