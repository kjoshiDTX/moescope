"""Mask experts in the router: their logits become -inf, so top-k never picks them.

No weight surgery: the experts stay loaded, so this measures quality impact and reports
the masked share as potential savings.
"""

from __future__ import annotations

import torch
from torch import nn

from ..adapters.base import Adapter


def keep_sets_with_min(keep: list[list[int]], counts_rank: list[list[int]], top_k: int) -> list[list[int]]:
    """Top up any layer that keeps fewer than top_k experts with its next-hottest ones."""
    out = []
    for kept, ranked in zip(keep, counts_rank):
        kept = list(kept)
        for e in ranked:
            if len(kept) >= top_k:
                break
            if e not in kept:
                kept.append(e)
        out.append(sorted(kept))
    return out


class RouterMask:
    """Context manager that installs per-layer expert masks on every router."""

    def __init__(self, adapter: Adapter, routers: list[tuple[str, nn.Module]], keep: list[list[int]]):
        if len(keep) != len(routers):
            raise ValueError(f"keep sets for {len(keep)} layers, model has {len(routers)} MoE layers")
        if any(len(k) < adapter.top_k for k in keep):
            raise ValueError(f"every layer must keep at least top_k={adapter.top_k} experts")
        self.adapter = adapter
        self.routers = routers
        self.keep = keep
        self._handles: list = []

    def _hook(self, layer: int):
        blocked_cache: dict[torch.device, torch.Tensor] = {}

        def hook(module, inputs, out):
            logits, _ = self.adapter.unpack(out)
            if logits.device not in blocked_cache:
                blocked = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
                blocked[self.keep[layer]] = False
                blocked_cache[logits.device] = blocked
            masked = logits.masked_fill(blocked_cache[logits.device], float("-inf"))
            weights_dtype = out[self.adapter.weights_at].dtype
            indices, weights = self.adapter.select(masked.reshape(-1, masked.shape[-1]), weights_dtype)
            indices = indices.reshape(out[self.adapter.indices_at].shape)
            weights = weights.reshape(out[self.adapter.weights_at].shape)
            return self.adapter.repack(out, masked, indices, weights)

        return hook

    def __enter__(self) -> RouterMask:
        self._handles = [m.register_forward_hook(self._hook(i)) for i, (_, m) in enumerate(self.routers)]
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
