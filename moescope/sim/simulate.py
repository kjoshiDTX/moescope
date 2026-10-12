"""Replay a trace through cache policies at several memory budgets."""

from __future__ import annotations

import numpy as np
import polars as pl

from ..analysis.utilization import id_matrix
from ..trace.reader import Run
from .policies import POLICIES, PinnedCache, count_hits

GIB = 2**30
ALL_POLICIES = ("lru", "lfu", "pinned")


def lookup_stream(run: Run) -> tuple[np.ndarray, int]:
    """Expert lookups in execution order, as keys layer * num_experts + expert.

    Order: prompt, then token position, then layer, then the token's experts highest weight first.
    Shared experts are always resident, so they are not lookups. Returns (keys, token count).
    """
    E = run.manifest.num_experts
    routes = run.routes.sort(["prompt_id", "token_pos", "layer"], maintain_order=True)
    ids = id_matrix(routes)
    if ids.size == 0:
        return np.zeros(0, dtype=np.int64), 0
    keys = routes["layer"].to_numpy().astype(np.int64)[:, None] * E + ids
    if run.manifest.shared_experts:
        keys = keys[~np.isin(ids, run.manifest.shared_experts)]
    tokens = routes.filter(pl.col("layer") == routes["layer"].min()).height
    return keys.ravel(), tokens


def slots_for_budget(budget_gb: float, expert_bytes: int) -> int:
    """Budget is the memory set aside for routed experts, in GiB."""
    return int(budget_gb * GIB // expert_bytes)


def simulate_run(run: Run, budgets_gb: list[float], policies: list[str] = list(ALL_POLICIES)) -> list[dict]:
    unknown = set(policies) - set(ALL_POLICIES)
    if unknown:
        raise ValueError(f"unknown policies {sorted(unknown)}; choose from {ALL_POLICIES}")
    keys, tokens = lookup_stream(run)
    key_list = keys.tolist()
    total_experts = run.manifest.num_layers * (run.manifest.num_experts - len(run.manifest.shared_experts))
    expert_bytes = run.manifest.expert_bytes
    ranked = np.argsort(-np.bincount(keys), kind="stable") if keys.size else np.zeros(0, dtype=np.int64)

    results = []
    for budget in budgets_gb:
        slots = slots_for_budget(budget, expert_bytes)
        for policy in policies:
            if policy == "pinned":
                hits = int(np.isin(keys, ranked[:slots]).sum())
            else:
                hits = count_hits(POLICIES[policy](slots), key_list)
            misses = len(key_list) - hits
            results.append({
                "policy": policy,
                "budget_gb": budget,
                "slots": slots,
                "resident_share": min(1.0, slots / total_experts) if total_experts else 0.0,
                "lookups": len(key_list),
                "hits": hits,
                "hit_rate": hits / len(key_list) if key_list else 0.0,
                "bytes_per_token": misses * expert_bytes / tokens if tokens else 0.0,
            })
    return results


def pin_list(run: Run, slots: int) -> list[dict]:
    """The `slots` most-used (layer, expert) pairs, for a pinned-cache configuration."""
    keys, _ = lookup_stream(run)
    if not keys.size:
        return []
    counts = np.bincount(keys)
    E = run.manifest.num_experts
    order = np.argsort(-counts, kind="stable")[:slots]
    return [{"layer": int(k // E), "expert": int(k % E), "lookups": int(counts[k])} for k in order if counts[k] > 0]


__all__ = ["lookup_stream", "simulate_run", "pin_list", "slots_for_budget", "PinnedCache"]
