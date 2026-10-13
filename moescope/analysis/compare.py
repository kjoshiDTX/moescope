"""Compare two workloads on the same model: which experts gain or lose share, per layer."""

from __future__ import annotations

import numpy as np
import polars as pl

from ..trace.reader import Run
from .utilization import count_matrix, id_matrix, shares


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon distance (base 2, so 0 = identical, 1 = disjoint)."""
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return float(np.sqrt(max(0.0, (kl(p, m) + kl(q, m)) / 2)))


def top1_agreement(a: Run, b: Run) -> dict | None:
    """For tokens both runs routed at the same (prompt, phase, position, layer): how often the
    top-1 expert matches, and the mean top-k overlap. Meaningful when both saw identical tokens,
    e.g. an HF run and a llama.cpp run of the same model on the same prompts."""
    keys = ["prompt_id", "phase", "token_pos", "layer"]
    pick = lambda r: r.routes.select(*keys, pl.col("expert_ids"))  # noqa: E731
    joined = pick(a).join(pick(b), on=keys, suffix="_b")
    if joined.height == 0:
        return None
    out = {}
    for phase in ("prefill", "decode", "all"):
        rows = joined if phase == "all" else joined.filter(pl.col("phase") == phase)
        if rows.height == 0:
            continue
        ia, ib = id_matrix(rows), id_matrix(rows.select(pl.col("expert_ids_b").alias("expert_ids")))
        overlap = (ia[:, :, None] == ib[:, None, :]).any(-1).mean(-1)
        out[phase] = {"rows": rows.height, "top1": float((ia[:, 0] == ib[:, 0]).mean()), "topk_overlap": float(overlap.mean())}
    return out


def compare_runs(a: Run, b: Run, n: int = 5) -> dict:
    ma, mb = a.manifest, b.manifest
    if (ma.num_layers, ma.num_experts) != (mb.num_layers, mb.num_experts):
        raise ValueError(f"runs have different shapes: {ma.num_layers}x{ma.num_experts} vs {mb.num_layers}x{mb.num_experts} "
                         f"({ma.model_id} vs {mb.model_id})")
    sa = shares(count_matrix(a.routes, ma.num_layers, ma.num_experts))
    sb = shares(count_matrix(b.routes, mb.num_layers, mb.num_experts))
    layers = []
    for layer in range(ma.num_layers):
        delta = sb[layer] - sa[layer]
        order = np.argsort(-delta, kind="stable")
        layers.append({
            "layer": layer,
            "js_distance": js_distance(sa[layer], sb[layer]) if sa[layer].sum() and sb[layer].sum() else 0.0,
            "gainers": [{"expert": int(e), "delta": float(delta[e])} for e in order[:n] if delta[e] > 0],
            "losers": [{"expert": int(e), "delta": float(delta[e])} for e in order[::-1][:n] if delta[e] < 0],
        })
    return {
        "a": a.name,
        "b": b.name,
        "same_model": ma.model_id == mb.model_id,
        "agreement": top1_agreement(a, b),
        "mean_js_distance": float(np.mean([l["js_distance"] for l in layers])),
        "layers": layers,
        "share_a": sa.round(6).tolist(),
        "share_b": sb.round(6).tolist(),
    }
