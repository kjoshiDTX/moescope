"""Compare two workloads on the same model: which experts gain or lose share, per layer."""

from __future__ import annotations

import numpy as np

from ..trace.reader import Run
from .utilization import count_matrix, shares


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon distance (base 2, so 0 = identical, 1 = disjoint)."""
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return float(np.sqrt(max(0.0, (kl(p, m) + kl(q, m)) / 2)))


def compare_runs(a: Run, b: Run, n: int = 5) -> dict:
    ma, mb = a.manifest, b.manifest
    if (ma.num_layers, ma.num_experts) != (mb.num_layers, mb.num_experts) or ma.model_id != mb.model_id:
        raise ValueError(f"runs profile different models: {ma.model_id} vs {mb.model_id}")
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
        "mean_js_distance": float(np.mean([l["js_distance"] for l in layers])),
        "layers": layers,
        "share_a": sa.round(6).tolist(),
        "share_b": sb.round(6).tolist(),
    }
