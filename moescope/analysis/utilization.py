"""Per-layer expert utilization, entropy and coverage. Pure functions over Polars / numpy.

Definitions (from the spec):
  utilization[l, e] = tokens routed to e in layer l / tokens in layer l   (sums to top_k per layer)
  share[l, e]       = utilization / top_k                                 (sums to 1 per layer)
  entropy           = Shannon entropy of the share distribution, in bits
  coverage@p        = smallest number of experts whose shares add up to at least p%
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..trace.schema import RunManifest

COVERAGE_LEVELS = (90, 95, 99)


def id_matrix(routes: pl.DataFrame) -> np.ndarray:
    """expert_ids as a dense (rows, k) int array."""
    if routes.height == 0:
        return np.zeros((0, 0), dtype=np.int64)
    k = int(routes["expert_ids"].list.len()[0])
    return routes["expert_ids"].list.to_array(k).to_numpy().astype(np.int64)


def count_matrix(routes: pl.DataFrame, num_layers: int, num_experts: int, weighted: bool = False) -> np.ndarray:
    """(layers, experts) count of routed tokens, or sum of routing weights when weighted."""
    if routes.height == 0:
        return np.zeros((num_layers, num_experts))
    ids = id_matrix(routes)
    layer = routes["layer"].to_numpy().astype(np.int64)
    flat = (layer[:, None] * num_experts + ids).ravel()
    w = None
    if weighted:
        k = ids.shape[1]
        w = routes["weights"].list.to_array(k).to_numpy().astype(np.float64).ravel()
    counts = np.bincount(flat, weights=w, minlength=num_layers * num_experts)
    counts = counts.reshape(num_layers, num_experts)
    return counts if weighted else counts.astype(np.int64)


def tokens_per_layer(routes: pl.DataFrame, num_layers: int) -> np.ndarray:
    return np.bincount(routes["layer"].to_numpy().astype(np.int64), minlength=num_layers)


def shares(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=-1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros(counts.shape, dtype=np.float64), where=totals > 0)


def entropy_bits(share: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(share > 0, -share * np.log2(share), 0.0)
    return terms.sum(axis=-1)


def coverage(counts: np.ndarray, p: float) -> np.ndarray:
    """Experts needed per layer to reach p% of routed tokens. Exact on integer counts."""
    ordered = -np.sort(-counts, axis=-1)
    cum = np.cumsum(ordered, axis=-1)
    target = counts.sum(axis=-1, keepdims=True) * (p / 100)
    reached = cum >= target - 1e-9 * np.maximum(target, 1)
    return reached.argmax(axis=-1) + 1


def coverage_sets(counts: np.ndarray, p: float, always_keep: list[int] | None = None) -> list[list[int]]:
    """For each layer, the expert ids that make up coverage@p (plus any always-kept experts)."""
    needed = coverage(counts, p)
    keep = []
    for row, n in zip(counts, needed):
        chosen = set(np.argsort(-row, kind="stable")[:n].tolist()) | set(always_keep or [])
        keep.append(sorted(chosen))
    return keep


def coverage_curve(counts: np.ndarray) -> np.ndarray:
    """(layers, experts) cumulative share when experts are taken hottest-first."""
    return np.cumsum(-np.sort(-shares(counts), axis=-1), axis=-1)


def layer_stats(routes: pl.DataFrame, manifest: RunManifest) -> pl.DataFrame:
    L, E = manifest.num_layers, manifest.num_experts
    counts = count_matrix(routes, L, E)
    h = entropy_bits(shares(counts))
    cols = {
        "layer": np.arange(L),
        "tokens": tokens_per_layer(routes, L),
        "active_experts": (counts > 0).sum(axis=1),
        "entropy_bits": h,
        "entropy_norm": h / np.log2(E) if E > 1 else np.zeros(L),
    }
    for p in COVERAGE_LEVELS:
        cols[f"coverage_{p}"] = coverage(counts, p)
    return pl.DataFrame(cols)


def hot_cold(counts: np.ndarray, n: int = 10) -> list[dict]:
    """The n hottest and coldest experts per layer, with their share of routed tokens."""
    share = shares(counts)
    out = []
    for layer, row in enumerate(share):
        order = np.argsort(-row, kind="stable")
        out.append({
            "layer": layer,
            "hottest": [{"expert": int(e), "share": float(row[e])} for e in order[:n]],
            "coldest": [{"expert": int(e), "share": float(row[e])} for e in order[::-1][:n]],
        })
    return out
