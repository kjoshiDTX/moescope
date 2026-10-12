"""How often two experts are chosen for the same token in the same layer."""

from __future__ import annotations

import numpy as np
import polars as pl

from .utilization import id_matrix


def coactivation(routes: pl.DataFrame, layer: int, num_experts: int) -> np.ndarray:
    """(E, E) token counts where both experts are in the top-k. The diagonal is each expert's count."""
    ids = id_matrix(routes.filter(pl.col("layer") == layer))
    if ids.size == 0:
        return np.zeros((num_experts, num_experts), dtype=np.int64)
    onehot = np.zeros((len(ids), num_experts), dtype=np.float32)
    np.put_along_axis(onehot, ids, 1.0, axis=1)
    return (onehot.T @ onehot).round().astype(np.int64)


def lift(co: np.ndarray, tokens: int) -> np.ndarray:
    """Observed pair rate / rate expected if experts were picked independently. 1 means no affinity."""
    if tokens == 0:
        return np.zeros(co.shape)
    p = np.diag(co) / tokens
    expected = np.outer(p, p) * tokens
    out = np.divide(co, expected, out=np.zeros(co.shape), where=expected > 0)
    np.fill_diagonal(out, 0)
    return out
