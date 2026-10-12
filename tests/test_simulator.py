"""Hand-worked cache traces. Lookup sequence A B A C B A D A B C with a 2-slot cache.

LRU (evict least recently used):
  A miss [A] · B miss [A B] · A HIT [B A] · C miss evict B [A C] · B miss evict A [C B]
  A miss evict C [B A] · D miss evict B [A D] · A HIT [D A] · B miss evict D [A B] · C miss evict A
  -> 2 hits
LFU (evict lowest use-count since admission, ties -> least recently used):
  A miss {A1} · B miss {A1 B1} · A HIT {A2 B1} · C miss evict B {A2 C1} · B miss evict C {A2 B1}
  A HIT {A3 B1} · D miss evict B {A3 D1} · A HIT {A4 D1} · B miss evict D {A4 B1} · C miss evict B
  -> 3 hits
Pinned (top-2 by frequency over the trace: A=4, B=3, preloaded, never evicted)
  -> A x4 + B x3 = 7 hits
"""

import numpy as np
import pytest

from moescope.sim.policies import LFUCache, LRUCache, PinnedCache, count_hits
from moescope.sim.simulate import lookup_stream, simulate_run

from .fixtures.golden import write_golden

A, B, C, D = 0, 1, 2, 3
TRACE = [A, B, A, C, B, A, D, A, B, C]


def test_lru_hand_trace():
    assert count_hits(LRUCache(2), TRACE) == 2


def test_lfu_hand_trace():
    assert count_hits(LFUCache(2), TRACE) == 3


def test_pinned_hand_trace():
    assert count_hits(PinnedCache.from_trace(2, TRACE), TRACE) == 7


@pytest.mark.parametrize("cls", [LRUCache, LFUCache])
def test_zero_and_huge_capacity(cls):
    assert count_hits(cls(0), TRACE) == 0
    assert count_hits(cls(100), TRACE) == len(TRACE) - len(set(TRACE))  # only compulsory misses


def test_lookup_stream_order_and_shared_exclusion(tmp_path):
    # 2 layers, top-2, expert 3 is shared and must never appear as a lookup.
    plan = {0: [((0, 3), 2)], 1: [((1, 3), 2)]}
    run_dir = write_golden(tmp_path / "s", plan, num_experts=4, top_k=2, shared=[3])
    from moescope.trace.reader import load_run

    keys, tokens = lookup_stream(load_run(run_dir))
    # token order: (tok0, L0), (tok0, L1), (tok1, L0), (tok1, L1); key = layer * E + expert
    assert keys.tolist() == [0, 5, 0, 5]
    assert tokens == 2


def test_simulate_run_budget_conversion(tmp_path):
    plan = {0: [((0,), 5), ((1,), 5)]}
    run_dir = write_golden(tmp_path / "b", plan, num_experts=4, top_k=1, expert_bytes=2**30 // 4)
    from moescope.trace.reader import load_run

    results = simulate_run(load_run(run_dir), budgets_gb=[0.25, 1.0], policies=["lru", "pinned"])
    by = {(r["policy"], r["budget_gb"]): r for r in results}
    assert by[("lru", 0.25)]["slots"] == 1
    assert by[("lru", 1.0)]["slots"] == 4
    assert by[("pinned", 0.25)]["hits"] == 5  # expert 0 pinned
    assert by[("lru", 1.0)]["hits"] == 8  # 10 lookups, 2 compulsory misses
    assert by[("lru", 1.0)]["bytes_per_token"] == pytest.approx(2 * (2**30 // 4) / 10)
