import math

import numpy as np
import pytest

from moescope.analysis.coactivation import coactivation
from moescope.analysis.compare import compare_runs, js_distance
from moescope.analysis.utilization import count_matrix, coverage, entropy_bits, layer_stats, shares
from moescope.trace.reader import load_run

from .fixtures.golden import planted_plan, write_golden


@pytest.fixture
def planted(tmp_path):
    return load_run(write_golden(tmp_path / "planted", planted_plan(), num_experts=16, top_k=1))


def test_utilization_recovers_planted(planted):
    counts = count_matrix(planted.routes, 2, 16)
    assert counts[0, :3].tolist() == [300, 300, 300]
    assert counts[0, 3:13].tolist() == [10] * 10
    assert counts[0, 13:].sum() == 0
    util = counts / 1000
    assert util[0, 0] == pytest.approx(0.3, abs=1e-12)
    assert util[1].sum() == pytest.approx(1.0)


def test_coverage_recovers_planted(planted):
    counts = count_matrix(planted.routes, 2, 16)
    assert coverage(counts, 90).tolist() == [3, 15]  # layer 1: ceil(0.9 * 16) of a near-uniform layer
    assert coverage(counts, 95).tolist()[0] == 8
    assert coverage(counts, 99).tolist()[0] == 12
    assert coverage(counts, 100).tolist()[0] == 13


def test_entropy_recovers_planted(planted):
    counts = count_matrix(planted.routes, 2, 16)
    expected = -(3 * 0.3 * math.log2(0.3) + 10 * 0.01 * math.log2(0.01))
    h = entropy_bits(shares(counts))
    assert h[0] == pytest.approx(expected, abs=1e-12)
    assert h[1] == pytest.approx(4.0, abs=1e-3)  # 1000 tokens over 16 experts is almost uniform


def test_layer_stats_table(planted):
    stats = layer_stats(planted.routes, planted.manifest)
    row0 = stats.filter(stats["layer"] == 0).row(0, named=True)
    assert row0["tokens"] == 1000
    assert row0["coverage_90"] == 3
    assert row0["active_experts"] == 13


def test_topk2_utilization_sums_to_k_and_coactivation(tmp_path):
    plan = {0: [((0, 1), 50), ((0, 2), 30), ((1, 2), 20)]}
    run = load_run(write_golden(tmp_path / "k2", plan, num_experts=4, top_k=2))
    counts = count_matrix(run.routes, 1, 4)
    assert counts[0].tolist() == [80, 70, 50, 0]
    assert (counts[0] / 100).sum() == pytest.approx(2.0)  # utilization sums to top_k
    co = coactivation(run.routes, layer=0, num_experts=4)
    assert co[0, 1] == co[1, 0] == 50
    assert co[0, 2] == 30 and co[1, 2] == 20
    assert np.diag(co).tolist() == [80, 70, 50, 0]
    assert co[3].sum() == 0


def test_js_distance_bounds():
    p = np.array([0.5, 0.5, 0, 0])
    assert js_distance(p, p) == pytest.approx(0)
    assert js_distance(p, np.array([0, 0, 0.5, 0.5])) == pytest.approx(1)


def test_compare_runs(tmp_path):
    a = load_run(write_golden(tmp_path / "a", {0: [((0,), 90), ((1,), 10)]}, num_experts=4, top_k=1, category="code"))
    b = load_run(write_golden(tmp_path / "b", {0: [((2,), 90), ((1,), 10)]}, num_experts=4, top_k=1, category="chat"))
    result = compare_runs(a, b)
    layer = result["layers"][0]
    assert layer["js_distance"] > 0.8
    assert layer["gainers"][0]["expert"] == 2
    assert layer["losers"][0]["expert"] == 0
