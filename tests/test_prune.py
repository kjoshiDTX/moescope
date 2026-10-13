import pytest
import torch

from moescope.backends.hf import HFBackend, RouterRecorder
from moescope.prune.evaluate import exact_match, keep_sets, load_metric
from moescope.prune.mask import RouterMask, keep_sets_with_min
from moescope.trace.reader import load_run

from .fixtures.golden import write_golden
from .tiny_models import FAMILIES, tiny_model


def _backend(family):
    backend = HFBackend(device="cpu")
    backend.attach(tiny_model(family), tokenizer=None, model_id=family)
    return backend


@pytest.mark.parametrize("family", FAMILIES)
def test_masked_experts_are_never_selected(family):
    b = _backend(family)
    E, k = b.info.num_experts, b.info.top_k
    keep = [[0, 2, 5] for _ in range(b.info.num_layers)]
    ids = torch.randint(0, 128, (1, 24))
    with torch.no_grad(), RouterMask(b.adapter, b.routers, keep), RouterRecorder(b.adapter, b.routers) as rec:
        b.model(ids)
        captured = rec.drain()
    assert len(captured) == b.info.num_layers
    for _, chosen, weights, _ in captured:
        assert set(chosen.ravel().tolist()) <= {0, 2, 5}
        assert abs(weights.sum(-1) - 1).max() < 1e-3


@pytest.mark.parametrize("family", FAMILIES)
def test_keeping_every_expert_changes_nothing(family):
    b = _backend(family)
    ids = torch.randint(0, 128, (1, 16))
    keep = [list(range(b.info.num_experts))] * b.info.num_layers
    with torch.no_grad():
        ref = b.model(ids).logits
        with RouterMask(b.adapter, b.routers, keep):
            masked = b.model(ids).logits
    torch.testing.assert_close(masked, ref)


def test_mask_rejects_too_small_keep_sets():
    b = _backend("granite")
    with pytest.raises(ValueError, match="top_k"):
        RouterMask(b.adapter, b.routers, [[0]] * b.info.num_layers)


def test_keep_sets_follow_coverage_and_respect_top_k(tmp_path):
    # Layer 0: 90% of tokens on expert 0 -> Cov@90 is {0}, topped up to top_k=2 with the next hottest.
    plan = {0: [((0,), 90), ((1,), 6), ((2,), 4)]}
    run = load_run(write_golden(tmp_path / "r", plan, num_experts=4, top_k=1))
    assert keep_sets(run, 90, top_k=1) == [[0]]
    assert keep_sets(run, 90, top_k=2) == [[0, 1]]
    assert keep_sets(run, 100, top_k=1) == [[0, 1, 2]]
    assert keep_sets_with_min([[3]], [[3, 1, 0, 2]], 3) == [[0, 1, 3]]


def test_metrics():
    assert exact_match("56.", "56") == 1.0
    assert exact_match("`ABC`\nmore text", "ABC") == 1.0
    assert exact_match("The answer is 56", "56") == 0.0
    assert load_metric("contains")("The answer is 56", "56") == 1.0
    assert load_metric("os.path:join") is not None
    with pytest.raises(ValueError):
        load_metric("nope")
