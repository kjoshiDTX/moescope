import json
from types import SimpleNamespace

import pytest
import torch

from moescope.train import CollapseDetector, MoEWatchCallback

from .tiny_models import tiny_model


def test_collapse_detector_needs_a_sustained_streak():
    d = CollapseDetector(factor=3.0, steps=200)
    assert d.update(100, [4.0, 1.0]) == []
    assert d.update(250, [4.0, 1.0]) == []
    assert d.update(300, [4.0, 1.0]) == [0]          # above 3x since step 100
    assert d.new_warnings([0]) == [0]
    assert d.new_warnings([0]) == []                  # warn once per streak
    assert d.update(310, [2.0, 1.0]) == []            # streak broken
    assert d.update(320, [5.0, 1.0]) == []            # new streak starts over


def _train(model, cb, steps, lr=0.0):
    args, control = SimpleNamespace(), SimpleNamespace()
    state = SimpleNamespace(global_step=0)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    cb.on_train_begin(args, state, control, model=model)
    model.train()
    for _ in range(steps):
        cb.on_step_begin(args, state, control)
        ids = torch.randint(0, 128, (2, 16))
        out = model(ids, labels=ids, output_router_logits=True)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
        state.global_step += 1
        cb.on_step_end(args, state, control)
    cb.on_train_end(args, state, control)


@pytest.mark.parametrize("family", ["granite", "qwen3"])
def test_callback_logs_every_n_steps(family, tmp_path):
    model = tiny_model(family)
    cb = MoEWatchCallback(tmp_path, log_every=2)
    _train(model, cb, steps=6, lr=1e-3)
    meta = json.loads((tmp_path / "train_meta.json").read_text())
    lines = [json.loads(l) for l in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert [l["step"] for l in lines] == [2, 4, 6]
    rec = lines[-1]
    assert len(rec["load"]) == meta["num_layers"] and len(rec["load"][0]) == meta["num_experts"]
    assert all(abs(sum(row) - 1) < 1e-3 for row in rec["load"])
    assert rec["aux_loss"] >= 0.99  # E * sum(f * P) is minimized at 1.0 by a balanced router
    assert rec["z_loss"] > 0 and all(r >= 1 for r in rec["max_load_ratio"])
    assert not cb._handles  # hooks removed at train end


def test_callback_flags_a_collapsed_router(tmp_path, caplog):
    model = tiny_model("granite")
    # Force layer 0's router to send everything to expert 0.
    router = [m for m in model.modules() if type(m).__name__ == "GraniteMoeTopKRouter"][0]
    with torch.no_grad():
        router.weight.zero_()
        router.weight[0] = 5.0
    cb = MoEWatchCallback(tmp_path, log_every=1, collapse_factor=3.0, collapse_steps=3)
    with caplog.at_level("WARNING", logger="moescope"):
        _train(model, cb, steps=5)
    lines = [json.loads(l) for l in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert lines[-1]["collapsed_layers"] == [0]
    assert lines[0]["collapsed_layers"] == []
    assert "router collapse in MoE layer 0" in caplog.text
