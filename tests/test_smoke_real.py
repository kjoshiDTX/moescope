"""Manual real-model smoke test: `pytest -m slow tests/test_smoke_real.py` (downloads ~2.7 GB once)."""

import pytest

from moescope.trace.reader import load_run
from moescope.trace.schema import RunManifest, validate
from moescope.trace.writer import TraceWriter

MODEL = "ibm-granite/granite-3.1-1b-a400m-instruct"


@pytest.mark.slow
def test_profile_ten_prompts(tmp_path):
    from moescope.backends.hf import HFBackend
    from moescope.prompts import load_prompts

    backend = HFBackend(max_new_tokens=8)
    info = backend.load(MODEL)
    backend.self_test()
    with TraceWriter(tmp_path) as w:
        for e in backend.run(load_prompts("examples/code.jsonl")[:10]):
            w.write(e)
        w.close(RunManifest("0.1.0", "hf", MODEL, info.num_layers, info.num_experts, info.top_k, info.shared_experts,
                            info.expert_bytes, info.dtype, "examples/code.jsonl", 8, "now"))
    validate(tmp_path)
    routes = load_run(tmp_path).routes
    assert routes.height > 0
    tokens = routes.filter(routes["layer"] == 0).height
    for layer in range(info.num_layers):
        assert routes.filter(routes["layer"] == layer)["expert_ids"].list.len().sum() == info.top_k * tokens
