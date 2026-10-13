import glob
import os
import shutil

import numpy as np
import pytest

from moescope.analysis.compare import top1_agreement
from moescope.backends.llamacpp import LlamaCppBackend, LlamaCppError, read_records
from moescope.trace.reader import load_run

from .fixtures.golden import write_golden


def _record(prompt_id, phase, pos, layer, ids, weights, lmax):
    ids = np.asarray(ids, np.int32)
    n, k = ids.shape
    head = np.array([prompt_id, phase, pos, layer, n, k], np.int32).tobytes()
    return head + ids.tobytes() + np.asarray(weights, np.float32).tobytes() + np.asarray(lmax, np.float32).tobytes()


def test_read_records_incremental(tmp_path):
    a = _record(3, 0, 0, 5, [[1, 2], [3, 4]], [[0.6, 0.4], [0.7, 0.3]], [1.0, 2.0])
    b = _record(3, 1, 2, 5, [[7, 0]], [[0.9, 0.1]], [3.0])
    path = tmp_path / "t.bin"
    path.write_bytes(a + b[:10])  # second record only partly written
    recs, offset = read_records(path)
    assert len(recs) == 1 and offset == len(a)
    assert recs[0][:4] == (3, 0, 0, 5)
    assert recs[0][4].tolist() == [[1, 2], [3, 4]]
    path.write_bytes(a + b)
    recs, offset = read_records(path, offset)
    assert len(recs) == 1 and recs[0][1] == 1 and recs[0][4].tolist() == [[7, 0]]
    assert offset == len(a + b)


def test_load_rejects_non_gguf(tmp_path):
    with pytest.raises(LlamaCppError, match="gguf"):
        LlamaCppBackend().load(str(tmp_path / "model.safetensors"))


def test_top1_agreement(tmp_path):
    a = load_run(write_golden(tmp_path / "a", {0: [((0, 1), 8), ((2, 3), 2)]}, num_experts=4, top_k=2))
    b = load_run(write_golden(tmp_path / "b", {0: [((0, 2), 8), ((3, 2), 2)]}, num_experts=4, top_k=2))
    agree = top1_agreement(a, b)["prefill"]
    assert agree["rows"] == 10
    assert agree["top1"] == pytest.approx(0.8)          # first 8 tokens share expert 0 at rank 1
    assert agree["topk_overlap"] == pytest.approx((8 * 0.5 + 2 * 1.0) / 10)


GGUF = os.environ.get("MOESCOPE_TEST_GGUF") or next(iter(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--bartowski--granite-3.1-1b-a400m-instruct-GGUF/snapshots/*/*.gguf"))), None)


@pytest.mark.slow
@pytest.mark.skipif(not GGUF or not shutil.which("llama-server"), reason="needs llama.cpp and a Granite MoE GGUF")
def test_llamacpp_trace_real_model():
    from moescope.prompts import Prompt

    backend = LlamaCppBackend(max_new_tokens=4)
    info = backend.load(GGUF)
    assert info.num_experts == 32 and info.top_k == 8 and info.expert_bytes > 0
    events = list(backend.run([Prompt(0, "def add(a, b):", "code")]))
    prefill = [e for e in events if e.phase == "prefill"]
    n = len(prefill[0])
    last = max(e.layer for e in prefill)
    assert all(len(e) == n for e in prefill if e.layer != last)
    assert [len(e) for e in prefill if e.layer == last] == [1]  # llama.cpp skips non-output tokens in the last layer
    assert [e.token_pos[0] for e in prefill if e.layer == last] == [n - 1]
    assert all(np.allclose(e.weights.sum(1), 1, atol=1e-4) for e in events)
