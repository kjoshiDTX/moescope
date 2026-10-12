"""Hooks must recover exactly the experts the MoE layer chose internally."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from moescope.adapters.base import SoftmaxTopKAdapter
from moescope.adapters.registry import UnsupportedModelError, adapter_for
from moescope.backends.hf import HFBackend, RouterRecorder
from moescope.trace.reader import load_run
from moescope.trace.schema import RunManifest, validate
from moescope.trace.writer import TraceWriter

from .tiny_models import FAMILIES, tiny_model


# ---- toy sparse MoE, in the style of the reference repo: router -> softmax -> top-k -> experts ----
class ToyTopKRouter(nn.Module):
    def __init__(self, dim, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.linear = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x):
        logits = self.linear(x)
        probs = F.softmax(logits, dim=-1)
        weights, indices = probs.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        return logits, weights, indices


class ToySparseMoE(nn.Module):
    def __init__(self, dim=16, num_experts=6, top_k=2):
        super().__init__()
        self.router = ToyTopKRouter(dim, num_experts, top_k)
        self.experts = nn.ModuleList(nn.Sequential(nn.Linear(dim, 4 * dim), nn.ReLU(), nn.Linear(4 * dim, dim)) for _ in range(num_experts))
        self.last_indices = None

    def forward(self, x):
        _, weights, indices = self.router(x)
        self.last_indices = indices
        out = torch.zeros_like(x)
        for slot in range(indices.shape[-1]):
            for e, expert in enumerate(self.experts):
                mask = indices[:, slot] == e
                if mask.any():
                    out[mask] += weights[mask, slot, None] * expert(x[mask])
        return out


class ToyAdapter(SoftmaxTopKAdapter):
    family = "toy"
    model_types = ("toy",)
    router_classes = ("ToyTopKRouter",)
    always_renormalize = True


class ToyConfig:
    model_type = "toy"
    num_experts = 6
    num_experts_per_tok = 2


def test_toy_moe_hook_recovers_internal_topk():
    torch.manual_seed(0)
    layers = nn.Sequential(ToySparseMoE(), ToySparseMoE())
    adapter = ToyAdapter(ToyConfig())
    x = torch.randn(10, 16)
    with RouterRecorder(adapter, adapter.find_routers(layers)) as rec:
        layers(x)
        captured = rec.drain()
    assert [c[0] for c in captured] == [0, 1]
    for (layer, ids, weights, _), moe in zip(captured, layers):
        assert np.array_equal(np.sort(ids, -1), np.sort(moe.last_indices.numpy(), -1))
        assert np.allclose(weights.sum(-1), 1, atol=1e-5)
        assert (np.diff(weights, axis=-1) <= 1e-7).all()  # highest weight first


def test_unsupported_model_lists_families():
    class Cfg:
        model_type = "llama"

    with pytest.raises(UnsupportedModelError, match="granite-moe"):
        adapter_for(Cfg())


@pytest.mark.parametrize("family", FAMILIES)
def test_tiny_model_trace(family, tmp_path):
    model = tiny_model(family)
    backend = HFBackend(device="cpu", max_new_tokens=5)
    info = backend.attach(model, tokenizer=None, model_id=f"tiny-{family}")
    backend.self_test()
    assert info.num_layers == (2 if family != "qwen3" else 2)  # qwen3's dense layer is skipped
    assert info.expert_bytes > 0

    ids = torch.randint(0, 128, (1, 7))
    with TraceWriter(tmp_path) as w:
        for e in backend.trace_ids(ids, prompt_id=3, category="code"):
            w.write(e)
        w.close(RunManifest("0.1.0", "hf", info.model_id, info.num_layers, info.num_experts, info.top_k,
                            info.shared_experts, info.expert_bytes, info.dtype, "x", 5, "now"))
    validate(tmp_path)
    df = load_run(tmp_path).routes
    prefill = df.filter(df["phase"] == "prefill")
    assert prefill.height == 7 * info.num_layers
    tokens = df.filter(df["layer"] == 0).height
    for layer in range(info.num_layers):  # every layer routes top_k experts for every token
        assert df.filter(df["layer"] == layer)["expert_ids"].list.len().sum() == info.top_k * tokens
    decode = df.filter(df["phase"] == "decode")
    assert decode["token_pos"].min() == 7 if decode.height else True
