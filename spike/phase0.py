"""Phase 0 spike: find the gate module, hook it, compare with output_router_logits=True."""

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "ibm-granite/granite-3.1-1b-a400m-instruct"
device = "mps" if torch.backends.mps.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device).eval()

routers = [(n, m) for n, m in model.named_modules() if type(m).__name__.endswith("TopKRouter")]
print(f"model class: {type(model).__name__}, router modules: {len(routers)}")
for n, m in routers[:2]:
    print(f"  {n}: {type(m).__name__} weight={tuple(m.weight.shape)} top_k={m.top_k}")

captured = {}


def make_hook(name):
    def hook(mod, inp, out):
        captured[name] = [o.detach() for o in out]
    return hook


handles = [m.register_forward_hook(make_hook(n)) for n, m in routers]
ids = tok("def fibonacci(n):", return_tensors="pt").input_ids[:, :5].to(device)
with torch.no_grad():
    out = model(ids, output_router_logits=True)
for h in handles:
    h.remove()

first = captured[routers[0][0]]
print("hook output shapes:", [tuple(t.shape) for t in first], [t.dtype for t in first])
idx, w, logits = first
print("layer 0 top-k ids for 5 tokens:\n", idx.cpu())
print("layer 0 weights row 0:", w[0].float().cpu().numpy().round(3), "sum", float(w[0].float().sum()))

mismatch = 0
for layer, (n, _) in enumerate(routers):
    ref_logits = out.router_logits[layer]
    ref_idx = ref_logits.float().topk(routers[layer][1].top_k, dim=-1).indices
    hook_idx = captured[n][0]
    same = torch.equal(ref_idx.sort(-1).values, hook_idx.sort(-1).values)
    mismatch += not same
print(f"layers where hook top-k != topk(output_router_logits): {mismatch}/{len(routers)}")
