"""Phase 1 check: peak extra memory from tracing. Run as two processes and compare.

    python spike/measure_memory.py off   # same prefill + greedy decode, no hooks, no writer
    python spike/measure_memory.py on    # moescope hooks + Parquet writer
"""

import resource
import sys
import tempfile

import torch

from moescope.backends.hf import HFBackend
from moescope.prompts import load_prompts
from moescope.trace.writer import TraceWriter

MODEL = "ibm-granite/granite-3.1-1b-a400m-instruct"
mode = sys.argv[1]
prompts = load_prompts("examples/code.jsonl")[:20]
backend = HFBackend(max_new_tokens=128)
backend.load(MODEL)
peak_mps = 0


def track():
    global peak_mps
    if backend.device == "mps":
        peak_mps = max(peak_mps, torch.mps.driver_allocated_memory())


rows = 0
if mode == "on":
    with tempfile.TemporaryDirectory() as d, TraceWriter(d) as w:
        for p in prompts:
            for e in backend.run([p]):
                w.write(e)
                rows += len(e)
                track()
else:
    with torch.no_grad():
        for p in prompts:
            ids = backend.encode(p.prompt)
            out = backend.model(ids, use_cache=True, logits_to_keep=1)
            past, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
            for _ in range(backend.max_new_tokens):
                if int(nxt) in backend.eos_ids():
                    break
                out = backend.model(nxt, past_key_values=past, use_cache=True)
                past, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
                track()
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # bytes on macOS
print(f"{mode}: peak RSS {rss:.0f} MiB, peak MPS driver {peak_mps / 2**20:.0f} MiB, rows {rows}")
