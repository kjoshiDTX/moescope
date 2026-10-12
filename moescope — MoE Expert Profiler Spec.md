# moescope — MoE Expert Profiler Spec

Oct 12, 2026 · @krish

## Overview

moescope is a local CLI plus dashboard that shows developers how their own workload uses the experts inside an open Mixture-of-Experts model, and turns that into memory and pruning decisions.

**Who it is for.** Developers who run open MoE models (Qwen3-MoE, gpt-oss, Granite MoE, OLMoE, Mixtral, DeepSeek-family) locally or on a small GPU box, and people fine-tuning MoE models.

**The problem.** Inference engines now cache and offload experts automatically, but nobody tells the developer which experts *their* prompts actually route to. Without that, cache size, quantization, and pruning choices are guesses.

**What it does, in one line each:**

- Records router decisions (top-k expert IDs and weights, per layer, per token) while the model runs on the user's prompts.
- Shows per-layer expert utilization, skew, and co-activation for that workload.
- Simulates expert-cache hit rates for a given memory budget and recommends engine settings.
- Prunes experts the workload never uses, then re-runs an eval to report the quality and memory tradeoff.
- During MoE fine-tuning, tracks aux loss, z-loss, per-expert load, and capacity drops, and flags router collapse.

**Non-goals (v1).** Not a new inference engine. Not a training framework. Not app-level LLM routing across providers. No cloud service; everything runs on the user's machine.

**Concept source.** [junfanz1/MoE-Mixture-of-Experts-in-PyTorch](https://github.com/junfanz1/MoE-Mixture-of-Experts-in-PyTorch) is used only as a reference for router, top-k, capacity, and load-balancing mechanics. It is an educational repo on synthetic data; no code is taken from it.

## Core concepts mapped to features

Every feature reads one MoE mechanism and turns it into a number a developer can act on.

| MoE concept | What it is | moescope feature that uses it |
| --- | --- | --- |
| Router / gate | Linear layer producing one logit per expert, per token | Router probes record the logits each forward pass |
| Top-k selection | Each token uses only the k highest-scoring experts | Utilization heatmap, co-activation matrix |
| Routing weights | Softmax over the chosen k, used to mix expert outputs | Weighted utilization (an expert picked at weight 0.05 matters less) |
| Capacity factor | Max tokens an expert accepts per batch; extras are dropped | Drop-rate panel in training diagnostics |
| Aux load-balancing loss | Penalizes uneven token share across experts | Training panel: aux loss curve, per-expert load |
| Router z-loss | Penalizes large router logits | Training panel: z-loss curve, logit magnitude histogram |
| Shared experts | Experts every token always uses (DeepSeek-style) | Excluded from pruning; always counted as resident in cache sims |
| Expert cache / offload | Hot experts in VRAM, cold ones in RAM or disk | Cache simulator and engine-flag recommendations |

Key derived metrics (definitions the code must follow):

- **Utilization** of expert e in layer l = tokens routed to e ÷ total tokens, over the profiled run.
- **Routing entropy** per layer = entropy of the utilization distribution. Low entropy means a few experts dominate.
- **Coverage@p** = smallest set of experts per layer that receives p% of routed tokens (report p = 90, 95, 99).
- **Co-activation** = how often experts i and j are both in a token's top-k, per layer.
- **Cache hit rate** = share of expert lookups already resident, under a given policy and budget.

## Architecture

moescope has three layers: backends that capture router traces from a running model, an analysis core that works only on saved traces, and two front-ends (CLI and a local dashboard) that read the analysis output.

The rule that keeps it simple: **capture once, analyze many times.** Every analysis, simulation, and chart reads Parquet traces from disk, never a live model. Only `profile` and `prune --eval` load a model.

&#91;embedded content: moescope architecture · capture, store, analyze, view\]

Only the backends and the pruner's eval step touch a model; everything right of the runs folder works on saved files.

**Components**

1. **Backends** (`moescope/backends/`) share one interface: `load(model_id) -> ModelInfo`, `run(prompts) -> Iterator[RouterEvent]`.
   - `hf`: Hugging Face transformers. Registers a forward hook on each MoE layer's gate module. v1 target.
   - `llamacpp`: GGUF models through llama.cpp's eval callback, reading the per-layer top-k tensors. v2 target (Phase 4).
2. **Model adapters** (`moescope/adapters/`): one small file per architecture family that says where the gate module lives, the expert count, top-k, and which experts are shared.
3. **Trace store** (`moescope/trace/`): writes RouterEvents to Parquet in batches, plus a `run.json` manifest.
4. **Analysis core** (`moescope/analysis/`): utilization, entropy, coverage, co-activation, per-prompt-category breakdowns. Pure functions over Polars DataFrames.
5. **Cache simulator** (`moescope/sim/`): replays the trace through LRU, LFU, and static-pinned policies at several budgets.
6. **Pruner** (`moescope/prune/`): masks unused experts in the router and re-runs a user eval.
7. **Training callback** (`moescope/train/`): a Hugging Face `TrainerCallback` that logs MoE losses and loads during fine-tuning.
8. **CLI** (`moescope/cli.py`, Typer) and **dashboard** (FastAPI serving JSON + a React/Vite/Tailwind app with D3 charts, launched by `moescope ui`).

**Tech stack:** Python 3.11+, PyTorch, transformers, Polars, PyArrow, Typer, Rich, FastAPI, Uvicorn; React, Vite, Tailwind, D3; pytest. Package with `uv`; ship the built dashboard inside the Python wheel.

**Repo layout**

```
moescope/
  backends/      hf.py, llamacpp.py, base.py
  adapters/      qwen_moe.py, granite_moe.py, olmoe.py, mixtral.py, registry.py
  trace/         schema.py, writer.py, reader.py
  analysis/      utilization.py, coactivation.py, report.py
  sim/           policies.py, simulate.py, recommend.py
  prune/         mask.py, evaluate.py
  train/         callback.py
  server/        api.py
  cli.py
web/             React app (Vite)
tests/           unit + golden-trace tests
examples/        sample prompt sets: code, chat, rag
```

## Data model

A profiling run is one directory: a `run.json` manifest plus a `routes.parquet` file with one row per (token, layer). Every backend must emit exactly this schema, so analysis never knows which engine produced it.

**`routes.parquet`** (one row per token per MoE layer)

| Column | Type | Meaning |
| --- | --- | --- |
| `prompt_id` | int32 | Index into the prompt file |
| `category` | string | User-supplied tag, such as `code`, `chat`, `rag` |
| `phase` | string (enum) | `prefill` or `decode` |
| `token_pos` | int32 | Position in the sequence |
| `layer` | int16 | MoE layer index (dense layers are skipped) |
| `expert_ids` | list\[int16\], length k | Chosen experts, highest weight first |
| `weights` | list\[float16\], length k | Routing weights after softmax over the top-k |
| `logit_max` | float16 | Largest router logit for this token (for z-loss style checks) |

**`run.json`** (manifest)

```json
{
  "moescope_version": "0.1.0",
  "backend": "hf",
  "model_id": "ibm-granite/granite-3.1-3b-a800m-instruct",
  "num_layers": 32,
  "num_experts": 40,
  "top_k": 8,
  "shared_experts": [],
  "expert_bytes": 12582912,
  "dtype": "bfloat16",
  "prompts_file": "examples/code.jsonl",
  "max_new_tokens": 128,
  "created_at": "2026-10-12T18:00:00-05:00"
}
```

The model values above are placeholders; the adapter fills them from the model config at load time.

**Prompt file format:** JSONL, one object per line: `{"id": 0, "category": "code", "prompt": "..."}`. `category` is optional and defaults to `default`.

**Size budget:** 10,000 tokens × 32 layers is 320,000 rows, which should stay under about 10 MB in Parquet with zstd. Write in batches of 50,000 rows to keep memory flat.

## Features and CLI

Six commands cover the full loop: capture, analyze, simulate, prune, watch training, and view.

**1. `moescope profile`** — capture router traces

```
moescope profile --model ibm-granite/granite-3.1-3b-a800m-instruct \
  --prompts examples/code.jsonl --max-new-tokens 128 \
  --device mps --out runs/granite-code
```

- Runs prefill and greedy decode for each prompt; tags each row with `phase`.
- Fails fast with a clear message if no adapter matches the model architecture, and lists supported families.
- Shows a Rich progress bar with tokens/sec.

**2. `moescope analyze`** — workload report

```
moescope analyze runs/granite-code [--compare runs/granite-chat]
```

- Prints per-layer utilization summary, entropy, Coverage@90/95/99, and the 10 hottest and coldest experts per layer.
- `--compare` shows which experts gain or lose share between two workloads (Jensen-Shannon distance per layer).
- Writes `report.json` for the dashboard.

**3. `moescope simulate`** — cache what-if

```
moescope simulate runs/granite-code --budget-gb 2,4,6,8 \
  --policy lru,lfu,pinned
```

- Replays the trace in token order. `pinned` keeps the top experts by utilization resident and never evicts.
- Budget in GB is converted to an expert count with `expert_bytes` from the manifest; shared experts and non-expert weights are always resident.
- Output: hit rate per policy and budget, and estimated bytes moved per token.
- `recommend` step prints concrete settings for llama.cpp (an `--n-cpu-moe` value or a `--moe-cache-mib` size, depending on the user's llama.cpp version) and a pin list for `pinned`. Treat engine flags as version-dependent; check `llama-server --help` at runtime before printing them.

**4. `moescope prune`** — workload-specific pruning

```
moescope prune runs/granite-code --keep-coverage 99 \
  --eval examples/code_eval.jsonl --out runs/granite-code/prune
```

- Keeps the experts that reach the chosen coverage per layer; masks the rest by setting their router logits to negative infinity (no weight surgery in v1).
- Runs the eval twice (full and pruned) and reports exact-match or a user metric, plus the share of expert parameters masked.
- v2: export a physically smaller checkpoint.

**5. `moescope train-watch`** — fine-tuning diagnostics

```python
from moescope.train import MoEWatchCallback
trainer = Trainer(..., callbacks=[MoEWatchCallback(log_dir="runs/ft")])
```

- Logs every N steps: aux loss, z-loss, per-expert token share per layer, capacity drop rate where the model exposes it.
- Warns when one expert in any layer exceeds 3× its fair share for 200 steps (router collapse). Threshold is configurable.

**6. `moescope ui`** — local dashboard

- Opens `localhost:7878`. Views: layer × expert heatmap; Coverage curve per layer; co-activation matrix for a chosen layer; workload compare; cache-sim chart (hit rate vs budget, one line per policy); prune results; training curves.
- Read-only over the runs folder. No auth, binds to 127.0.0.1 only.

## Build plan

Five phases over about six weeks; each phase ends with a demo-able command and passing tests before the next starts.

**Phase 0 — Spike (2–3 days)**

- Load one small MoE model with transformers on Apple Silicon (MPS). First choice: a Granite MoE model (about 1B–3B total params); fallback: OLMoE on a CUDA box.
- Print the module tree, find the gate module, register a hook, and print top-k IDs for 5 tokens.
- *Done when:* hook output matches the model's own `output_router_logits=True` result (where supported) for the same input.

**Phase 1 — Capture (week 1)**

- Adapter interface + registry, `hf` backend, Parquet writer, `run.json`, `profile` command.
- Adapters for two families (Granite MoE, Qwen-MoE).
- *Done when:* `profile` on 50 prompts writes a valid trace; schema test passes; peak extra memory from tracing < 200 MB.

**Phase 2 — Analyze and simulate (week 2)**

- Utilization, entropy, Coverage@p, co-activation, `--compare`; LRU/LFU/pinned simulator; `recommend`.
- *Done when:* golden-trace tests pass (see Testing); code vs chat workloads show measurably different expert sets on at least one model.

**Phase 3 — Dashboard (week 3)**

- FastAPI JSON endpoints over runs; React views listed under `moescope ui`.
- *Done when:* `moescope ui` opens and renders every view from a sample run with no network access.

**Phase 4 — Prune and llama.cpp backend (weeks 4–5)**

- Router masking + eval loop. `llamacpp` backend via the eval callback, emitting the same schema.
- *Done when:* pruning at Coverage@99 reports quality and masked share; the llama.cpp trace for a GGUF model matches the HF trace of the same model on ≥ 95% of top-1 choices for identical prompts (quantization causes some drift).

**Phase 5 — Training callback and release (week 6)**

- `MoEWatchCallback`, collapse warning, training charts.
- README with a 2-minute quickstart, a GIF of the dashboard, and one written finding from a real run (for example, "code prompts used 41% of experts at Coverage@95" — whatever the real number is).
- Publish to PyPI as `moescope`.

## Testing and validation

Most tests run on synthetic traces with known answers, so CI never needs a GPU or a model download.

- **Golden traces.** A generator in `tests/fixtures/` builds traces with a planted distribution (for example, layer 0 sends 90% of tokens to experts {0, 1, 2}). Tests assert that utilization, Coverage@90, and entropy recover the planted values exactly.
- **Simulator correctness.** A 10-lookup hand-written trace with a 2-slot cache; assert exact hit counts for LRU, LFU, and pinned against values worked out by hand in the test file.
- **Toy MoE model.** A tiny PyTorch MoE layer (router → softmax → top-k → experts, written in the style of the reference repo's Sparse MoE) with fixed weights. The `hf`-style hook must recover the same top-k IDs the layer computes internally.
- **Schema tests.** Every backend's output passes `trace.schema.validate()`; wrong dtypes or list lengths fail.
- **Real-model smoke test** (manual, marked `@pytest.mark.slow`): profile 10 prompts on one small MoE model and check the trace is non-empty and every layer's routed-token counts sum to top\_k × tokens.
- **Dashboard.** Playwright test loads `moescope ui` against a fixture run and checks each view renders without console errors.

## Risks, open questions, and instructions for Claude Code

**Risks**

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Gate module names and output shapes differ per architecture and transformers version | Hooks silently capture nothing or the wrong tensor | One adapter per family; adapter self-test at load compares hook output with the model's own router logits |
| 16 GB Mac limits which models fit in bf16 | Can't test bigger models locally | Develop on small MoE models; run larger ones through the llama.cpp backend with quantized GGUF, or a rented GPU |
| Engine flags (llama.cpp MoE offload/cache) change between releases | `recommend` prints flags that don't exist | Detect available flags from `--help` at runtime; print the version checked |
| Quantized and full-precision models route slightly differently | Recommendations from one don't transfer exactly to the other | Always profile the exact model file the user will run; report top-1 agreement when comparing |
| Masking experts is not the same as removing them | Pruning shows quality impact but not real memory savings | v1 reports masked share as "potential savings"; v2 adds checkpoint export |

**Open questions**

- [ ] Which two model families to support first — confirm in Phase 0 based on what actually fits and hooks cleanly on MPS.
- [ ] Does the target llama.cpp version expose per-layer top-k tensors through the eval callback for all MoE GGUF architectures?
- [ ] Should `simulate` model prefetching (predicting next-layer experts), or stay with reactive policies in v1?

**Instructions for Claude Code**

1. Start with Phase 0 only. Report the gate module path, its output shape, and the hook result before writing any other code.
2. Build phases in order; do not start a phase until the previous phase's "done when" checks pass.
3. Every backend emits the exact `routes.parquet` schema in Data model. Change the schema only by editing `trace/schema.py` and this spec together.
4. Analysis and simulation code must not import torch; they work on Parquet/Polars only.
5. Write tests first for metrics and the simulator, using golden traces.
6. Never hardcode engine flags or model internals that can be read at runtime; read them and fail with a clear message if missing.
7. Keep the dashboard read-only, bound to 127.0.0.1, and fully offline (no CDN assets).
8. Ask before adding any dependency not listed in the tech stack.
