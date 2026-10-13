"""Run an eval set with and without expert masks and report the quality / memory tradeoff."""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable

import numpy as np
import torch

from ..analysis.utilization import count_matrix, coverage_sets
from ..backends.hf import HFBackend
from ..prompts import Prompt
from ..trace.reader import Run
from .mask import RouterMask, keep_sets_with_min

Metric = Callable[[str, str], float]


def _normalize(text: str) -> str:
    text = text.strip().splitlines()[0].strip() if text.strip() else ""
    text = text.strip("`\"' ")
    return re.sub(r"[.\s]+$", "", text)


def exact_match(prediction: str, reference: str) -> float:
    """First line of the output, trimmed of quotes, backticks and a trailing period, equals the answer."""
    return float(_normalize(prediction) == _normalize(reference))


def contains(prediction: str, reference: str) -> float:
    return float(reference.strip().lower() in prediction.lower())


BUILTIN_METRICS: dict[str, Metric] = {"exact": exact_match, "contains": contains}


def load_metric(spec: str) -> Metric:
    """'exact', 'contains', or 'package.module:function' taking (prediction, reference) -> float."""
    if spec in BUILTIN_METRICS:
        return BUILTIN_METRICS[spec]
    if ":" not in spec:
        raise ValueError(f"unknown metric {spec!r}; use exact, contains, or module:function")
    module, fn = spec.split(":", 1)
    return getattr(importlib.import_module(module), fn)


@torch.no_grad()
def generate(backend: HFBackend, prompt: str, max_new_tokens: int) -> str:
    ids = backend.encode(prompt)
    pad = getattr(backend.tokenizer, "pad_token_id", None) or getattr(backend.tokenizer, "eos_token_id", None)
    out = backend.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=pad)
    return backend.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def keep_sets(run: Run, coverage_pct: float, top_k: int) -> list[list[int]]:
    m = run.manifest
    counts = count_matrix(run.routes, m.num_layers, m.num_experts)
    ranked = [np.argsort(-row, kind="stable").tolist() for row in counts]
    return keep_sets_with_min(coverage_sets(counts, coverage_pct, always_keep=m.shared_experts), ranked, top_k)


def prune_eval(
    backend: HFBackend,
    run: Run,
    examples: list[Prompt],
    coverages: list[float],
    metric: Metric,
    max_new_tokens: int = 32,
    progress: Callable[[str], None] | None = None,
) -> dict:
    m = run.manifest
    if backend.info.num_layers != m.num_layers or backend.info.num_experts != m.num_experts:
        raise ValueError(
            f"model has {backend.info.num_layers}x{backend.info.num_experts} MoE layers x experts but the run "
            f"profiled {m.num_layers}x{m.num_experts}; prune with the model that was profiled"
        )
    routed_total = m.num_layers * (m.num_experts - len(m.shared_experts))
    rows = [{"id": ex.id, "prompt": ex.prompt, "answer": ex.answer, "outputs": {}, "scores": {}} for ex in examples]

    def run_eval(key: str) -> float:
        for row, ex in zip(rows, examples):
            if progress:
                progress(key)
            text = generate(backend, ex.prompt, max_new_tokens)
            row["outputs"][key] = text.strip()
            row["scores"][key] = float(metric(text, ex.answer))
        return float(np.mean([r["scores"][key] for r in rows]))

    baseline = run_eval("full")
    results = []
    for cov in coverages:
        kept = keep_sets(run, cov, m.top_k)
        masked = routed_total - sum(len(set(k) - set(m.shared_experts)) for k in kept)
        with RouterMask(backend.adapter, backend.routers, kept):
            score = run_eval(f"{cov:g}")
        results.append({
            "keep_coverage": cov,
            "masked_experts": masked,
            "masked_share": masked / routed_total,
            "masked_bytes": masked * m.expert_bytes,
            "score": score,
            "kept": kept,
        })
    return {
        "source_run": run.name,
        "model_id": backend.info.model_id,
        "num_layers": m.num_layers,
        "num_experts": m.num_experts,
        "expert_bytes": m.expert_bytes,
        "baseline": {"score": baseline, "n": len(examples)},
        "results": results,
        "examples": rows,
    }
