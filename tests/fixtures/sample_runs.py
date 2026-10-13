"""A complete synthetic runs folder: two profile runs, a simulation, a prune result, a training log.

Used by the dashboard tests and for screenshots: `python -m tests.fixtures.sample_runs <dir>`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from moescope.analysis.report import build_report
from moescope.analysis.utilization import count_matrix, coverage_sets
from moescope.sim.recommend import LlamaCpp, recommend
from moescope.sim.simulate import pin_list, simulate_run
from moescope.trace.reader import load_run
from moescope.trace.schema import RouterEvent, RunManifest
from moescope.trace.writer import TraceWriter

L, E, K = 8, 16, 2
EXPERT_BYTES = 8 * 2**20


def _profile(run_dir: Path, category: str, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    # Each layer prefers a different handful of experts; the category shifts the preference.
    prefs = rng.dirichlet(np.full(E, 0.4), size=L)
    with TraceWriter(run_dir) as w:
        for prompt_id in range(6):
            n_prompt, n_new = int(rng.integers(12, 30)), int(rng.integers(10, 25))
            for phase, start, n in (("prefill", 0, n_prompt), ("decode", n_prompt, n_new)):
                for layer in range(L):
                    logits = np.log(prefs[layer]) + rng.gumbel(size=(n, E))
                    ids = np.argsort(-logits, axis=1)[:, :K]
                    top = np.take_along_axis(logits, ids, 1)
                    weights = np.exp(top - top.max(1, keepdims=True))
                    weights /= weights.sum(1, keepdims=True)
                    w.write(RouterEvent(prompt_id, category, phase, layer, np.arange(start, start + n), ids,
                                        weights, logits.max(1)))
        w.close(RunManifest("0.1.0", "hf", "sample/tiny-moe", L, E, K, [], EXPERT_BYTES, "bfloat16",
                            f"examples/{category}.jsonl", 32, "2026-10-12T18:00:00-05:00"))
    run = load_run(run_dir)
    (run_dir / "report.json").write_text(json.dumps(build_report(run)))
    results = simulate_run(run, [0.25, 0.5, 0.75, 1.0])
    rec = recommend(run.manifest, results, LlamaCpp(None, error="llama.cpp not found on PATH"))
    (run_dir / "sim.json").write_text(json.dumps({"phase": "all", "results": results, "recommendation": rec,
                                                  "pins": pin_list(run, rec["choice"]["slots"])}))
    return run_dir


def _prune(out: Path, source: Path) -> None:
    run = load_run(source)
    counts = count_matrix(run.routes, L, E)
    results = []
    for p, score in ((90, 0.55), (95, 0.65), (99, 0.7)):
        kept = coverage_sets(counts, p)
        masked = L * E - sum(len(k) for k in kept)
        results.append({"keep_coverage": p, "masked_experts": masked, "masked_share": masked / (L * E),
                        "masked_bytes": masked * EXPERT_BYTES, "score": score, "kept": kept})
    examples = [{"id": i, "prompt": f"What is {i} + {i}?", "answer": str(2 * i),
                 "outputs": {"full": str(2 * i), "90": str(2 * i + (i % 3 == 0)), "95": str(2 * i), "99": str(2 * i)},
                 "scores": {"full": 1.0, "90": float(i % 3 != 0), "95": 1.0, "99": 1.0}} for i in range(5)]
    out.mkdir(parents=True, exist_ok=True)
    (out / "prune.json").write_text(json.dumps({
        "source_run": source.name, "model_id": "sample/tiny-moe", "eval_file": "examples/code_eval.jsonl",
        "metric": "exact", "num_layers": L, "num_experts": E, "expert_bytes": EXPERT_BYTES,
        "baseline": {"score": 0.7, "n": 20}, "results": results, "examples": examples,
    }))


def _train(out: Path) -> None:
    rng = np.random.default_rng(7)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_meta.json").write_text(json.dumps({
        "model_id": "sample/tiny-moe", "num_layers": L, "num_experts": E, "top_k": K,
        "log_every": 10, "collapse_factor": 3.0, "collapse_steps": 200}))
    lines = []
    skew = np.zeros(L)
    for step in range(10, 610, 10):
        skew[3] += 0.06  # layer 3 slowly collapses onto expert 0
        load = rng.dirichlet(np.full(E, 30.0), size=L)
        load[3] = (1 - min(0.5, skew[3] / 10)) * load[3]
        load[3, 0] += 1 - load[3].sum()
        ratio = (load.max(1) * E).tolist()
        lines.append(json.dumps({
            "step": step, "aux_loss": float(1.0 + 0.3 * np.exp(-step / 150) + rng.normal(0, 0.01)),
            "z_loss": float(2.0 + 0.5 * np.exp(-step / 200) + rng.normal(0, 0.02)),
            "logit_max_mean": float(3 + step / 300), "load": np.round(load, 5).tolist(), "max_load_ratio": ratio,
            "drop_rate": None, "collapsed_layers": [3] if step >= 400 else [],
        }))
    (out / "train.jsonl").write_text("\n".join(lines) + "\n")


def make_sample_runs(root: Path | str) -> Path:
    root = Path(root)
    code = _profile(root / "sample-code", "code", seed=1)
    _profile(root / "sample-chat", "chat", seed=2)
    _prune(code / "prune", code)
    _train(root / "ft")
    return root


if __name__ == "__main__":
    print(make_sample_runs(sys.argv[1] if len(sys.argv) > 1 else "runs/sample"))
