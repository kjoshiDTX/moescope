"""Build traces with a planted routing distribution, so metrics have known answers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from moescope.trace.schema import RouterEvent, RunManifest
from moescope.trace.writer import TraceWriter


def write_golden(run_dir: Path, plan: dict[int, list[tuple[tuple[int, ...], int]]], num_experts: int, top_k: int,
                 category: str = "default", expert_bytes: int = 1_000_000, shared: list[int] | None = None) -> Path:
    """plan[layer] = [(expert_ids, n_tokens), ...]. Every layer must cover the same number of tokens.

    Tokens are laid out in plan order, all in one prompt, all prefill.
    """
    totals = {sum(n for _, n in rows) for rows in plan.values()}
    assert len(totals) == 1, "every layer needs the same token count"
    with TraceWriter(run_dir) as w:
        for layer, rows in plan.items():
            ids = np.array([e for e, n in rows for _ in range(n)], dtype=np.int16).reshape(-1, top_k)
            n = len(ids)
            w.write(RouterEvent(prompt_id=0, category=category, phase="prefill", layer=layer,
                                token_pos=np.arange(n), expert_ids=ids, weights=np.full((n, top_k), 1 / top_k),
                                logit_max=np.zeros(n)))
        w.close(RunManifest("0.1.0", "golden", "golden", len(plan), num_experts, top_k, shared or [],
                            expert_bytes, "float16", "golden.jsonl", 0, "2026-10-12T00:00:00+00:00"))
    return Path(run_dir)


def planted_plan() -> dict:
    """Layer 0: 90% of 1000 tokens go to experts {0,1,2} (300 each); the other 100 spread
    10 each over experts 3..12. Layer 1: uniform over 16 experts. top_k = 1, 16 experts."""
    layer0 = [((0,), 300), ((1,), 300), ((2,), 300)] + [((e,), 10) for e in range(3, 13)]
    layer1 = [((e,), 1000 // 16 + (1 if e < 1000 % 16 else 0)) for e in range(16)]
    return {0: layer0, 1: layer1}
