"""Build report.json (what the dashboard reads) and render it for the terminal."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table

from ..trace.reader import Run
from .utilization import COVERAGE_LEVELS, count_matrix, coverage_curve, hot_cold, layer_stats, tokens_per_layer


def _matrix(m: np.ndarray, digits: int = 5) -> list[list[float]]:
    return np.round(m, digits).tolist()


def _utilization(routes: pl.DataFrame, L: int, E: int) -> dict:
    counts = count_matrix(routes, L, E)
    tokens = np.maximum(tokens_per_layer(routes, L), 1)[:, None]
    return {"tokens": int(routes.filter(pl.col("layer") == 0).height), "utilization": _matrix(counts / tokens)}


def build_report(run: Run) -> dict:
    m = run.manifest
    L, E = m.num_layers, m.num_experts
    routes = run.routes
    counts = count_matrix(routes, L, E)
    tokens = np.maximum(tokens_per_layer(routes, L), 1)[:, None]
    stats = layer_stats(routes, m)
    logit = routes.group_by("layer").agg(
        pl.col("logit_max").cast(pl.Float32).mean().alias("logit_max_mean"),
        pl.col("logit_max").cast(pl.Float32).quantile(0.99).alias("logit_max_p99"),
    )
    stats = stats.join(logit, on="layer", how="left").sort("layer")
    cov = {p: stats[f"coverage_{p}"].to_numpy() for p in COVERAGE_LEVELS}

    report = {
        "run": run.name,
        "manifest": asdict(m),
        "tokens": int(routes.filter(pl.col("layer") == 0).height),
        "prompts": int(routes["prompt_id"].n_unique()) if routes.height else 0,
        "categories": sorted(routes["category"].unique().to_list()),
        "phases": {p: int(routes.filter((pl.col("phase") == p) & (pl.col("layer") == 0)).height) for p in ("prefill", "decode")},
        "summary": {
            "mean_entropy_norm": float(stats["entropy_norm"].mean()),
            **{f"coverage_{p}_share": float(cov[p].sum() / (L * E)) for p in COVERAGE_LEVELS},
            "never_used_share": float((counts == 0).sum() / (L * E)),
        },
        "layers": stats.to_dicts(),
        "utilization": _matrix(counts / tokens),
        "weighted_utilization": _matrix(count_matrix(routes, L, E, weighted=True) / tokens),
        "coverage_curve": _matrix(coverage_curve(counts)),
        "hot_cold": hot_cold(counts),
        "by_category": {},
        "by_phase": {},
    }
    categories = report["categories"]
    if len(categories) > 1:
        report["by_category"] = {c: _utilization(routes.filter(pl.col("category") == c), L, E) for c in categories}
    for phase, n in report["phases"].items():
        if n:
            report["by_phase"][phase] = _utilization(routes.filter(pl.col("phase") == phase), L, E)
    return report


def render_report(report: dict, console: Console, top_n: int = 10) -> None:
    m = report["manifest"]
    s = report["summary"]
    console.print(
        f"[bold]{report['run']}[/] · {m['model_id']} · {report['prompts']} prompts · {report['tokens']:,} tokens "
        f"(prefill {report['phases']['prefill']:,}, decode {report['phases']['decode']:,}) · "
        f"{m['num_layers']} layers × {m['num_experts']} experts, top-{m['top_k']}"
    )
    table = Table(header_style="bold", box=None, pad_edge=False)
    for col, just in [("layer", "right"), ("active", "right"), ("entropy", "right"), ("cov90", "right"),
                      ("cov95", "right"), ("cov99", "right"), (f"hottest {top_n}", "left"), (f"coldest {top_n}", "left")]:
        table.add_column(col, justify=just)
    for row, hc in zip(report["layers"], report["hot_cold"]):
        table.add_row(
            str(row["layer"]),
            f"{row['active_experts']}/{m['num_experts']}",
            f"{row['entropy_norm']:.2f}",
            str(row["coverage_90"]),
            str(row["coverage_95"]),
            str(row["coverage_99"]),
            " ".join(str(x["expert"]) for x in hc["hottest"][:top_n]),
            " ".join(str(x["expert"]) for x in hc["coldest"][:top_n]),
        )
    console.print(table)
    console.print(
        f"Coverage@90/95/99 uses [bold]{s['coverage_90_share']:.0%}[/] / [bold]{s['coverage_95_share']:.0%}[/] / "
        f"[bold]{s['coverage_99_share']:.0%}[/] of all routed experts · never used: {s['never_used_share']:.0%} · "
        f"mean normalized entropy {s['mean_entropy_norm']:.2f} (1.0 = perfectly even)"
    )


def render_compare(result: dict, console: Console) -> None:
    console.print(f"[bold]{result['a']}[/] → [bold]{result['b']}[/] · mean Jensen-Shannon distance {result['mean_js_distance']:.3f} (0 = same experts, 1 = disjoint)")
    table = Table(header_style="bold", box=None, pad_edge=False)
    for col in ("layer", "JS dist", "gain share in b", "lose share in b"):
        table.add_column(col)
    for layer in result["layers"]:
        table.add_row(
            str(layer["layer"]),
            f"{layer['js_distance']:.3f}",
            " ".join(f"{g['expert']}(+{g['delta']:.1%})" for g in layer["gainers"][:3]),
            " ".join(f"{g['expert']}({g['delta']:.1%})" for g in layer["losers"][:3]),
        )
    console.print(table)
