"""Load a saved run into Polars. Analysis code starts here; nothing below this imports torch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .schema import ROUTES_FILE, RunManifest


@dataclass
class Run:
    path: Path
    manifest: RunManifest
    routes: pl.DataFrame

    @property
    def name(self) -> str:
        return self.path.name


def load_run(run_dir: Path | str, phase: str | None = None, category: str | None = None) -> Run:
    run_dir = Path(run_dir)
    manifest = RunManifest.read(run_dir)
    lf = pl.scan_parquet(run_dir / ROUTES_FILE)
    if phase and phase != "all":
        lf = lf.filter(pl.col("phase") == phase)
    if category:
        lf = lf.filter(pl.col("category") == category)
    return Run(run_dir, manifest, lf.collect())


def find_runs(root: Path | str) -> list[Path]:
    """Every directory under root (including root) that holds a run.json."""
    root = Path(root)
    return sorted(p.parent for p in root.rglob("run.json"))
