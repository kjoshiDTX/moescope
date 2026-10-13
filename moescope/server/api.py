"""Read-only JSON API over a runs folder, plus the built dashboard. Never writes to disk."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse

from ..analysis.coactivation import coactivation, lift
from ..analysis.compare import compare_runs
from ..analysis.report import build_report
from ..trace.reader import find_runs, load_run
from ..trace.schema import RunManifest

STATIC = Path(__file__).parent / "static"


def create_app(runs_dir: Path | str) -> FastAPI:
    root = Path(runs_dir).resolve()
    app = FastAPI(title="moescope", docs_url=None, redoc_url=None, openapi_url=None)

    def resolve(name: str) -> Path:
        path = (root / name).resolve()
        if path != root and root not in path.parents:
            raise HTTPException(404, "not found")
        return path

    def run_dir(name: str) -> Path:
        path = resolve(name)
        if not (path / "run.json").exists():
            raise HTTPException(404, f"no run named {name!r}")
        return path

    def rel(path: Path) -> str:
        return str(path.relative_to(root)) if path != root else "."

    @lru_cache(maxsize=8)
    def cached_run(path: Path, mtime: float):
        return load_run(path)

    def get_run(name: str):
        path = run_dir(name)
        return cached_run(path, (path / "routes.parquet").stat().st_mtime)

    def read_json(path: Path):
        return json.loads(path.read_text())

    @app.get("/api/runs")
    def runs():
        out = []
        for path in find_runs(root):
            m = RunManifest.read(path)
            report = path / "report.json"
            tokens = read_json(report).get("tokens") if report.exists() else None
            out.append({
                "name": rel(path),
                "manifest": m.__dict__,
                "tokens": tokens,
                "has_sim": (path / "sim.json").exists(),
            })
        return out

    @app.get("/api/runs/{name:path}/report")
    def report(name: str):
        path = run_dir(name)
        cached = path / "report.json"
        routes = path / "routes.parquet"
        if cached.exists() and cached.stat().st_mtime >= routes.stat().st_mtime:
            return read_json(cached)
        return build_report(get_run(name))

    @app.get("/api/runs/{name:path}/coactivation")
    def coact(name: str, layer: int = Query(0, ge=0)):
        run = get_run(name)
        m = run.manifest
        if layer >= m.num_layers:
            raise HTTPException(400, f"layer must be < {m.num_layers}")
        co = coactivation(run.routes, layer, m.num_experts)
        tokens = int((run.routes["layer"] == layer).sum())
        return {"layer": layer, "tokens": tokens, "counts": co.tolist(), "lift": lift(co, tokens).round(4).tolist()}

    @app.get("/api/runs/{name:path}/sim")
    def sim(name: str):
        path = run_dir(name) / "sim.json"
        if not path.exists():
            raise HTTPException(404, f"no simulation yet; run `moescope simulate {name}`")
        return read_json(path)

    @app.get("/api/compare")
    def compare(a: str, b: str):
        try:
            return compare_runs(get_run(a), get_run(b))
        except ValueError as e:
            raise HTTPException(400, str(e)) from None

    @app.get("/api/prune")
    def prune_results():
        return [{"name": rel(p.parent), **read_json(p)} for p in sorted(root.rglob("prune.json"))]

    @app.get("/api/train")
    def train_logs():
        out = []
        for log in sorted(root.rglob("train.jsonl")):
            meta = log.parent / "train_meta.json"
            steps = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
            out.append({"name": rel(log.parent), "meta": read_json(meta) if meta.exists() else {}, "steps": steps})
        return out

    @app.get("/{path:path}", include_in_schema=False)
    def static(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "not found")
        if not (STATIC / "index.html").exists():
            return HTMLResponse("<p>Dashboard not built. Run <code>npm run build</code> in <code>web/</code>.</p>", 503)
        target = (STATIC / path).resolve()
        if path and target.is_file() and STATIC.resolve() in target.parents:
            return FileResponse(target)
        return FileResponse(STATIC / "index.html")

    return app
