"""moescope command line. Heavy imports (torch, transformers) happen inside the commands that need them."""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from . import __version__

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Profile how your workload uses the experts in an MoE model.")
console = Console()
err = Console(stderr=True)


@app.callback()
def main() -> None:
    """Profile how your workload uses the experts in an MoE model."""


def fail(message: str) -> None:
    err.print(f"[bold red]error:[/] {message}")
    raise typer.Exit(1)


@app.command()
def profile(
    model: Annotated[str, typer.Option(help="Hugging Face model id or local path")],
    prompts: Annotated[Path, typer.Option(help="JSONL prompt file", exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option(help="Run directory to create")],
    max_new_tokens: Annotated[int, typer.Option(help="Greedy decode steps per prompt")] = 128,
    device: Annotated[str | None, typer.Option(help="cuda, mps or cpu (default: best available)")] = None,
    dtype: Annotated[str | None, typer.Option(help="Load dtype, e.g. bfloat16 (default: model's own)")] = None,
    backend: Annotated[str, typer.Option(help="Capture backend")] = "hf",
    raw_prompts: Annotated[bool, typer.Option("--raw-prompts", help="Skip the tokenizer chat template")] = False,
    limit: Annotated[int | None, typer.Option(help="Only use the first N prompts")] = None,
    tokenizer: Annotated[str | None, typer.Option(help="llamacpp only: tokenize with this HF tokenizer so tokens match an hf run")] = None,
):
    """Run the model on your prompts and record every router decision."""
    from .adapters.registry import UnsupportedModelError
    from .backends.llamacpp import LlamaCppError
    from .prompts import load_prompts
    from .trace.schema import RunManifest, validate
    from .trace.writer import TraceWriter

    if (out / "routes.parquet").exists():
        fail(f"{out} already holds a trace; choose a new --out")
    try:
        prompt_list = load_prompts(prompts)[:limit]
    except ValueError as e:
        fail(str(e))

    if backend == "hf":
        from .backends.hf import HFBackend

        engine = HFBackend(device=device, dtype=dtype, max_new_tokens=max_new_tokens, chat_template=not raw_prompts)
    elif backend == "llamacpp":
        from .backends.llamacpp import LlamaCppBackend

        hf_tok = None
        if tokenizer:
            from transformers import AutoTokenizer

            hf_tok = AutoTokenizer.from_pretrained(tokenizer)
        engine = LlamaCppBackend(max_new_tokens=max_new_tokens, chat_template=not raw_prompts, tokenizer=hf_tok)
    else:
        fail(f"unknown backend {backend!r}; choose hf or llamacpp")

    with console.status(f"Loading {model} …"):
        try:
            info = engine.load(model)
        except (UnsupportedModelError, LlamaCppError) as e:
            fail(str(e))
    if hasattr(engine, "self_test"):
        with console.status("Checking router hooks against the model's own router logits …"):
            try:
                engine.self_test()
            except RuntimeError as e:
                fail(str(e))
    console.print(
        f"[bold]{info.model_id}[/] · {info.family} · {info.num_layers} MoE layers × {info.num_experts} experts, "
        f"top-{info.top_k} · {info.expert_bytes / 2**20:.1f} MiB/expert · {getattr(engine, 'device', '')}"
    )

    manifest = RunManifest(
        moescope_version=__version__,
        backend=engine.name,
        model_id=info.model_id,
        num_layers=info.num_layers,
        num_experts=info.num_experts,
        top_k=info.top_k,
        shared_experts=info.shared_experts,
        expert_bytes=info.expert_bytes,
        dtype=info.dtype,
        prompts_file=str(prompts),
        max_new_tokens=max_new_tokens,
        created_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    tokens = 0
    start = time.perf_counter()
    columns = [TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TextColumn("{task.fields[rate]}"), TimeElapsedColumn()]
    with TraceWriter(out) as writer, Progress(*columns, console=console) as progress:
        task = progress.add_task("prompts", total=len(prompt_list), rate="")
        seen: set[int] = set()
        for event in engine.run(prompt_list):
            writer.write(event)
            if event.prompt_id not in seen:  # a new prompt started, so the previous one finished
                if seen:
                    progress.update(task, advance=1)
                seen.add(event.prompt_id)
            if event.layer == 0:
                tokens += len(event)
                progress.update(task, rate=f"{tokens / (time.perf_counter() - start):.1f} tok/s")
        progress.update(task, completed=len(prompt_list))
        writer.close(manifest)

    validate(out)
    console.print(f"Wrote {writer.rows_written:,} rows ({tokens:,} tokens) to [bold]{out}[/]")


def _load(run_dir: Path, phase: str = "all", category: str | None = None):
    from .trace.reader import load_run

    if phase not in ("all", "prefill", "decode"):
        fail("--phase must be all, prefill or decode")
    try:
        return load_run(run_dir, phase=phase, category=category)
    except FileNotFoundError as e:
        fail(str(e))


def _floats(text: str, name: str) -> list[float]:
    try:
        return [float(x) for x in text.split(",") if x.strip()]
    except ValueError:
        fail(f"{name} must be a comma-separated list of numbers, got {text!r}")


@app.command()
def analyze(
    run_dir: Annotated[Path, typer.Argument(help="Run directory from `moescope profile`")],
    compare: Annotated[Path | None, typer.Option(help="Second run to compare against")] = None,
    phase: Annotated[str, typer.Option(help="all, prefill or decode")] = "all",
    category: Annotated[str | None, typer.Option(help="Only analyze prompts with this category")] = None,
    top: Annotated[int, typer.Option(help="Hottest/coldest experts to list per layer")] = 10,
):
    """Per-layer utilization, entropy, coverage, and hot/cold experts. Writes report.json."""
    import json

    from .analysis.compare import compare_runs
    from .analysis.report import build_report, render_compare, render_report

    run = _load(run_dir, phase, category)
    report = build_report(run)
    render_report(report, console, top_n=top)
    if phase == "all" and category is None:
        (run_dir / "report.json").write_text(json.dumps(report))
        console.print(f"Wrote {run_dir / 'report.json'}")
    if compare:
        try:
            result = compare_runs(run, _load(compare, phase, category))
        except ValueError as e:
            fail(str(e))
        console.print()
        render_compare(result, console)


@app.command()
def simulate(
    run_dir: Annotated[Path, typer.Argument(help="Run directory from `moescope profile`")],
    budget_gb: Annotated[str, typer.Option(help="Comma-separated expert-cache budgets in GiB")] = "2,4,6,8",
    policy: Annotated[str, typer.Option(help="Comma-separated: lru, lfu, pinned")] = "lru,lfu,pinned",
    phase: Annotated[str, typer.Option(help="all, prefill or decode")] = "all",
    target_hit_rate: Annotated[float, typer.Option(help="Hit rate the recommendation aims for")] = 0.95,
    llama_bin: Annotated[str | None, typer.Option(help="llama-server or llama-cli binary to read flags from")] = None,
):
    """Replay the trace through expert caches and recommend engine settings. Writes sim.json."""
    import json

    from rich.table import Table

    from .sim.recommend import detect_llamacpp, recommend
    from .sim.simulate import pin_list, simulate_run

    run = _load(run_dir, phase)
    budgets = _floats(budget_gb, "--budget-gb")
    policies = [p.strip() for p in policy.split(",") if p.strip()]
    with console.status("Replaying trace …"):
        try:
            results = simulate_run(run, budgets, policies)
        except ValueError as e:
            fail(str(e))

    m = run.manifest
    console.print(f"[bold]{run.name}[/] · {results[0]['lookups']:,} expert lookups · "
                  f"{m.expert_bytes / 2**20:.1f} MiB per expert · {m.num_layers * m.num_experts} routed experts "
                  f"({m.num_layers * m.num_experts * m.expert_bytes / 2**30:.2f} GiB)")
    table = Table(header_style="bold", box=None, pad_edge=False)
    table.add_column("budget", justify="right")
    table.add_column("slots", justify="right")
    for p in policies:
        table.add_column(f"{p} hit", justify="right")
        table.add_column(f"{p} MiB/tok", justify="right")
    for b in budgets:
        rows = {r["policy"]: r for r in results if r["budget_gb"] == b}
        cells = [f"{b:g} GiB", str(next(iter(rows.values()))["slots"])]
        for p in policies:
            cells += [f"{rows[p]['hit_rate']:.1%}", f"{rows[p]['bytes_per_token'] / 2**20:.1f}"]
        table.add_row(*cells)
    console.print(table)

    llama = detect_llamacpp(llama_bin)
    rec = recommend(m, results, llama, target_hit_rate)
    choice = rec["choice"]
    pins = pin_list(run, choice["slots"])
    verdict = "reaches" if choice["meets_target"] else "is the closest to"
    console.print(f"\n[bold]Recommendation:[/] {choice['budget_gb']:g} GiB with {choice['policy']} "
                  f"({choice['hit_rate']:.1%} hit rate) {verdict} the {target_hit_rate:.0%} target.")
    if llama.error:
        console.print(f"[yellow]{llama.error}[/]; engine flags not printed.")
    else:
        console.print(f"Checked {llama.binary} (version {llama.version or 'unknown'}).")
        if rec["settings"]:
            console.print("  " + " ".join(f"{s['flag']}" + (f" {s['value']}" if s["value"] is not None else "") for s in rec["settings"]))
            for s in rec["settings"]:
                console.print(f"  [dim]{s['flag']}: {s['why']}[/]")
        else:
            console.print(f"  This llama.cpp build lists none of the MoE offload flags moescope knows; flags not printed.")
    console.print(f"Pin list for the pinned policy: {len(pins)} (layer, expert) pairs, saved in sim.json.")
    (run_dir / "sim.json").write_text(json.dumps({"phase": phase, "results": results, "recommendation": rec, "pins": pins}))
    console.print(f"Wrote {run_dir / 'sim.json'}")


@app.command()
def prune(
    run_dir: Annotated[Path, typer.Argument(help="Run directory from `moescope profile`")],
    eval_file: Annotated[Path, typer.Option("--eval", help="JSONL eval set with prompt and answer", exists=True, dir_okay=False)],
    keep_coverage: Annotated[str, typer.Option(help="Comma-separated coverage levels to keep, e.g. 90,95,99")] = "99",
    out: Annotated[Path | None, typer.Option(help="Output directory (default: <run>/prune)")] = None,
    metric: Annotated[str, typer.Option(help="exact, contains, or module:function(prediction, reference)")] = "exact",
    max_new_tokens: Annotated[int, typer.Option(help="Generation length per eval example")] = 32,
    model: Annotated[str | None, typer.Option(help="Model to load (default: the run's model)")] = None,
    device: Annotated[str | None, typer.Option(help="cuda, mps or cpu")] = None,
    dtype: Annotated[str | None, typer.Option(help="Load dtype, e.g. bfloat16")] = None,
    limit: Annotated[int | None, typer.Option(help="Only use the first N eval examples")] = None,
):
    """Mask experts outside each coverage set and re-run an eval. Writes prune.json."""
    import json

    from rich.table import Table

    from .backends.hf import HFBackend
    from .prompts import load_prompts
    from .prune.evaluate import load_metric, prune_eval

    run = _load(run_dir)
    if run.manifest.backend != "hf" and model is None:
        fail(f"run was captured with the {run.manifest.backend} backend; pass --model with the matching HF model")
    coverages = _floats(keep_coverage, "--keep-coverage")
    if not coverages or any(not 0 < c <= 100 for c in coverages):
        fail("--keep-coverage values must be in (0, 100]")
    try:
        examples = load_prompts(eval_file)[:limit]
        score = load_metric(metric)
    except (ValueError, ImportError, AttributeError) as e:
        fail(str(e))
    missing = [ex.id for ex in examples if ex.answer is None]
    if missing:
        fail(f"eval examples need an 'answer' field; missing on ids {missing[:5]}")

    backend = HFBackend(device=device, dtype=dtype)
    with console.status(f"Loading {model or run.manifest.model_id} …"):
        backend.load(model or run.manifest.model_id)
    total = len(examples) * (len(coverages) + 1)
    columns = [TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn()]
    with Progress(*columns, console=console) as progress:
        task = progress.add_task("eval", total=total)
        try:
            result = prune_eval(backend, run, examples, sorted(coverages, reverse=True), score, max_new_tokens,
                                progress=lambda key: progress.update(task, advance=1, description=f"eval ({key})"))
        except ValueError as e:
            fail(str(e))
    result.update({"eval_file": str(eval_file), "metric": metric})

    table = Table(header_style="bold", box=None, pad_edge=False)
    for col in ("experts kept", "masked", "potential savings", metric):
        table.add_column(col, justify="right")
    table.add_row("all", "0%", "–", f"{result['baseline']['score']:.1%}")
    for r in result["results"]:
        table.add_row(f"Cov@{r['keep_coverage']:g}", f"{r['masked_share']:.1%}", f"{r['masked_bytes'] / 2**20:,.0f} MiB", f"{r['score']:.1%}")
    console.print(table)
    console.print("[dim]Masked experts stay loaded; savings are what a pruned checkpoint would free.[/]")
    out = out or run_dir / "prune"
    out.mkdir(parents=True, exist_ok=True)
    (out / "prune.json").write_text(json.dumps(result))
    console.print(f"Wrote {out / 'prune.json'}")


@app.command("train-watch")
def train_watch(
    log_dir: Annotated[Path, typer.Argument(help="log_dir given to MoEWatchCallback")],
):
    """Summarize a MoEWatchCallback training log: losses, load balance, collapse warnings.

    Add the callback in Python: Trainer(..., callbacks=[MoEWatchCallback(log_dir="runs/ft")])
    """
    import json

    from rich.table import Table

    log = log_dir / "train.jsonl"
    if not log.exists():
        fail(f"no train.jsonl in {log_dir}; add MoEWatchCallback(log_dir={str(log_dir)!r}) to your Trainer")
    steps = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    meta_path = log_dir / "train_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if not steps:
        fail(f"{log} has no logged steps yet")
    console.print(f"[bold]{log_dir}[/] · {meta.get('model_id', '?')} · {len(steps)} logged steps, last step {steps[-1]['step']}")
    table = Table(header_style="bold", box=None, pad_edge=False)
    for col in ("step", "aux loss", "z-loss", "max load ratio", "worst layer", "collapsed"):
        table.add_column(col, justify="right")
    stride = max(1, len(steps) // 12)
    for s in steps[::stride] + ([steps[-1]] if (len(steps) - 1) % stride else []):
        worst = max(range(len(s["max_load_ratio"])), key=lambda l: s["max_load_ratio"][l])
        table.add_row(str(s["step"]), f"{s['aux_loss']:.4f}", f"{s['z_loss']:.3f}",
                      f"{s['max_load_ratio'][worst]:.2f}×", str(worst), ", ".join(map(str, s["collapsed_layers"])) or "–")
    console.print(table)
    collapsed = sorted({l for s in steps for l in s["collapsed_layers"]})
    if collapsed:
        first = {l: next(s["step"] for s in steps if l in s["collapsed_layers"]) for l in collapsed}
        err.print(f"[bold red]Router collapse:[/] " + ", ".join(f"layer {l} from step {first[l]}" for l in collapsed)
                  + f" (one expert above {meta.get('collapse_factor', 3)}× fair share for {meta.get('collapse_steps', 200)}+ steps)")
    else:
        console.print("No router collapse detected.")


@app.command()
def ui(
    runs_dir: Annotated[Path, typer.Argument(help="Folder of runs to serve")] = Path("runs"),
    port: Annotated[int, typer.Option(help="Port on 127.0.0.1")] = 7878,
    open_browser: Annotated[bool, typer.Option("--open/--no-open", help="Open a browser tab")] = True,
):
    """Open the local, read-only dashboard over a runs folder."""
    import threading
    import webbrowser

    import uvicorn

    from .server.api import STATIC, create_app

    if not runs_dir.is_dir():
        fail(f"{runs_dir} is not a directory")
    if not (STATIC / "index.html").exists():
        err.print("[yellow]Dashboard assets are missing; run `npm run build` in web/. Serving the API only.[/]")
    url = f"http://127.0.0.1:{port}"
    console.print(f"moescope dashboard on [bold]{url}[/] (serving {runs_dir.resolve()}) · Ctrl+C to stop")
    if open_browser:
        threading.Timer(1.0, webbrowser.open, args=[url]).start()
    uvicorn.run(create_app(runs_dir), host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    app()
