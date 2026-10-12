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
):
    """Run the model on your prompts and record every router decision."""
    from .adapters.registry import UnsupportedModelError
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

        engine = LlamaCppBackend(max_new_tokens=max_new_tokens)
    else:
        fail(f"unknown backend {backend!r}; choose hf or llamacpp")

    with console.status(f"Loading {model} …"):
        try:
            info = engine.load(model)
        except UnsupportedModelError as e:
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
        for p in prompt_list:
            for event in engine.run([p]):
                writer.write(event)
                if event.layer == 0:
                    tokens += len(event)
            progress.update(task, advance=1, rate=f"{tokens / (time.perf_counter() - start):.1f} tok/s")
        writer.close(manifest)

    validate(out)
    console.print(f"Wrote {writer.rows_written:,} rows ({tokens:,} tokens) to [bold]{out}[/]")


if __name__ == "__main__":
    app()
