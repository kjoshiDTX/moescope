"""llama.cpp backend for GGUF models.

A small C++ helper (llamacpp_trace.cpp, next to this file) reads the per-layer router tensors
through llama.cpp's eval callback. It is compiled on first use against the installed libllama
and cached under ~/.cache/moescope, keyed by the source and the llama.cpp version.

One difference from the hf backend: llama.cpp evaluates the last layer only for tokens whose
logits it needs, so in prefill the last MoE layer routes just the final prompt token. The trace
records what llama.cpp actually ran; those skipped experts are never loaded by the engine either.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..prompts import Prompt
from ..trace.schema import RouterEvent
from .base import ModelInfo

SOURCE = Path(__file__).with_name("llamacpp_trace.cpp")
CACHE = Path(os.environ.get("MOESCOPE_CACHE", Path.home() / ".cache" / "moescope"))


class LlamaCppError(RuntimeError):
    pass


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


def _build_flags() -> list[str]:
    """Compiler flags for libllama and ggml, from pkg-config, falling back to Homebrew prefixes."""
    if shutil.which("pkg-config"):
        try:
            flags = _run(["pkg-config", "--cflags", "--libs", "llama", "ggml"]).split()
            libdirs = [f[2:] for f in flags if f.startswith("-L")]
            return flags + [f"-Wl,-rpath,{d}" for d in libdirs]
        except subprocess.CalledProcessError:
            pass
    if shutil.which("brew"):
        try:
            prefixes = [_run(["brew", "--prefix", name]) for name in ("llama.cpp", "ggml")]
        except subprocess.CalledProcessError:
            prefixes = []
        if prefixes and all(Path(p, "include").exists() for p in prefixes):
            flags = [f"-I{p}/include" for p in prefixes] + [f"-L{p}/lib" for p in prefixes]
            return flags + ["-lllama", "-lggml", "-lggml-base"] + [f"-Wl,-rpath,{p}/lib" for p in prefixes]
    raise LlamaCppError(
        "llama.cpp headers and libraries not found. Install llama.cpp (e.g. `brew install llama.cpp`) "
        "so that pkg-config can find `llama` and `ggml`."
    )


def build_helper(force: bool = False) -> Path:
    flags = _build_flags()
    key = hashlib.sha256(SOURCE.read_bytes() + " ".join(flags).encode()).hexdigest()[:12]
    binary = CACHE / f"llamacpp-trace-{key}"
    if binary.exists() and not force:
        return binary
    compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        raise LlamaCppError("no C++ compiler found (need c++, clang++ or g++) to build the llama.cpp tracer")
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = binary.with_suffix(".tmp")
    proc = subprocess.run([compiler, "-std=c++17", "-O2", str(SOURCE), "-o", str(tmp), *flags], capture_output=True, text=True)
    if proc.returncode != 0:
        raise LlamaCppError(f"building the llama.cpp tracer failed; your llama.cpp may be too old or too new:\n{proc.stderr[-2000:]}")
    tmp.rename(binary)
    return binary


def read_records(path: Path, offset: int = 0) -> tuple[list[tuple], int]:
    """Parse complete records from the helper's output starting at byte offset."""
    data = path.read_bytes()[offset:]
    records, pos = [], 0
    while pos + 24 <= len(data):
        prompt_id, phase, token_pos, layer, n, k = np.frombuffer(data, np.int32, 6, pos)
        size = 24 + n * k * 8 + n * 4
        if pos + size > len(data):
            break
        body = pos + 24
        ids = np.frombuffer(data, np.int32, n * k, body).reshape(n, k)
        weights = np.frombuffer(data, np.float32, n * k, body + n * k * 4).reshape(n, k)
        lmax = np.frombuffer(data, np.float32, n, body + n * k * 8)
        records.append((int(prompt_id), int(phase), int(token_pos), int(layer), ids, weights, lmax))
        pos += size
    return records, offset + pos


class LlamaCppBackend:
    name = "llamacpp"

    def __init__(self, max_new_tokens: int = 128, chat_template: bool = True, n_gpu_layers: int | None = None, tokenizer=None):
        """tokenizer: optional HF tokenizer; when given, prompts are tokenized in Python so the token
        sequence matches an HF run exactly (used for HF vs llama.cpp agreement checks)."""
        self.max_new_tokens = max_new_tokens
        self.chat_template = chat_template
        self.n_gpu_layers = n_gpu_layers
        self.tokenizer = tokenizer
        self.model_path: Path | None = None
        self.layer_map: dict[int, int] = {}
        self.info: ModelInfo | None = None

    def _cmd(self, input_path: Path, out_path: Path, n_predict: int) -> list[str]:
        cmd = [str(self.binary), "--model", str(self.model_path), "--input", str(input_path), "--out", str(out_path),
               "--n-predict", str(n_predict)]
        if self.chat_template:
            cmd.append("--chat")
        if self.n_gpu_layers is not None:
            cmd += ["--n-gpu-layers", str(self.n_gpu_layers)]
        return cmd

    def _input(self, prompts: list[Prompt]) -> str:
        parts = []
        for p in prompts:
            if self.tokenizer is not None:
                ids = self._hf_ids(p.prompt)
                parts.append(f"TOK {p.id} {len(ids)}\n{' '.join(map(str, ids))}\n")
            else:
                raw = p.prompt.encode()
                parts.append(f"TXT {p.id} {len(raw)}\n")
                parts.append(raw.decode() + "\n")
        return "".join(parts)

    def _hf_ids(self, text: str) -> list[int]:
        tok = self.tokenizer
        if self.chat_template and getattr(tok, "chat_template", None):
            text = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
            return tok(text, add_special_tokens=False).input_ids
        return tok(text).input_ids

    def load(self, model_id: str) -> ModelInfo:
        path = Path(model_id).expanduser()
        if not path.is_file() or path.suffix != ".gguf":
            raise LlamaCppError(f"the llamacpp backend takes a path to a .gguf file, got {model_id!r}")
        self.model_path = path.absolute()  # not resolve(): keep the readable name, not a cache blob hash
        self.binary = build_helper()
        # Probe with one short prompt: reads model info and finds which blocks are MoE layers.
        with tempfile.TemporaryDirectory() as d:
            inp, out = Path(d, "in.txt"), Path(d, "out.bin")
            inp.write_text("TXT 0 5\nHello\n")
            proc = subprocess.run(self._cmd(inp, out, 0), capture_output=True, text=True)
            meta = next((json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")), None)
            if proc.returncode != 0 or meta is None:
                raise LlamaCppError(f"llama.cpp tracer failed on {path.name}: {proc.stderr.strip()[-1000:]}")
            records, _ = read_records(out)
        moe_blocks = sorted({r[3] for r in records})
        if not moe_blocks:
            raise LlamaCppError("the eval callback saw no ffn_moe_logits / ffn_moe_topk tensors; this llama.cpp build or "
                                "architecture names its MoE tensors differently")
        if meta["n_expert_shared"]:
            raise LlamaCppError("models with shared experts are not supported by the llamacpp backend yet")
        self.layer_map = {block: i for i, block in enumerate(moe_blocks)}
        self.info = ModelInfo(
            model_id=str(self.model_path),
            family=meta["arch"],
            num_layers=len(moe_blocks),
            num_experts=meta["n_expert"],
            top_k=meta["n_expert_used"],
            shared_experts=[],
            expert_bytes=meta["expert_bytes"],
            dtype=meta["desc"].split()[-1],
        )
        return self.info

    def run(self, prompts: list[Prompt]) -> Iterator[RouterEvent]:
        categories = {p.id: p.category for p in prompts}
        with tempfile.TemporaryDirectory() as d:
            inp, out = Path(d, "in.txt"), Path(d, "out.bin")
            inp.write_text(self._input(prompts), encoding="utf-8")
            out.touch()
            proc = subprocess.Popen(self._cmd(inp, out, self.max_new_tokens), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            offset = 0
            for line in proc.stdout:
                if not line.startswith("done "):
                    continue
                records, offset = read_records(out, offset)
                yield from self._events(records, categories)
            stderr = proc.stderr.read()
            if proc.wait() != 0:
                raise LlamaCppError(f"llama.cpp tracer failed: {stderr.strip()[-1000:]}")
            records, offset = read_records(out, offset)
            yield from self._events(records, categories)

    def _events(self, records, categories) -> Iterator[RouterEvent]:
        for prompt_id, phase, token_pos, block, ids, weights, lmax in records:
            if block not in self.layer_map:
                raise LlamaCppError(f"router tensors for block {block}, which the probe did not see")
            yield RouterEvent(
                prompt_id=prompt_id,
                category=categories.get(prompt_id, "default"),
                phase="prefill" if phase == 0 else "decode",
                layer=self.layer_map[block],
                token_pos=np.arange(token_pos, token_pos + len(ids), dtype=np.int32),
                expert_ids=ids,
                weights=weights,
                logit_max=lmax,
            )
