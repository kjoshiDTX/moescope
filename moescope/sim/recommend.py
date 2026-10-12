"""Turn simulation results into engine settings.

Engine flags change between llama.cpp releases, so nothing here is assumed: the binary's
--help output is read at runtime and only flags it actually lists are recommended.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field

from ..trace.schema import RunManifest

# Flags moescope knows how to fill in, if (and only if) the installed binary lists them.
KNOWN_FLAGS = ("--n-cpu-moe", "--moe-cache-mib", "--cpu-moe")


@dataclass
class LlamaCpp:
    binary: str | None
    version: str | None = None
    flags: set[str] = field(default_factory=set)
    error: str | None = None


def detect_llamacpp(binary: str | None = None) -> LlamaCpp:
    path = binary or shutil.which("llama-server") or shutil.which("llama-cli")
    if not path:
        return LlamaCpp(None, error="llama.cpp not found on PATH (looked for llama-server, llama-cli); pass --llama-bin")
    try:
        help_text = subprocess.run([path, "--help"], capture_output=True, text=True, timeout=20)
        version = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as e:
        return LlamaCpp(path, error=f"could not run {path} --help: {e}")
    text = help_text.stdout + help_text.stderr
    found = {f for f in KNOWN_FLAGS if re.search(rf"(^|[\s,]){re.escape(f)}(?=[\s,=]|$)", text, re.M)}
    vtext = (version.stdout + version.stderr).strip()
    match = re.search(r"version:?\s*(\S+(?:\s*\([0-9a-f]+\))?)", vtext)
    return LlamaCpp(path, version=match.group(1) if match else (vtext.splitlines() or [None])[0], flags=found)


def choose_budget(results: list[dict], target_hit_rate: float) -> dict:
    """Smallest budget whose best policy reaches the target; otherwise the best result overall."""
    best_per_budget: dict[float, dict] = {}
    for r in results:
        cur = best_per_budget.get(r["budget_gb"])
        if cur is None or r["hit_rate"] > cur["hit_rate"]:
            best_per_budget[r["budget_gb"]] = r
    for budget in sorted(best_per_budget):
        if best_per_budget[budget]["hit_rate"] >= target_hit_rate:
            return {**best_per_budget[budget], "meets_target": True}
    best = max(best_per_budget.values(), key=lambda r: (r["hit_rate"], -r["budget_gb"]))
    return {**best, "meets_target": False}


def recommend(manifest: RunManifest, results: list[dict], llama: LlamaCpp, target_hit_rate: float = 0.95) -> dict:
    pick = choose_budget(results, target_hit_rate)
    slots = pick["slots"]
    routed_per_layer = manifest.num_experts - len(manifest.shared_experts)
    gpu_layers = min(manifest.num_layers, slots // max(routed_per_layer, 1))
    settings: list[dict] = []
    if "--moe-cache-mib" in llama.flags:
        settings.append({"flag": "--moe-cache-mib", "value": int(pick["budget_gb"] * 1024),
                         "why": "expert cache sized to the chosen budget"})
    if "--n-cpu-moe" in llama.flags:
        settings.append({"flag": "--n-cpu-moe", "value": manifest.num_layers - gpu_layers,
                         "why": f"the budget holds every expert of {gpu_layers} of {manifest.num_layers} MoE layers on the GPU"})
    elif "--cpu-moe" in llama.flags and gpu_layers == 0:
        settings.append({"flag": "--cpu-moe", "value": None, "why": "the budget cannot hold one full MoE layer"})
    return {
        "target_hit_rate": target_hit_rate,
        "choice": pick,
        "llamacpp": {"binary": llama.binary, "version": llama.version, "flags_found": sorted(llama.flags), "error": llama.error},
        "settings": settings,
    }
