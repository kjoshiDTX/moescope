"""Pick the adapter for a model config, or fail with the list of supported families."""

from __future__ import annotations

from typing import Any

from .base import Adapter
from .granite_moe import GraniteMoeAdapter
from .mixtral import MixtralAdapter
from .olmoe import OlmoeAdapter
from .qwen_moe import QwenMoeAdapter

ADAPTERS: list[type[Adapter]] = [GraniteMoeAdapter, QwenMoeAdapter, OlmoeAdapter, MixtralAdapter]


class UnsupportedModelError(RuntimeError):
    pass


def supported_families() -> str:
    return ", ".join(f"{a.family} ({'/'.join(a.model_types)})" for a in ADAPTERS)


def adapter_for(config: Any) -> Adapter:
    for cls in ADAPTERS:
        if cls.matches(config):
            return cls(config)
    model_type = getattr(config, "model_type", "unknown")
    raise UnsupportedModelError(
        f"No moescope adapter for model_type {model_type!r}. Supported families: {supported_families()}."
    )
