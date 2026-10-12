"""The interface every capture backend implements."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from ..prompts import Prompt
from ..trace.schema import RouterEvent


@dataclass
class ModelInfo:
    model_id: str
    family: str
    num_layers: int  # MoE layers only
    num_experts: int
    top_k: int
    shared_experts: list[int]
    expert_bytes: int
    dtype: str


class Backend(Protocol):
    name: str

    def load(self, model_id: str) -> ModelInfo: ...

    def run(self, prompts: list[Prompt]) -> Iterator[RouterEvent]: ...
