"""Prompt and eval files: JSONL, one object per line."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Prompt:
    id: int
    prompt: str
    category: str = "default"
    answer: str | None = None  # eval files only


def load_prompts(path: Path | str) -> list[Prompt]:
    prompts = []
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{lineno}: invalid JSON ({e.msg})") from None
        if "prompt" not in obj:
            raise ValueError(f"{path}:{lineno}: missing 'prompt'")
        prompts.append(
            Prompt(
                id=int(obj.get("id", len(prompts))),
                prompt=obj["prompt"],
                category=obj.get("category") or "default",
                answer=obj.get("answer"),
            )
        )
    if not prompts:
        raise ValueError(f"{path}: no prompts")
    return prompts
