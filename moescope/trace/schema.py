"""The on-disk trace format. Every backend emits exactly this; analysis never knows the engine.

A run is a directory holding `run.json` (RunManifest) and `routes.parquet` (ROUTES_SCHEMA),
one row per (token, MoE layer). Change this file and the spec together, never one alone.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROUTES_FILE = "routes.parquet"
MANIFEST_FILE = "run.json"
PHASES = ("prefill", "decode")

ROUTES_SCHEMA = pa.schema(
    [
        ("prompt_id", pa.int32()),
        ("category", pa.string()),
        ("phase", pa.string()),
        ("token_pos", pa.int32()),
        ("layer", pa.int16()),
        ("expert_ids", pa.list_(pa.int16())),
        ("weights", pa.list_(pa.float16())),
        ("logit_max", pa.float16()),
    ]
)


class SchemaError(ValueError):
    pass


@dataclass
class RunManifest:
    moescope_version: str
    backend: str
    model_id: str
    num_layers: int
    num_experts: int
    top_k: int
    shared_experts: list[int]
    expert_bytes: int
    dtype: str
    prompts_file: str
    max_new_tokens: int
    created_at: str

    def write(self, run_dir: Path) -> None:
        (Path(run_dir) / MANIFEST_FILE).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def read(cls, run_dir: Path) -> RunManifest:
        path = Path(run_dir) / MANIFEST_FILE
        if not path.exists():
            raise FileNotFoundError(f"{run_dir} is not a moescope run (no {MANIFEST_FILE})")
        return cls(**json.loads(path.read_text()))


@dataclass
class RouterEvent:
    """Routing for a contiguous span of tokens in one MoE layer, as numpy arrays.

    Backends yield these; the writer flattens them into one row per token.
    """

    prompt_id: int
    category: str
    phase: str
    layer: int
    token_pos: np.ndarray  # (n,) int
    expert_ids: np.ndarray  # (n, k) int, highest weight first
    weights: np.ndarray  # (n, k) float, softmax over the chosen k
    logit_max: np.ndarray  # (n,) float
    extra: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.token_pos)


def validate(source: Path | str | pa.Table, top_k: int | None = None, num_experts: int | None = None) -> pa.Table:
    """Check a routes table (or a run dir / parquet path) against the schema. Returns the table.

    When given a run directory, top_k and num_experts default to the manifest's values.
    """
    if isinstance(source, pa.Table):
        table = source
    else:
        path = Path(source)
        if path.is_dir():
            manifest = RunManifest.read(path)
            top_k = top_k if top_k is not None else manifest.top_k
            num_experts = num_experts if num_experts is not None else manifest.num_experts
            path = path / ROUTES_FILE
        table = pq.read_table(path)

    for expected in ROUTES_SCHEMA:
        if expected.name not in table.schema.names:
            raise SchemaError(f"missing column {expected.name!r}")
        actual = table.schema.field(expected.name).type
        if not actual.equals(expected.type):
            raise SchemaError(f"column {expected.name!r} has type {actual}, expected {expected.type}")
    extra = set(table.schema.names) - set(ROUTES_SCHEMA.names)
    if extra:
        raise SchemaError(f"unexpected columns {sorted(extra)}")

    for name in ROUTES_SCHEMA.names:
        if table.column(name).null_count:
            raise SchemaError(f"column {name!r} contains nulls")

    phases = set(table.column("phase").unique().to_pylist())
    if not phases <= set(PHASES):
        raise SchemaError(f"phase values {sorted(phases - set(PHASES))} not in {PHASES}")

    ids_len = np.asarray(table.column("expert_ids").combine_chunks().value_lengths())
    w_len = np.asarray(table.column("weights").combine_chunks().value_lengths())
    if not np.array_equal(ids_len, w_len):
        raise SchemaError("expert_ids and weights have different lengths in some rows")
    if len(ids_len) and top_k is not None and not (ids_len == top_k).all():
        raise SchemaError(f"expert_ids rows must have length top_k={top_k}")
    if len(ids_len) and ids_len.min() != ids_len.max():
        raise SchemaError("expert_ids rows have inconsistent lengths")

    if num_experts is not None and table.num_rows:
        flat = np.asarray(table.column("expert_ids").combine_chunks().flatten())
        if flat.min() < 0 or flat.max() >= num_experts:
            raise SchemaError(f"expert ids outside [0, {num_experts})")
    return table
