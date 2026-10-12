"""Buffer RouterEvents and write them to routes.parquet in fixed-size batches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import ROUTES_FILE, ROUTES_SCHEMA, RouterEvent, RunManifest

BATCH_ROWS = 50_000


class TraceWriter:
    def __init__(self, run_dir: Path | str, batch_rows: int = BATCH_ROWS):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.batch_rows = batch_rows
        self._writer = pq.ParquetWriter(self.run_dir / ROUTES_FILE, ROUTES_SCHEMA, compression="zstd")
        self._pending: list[RouterEvent] = []
        self._pending_rows = 0
        self.rows_written = 0

    def write(self, event: RouterEvent) -> None:
        self._pending.append(event)
        self._pending_rows += len(event)
        if self._pending_rows >= self.batch_rows:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        events = self._pending
        n = self._pending_rows
        k = events[0].expert_ids.shape[1]
        ids = np.concatenate([e.expert_ids for e in events]).astype(np.int16).reshape(-1)
        weights = np.concatenate([e.weights for e in events]).astype(np.float16).reshape(-1)
        offsets = np.arange(0, n * k + 1, k, dtype=np.int32)
        columns = [
            pa.array(np.concatenate([np.full(len(e), e.prompt_id, np.int32) for e in events])),
            pa.DictionaryArray.from_arrays(  # cheap repeated strings, decoded to plain string below
                np.concatenate([np.full(len(e), i, np.int32) for i, e in enumerate(events)]),
                pa.array([e.category for e in events]),
            ).cast(pa.string()),
            pa.DictionaryArray.from_arrays(
                np.concatenate([np.full(len(e), i, np.int32) for i, e in enumerate(events)]),
                pa.array([e.phase for e in events]),
            ).cast(pa.string()),
            pa.array(np.concatenate([e.token_pos for e in events]).astype(np.int32)),
            pa.array(np.concatenate([np.full(len(e), e.layer, np.int16) for e in events])),
            pa.ListArray.from_arrays(offsets, pa.array(ids)),
            pa.ListArray.from_arrays(offsets, pa.array(weights)),
            pa.array(np.concatenate([e.logit_max for e in events]).astype(np.float16)),
        ]
        self._writer.write_table(pa.Table.from_arrays(columns, schema=ROUTES_SCHEMA))
        self.rows_written += n
        self._pending, self._pending_rows = [], 0

    def close(self, manifest: RunManifest | None = None) -> None:
        self.flush()
        self._writer.close()
        if manifest is not None:
            manifest.write(self.run_dir)

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, *exc) -> None:
        if self._writer.is_open:
            self.flush()
            self._writer.close()
