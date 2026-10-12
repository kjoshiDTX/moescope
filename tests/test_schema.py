import numpy as np
import pyarrow as pa
import pytest

from moescope.trace.reader import load_run
from moescope.trace.schema import ROUTES_SCHEMA, RouterEvent, RunManifest, SchemaError, validate
from moescope.trace.writer import TraceWriter


def manifest(**kw):
    base = dict(moescope_version="0.1.0", backend="test", model_id="toy", num_layers=2, num_experts=4, top_k=2,
                shared_experts=[], expert_bytes=1000, dtype="float32", prompts_file="x.jsonl", max_new_tokens=4,
                created_at="2026-10-12T18:00:00-05:00")
    return RunManifest(**{**base, **kw})


def event(layer=0, n=3, k=2, phase="prefill", start=0):
    rng = np.random.default_rng(layer)
    return RouterEvent(prompt_id=0, category="code", phase=phase, layer=layer,
                       token_pos=np.arange(start, start + n), expert_ids=rng.integers(0, 4, (n, k)),
                       weights=np.full((n, k), 1 / k), logit_max=rng.normal(size=n))


def good_table():
    return pa.table({
        "prompt_id": pa.array([0], pa.int32()), "category": ["code"], "phase": ["prefill"],
        "token_pos": pa.array([0], pa.int32()), "layer": pa.array([0], pa.int16()),
        "expert_ids": pa.array([[1, 2]], pa.list_(pa.int16())),
        "weights": pa.array([[0.5, 0.5]], pa.list_(pa.float16())),
        "logit_max": pa.array(np.array([1.0], np.float16)),
    }, schema=ROUTES_SCHEMA)


def test_writer_roundtrip_and_validate(tmp_path):
    with TraceWriter(tmp_path, batch_rows=4) as w:  # small batches force several flushes
        for layer in range(2):
            w.write(event(layer))
            w.write(event(layer, n=1, phase="decode", start=3))
        w.close(manifest())
    table = validate(tmp_path)
    assert table.num_rows == 8
    run = load_run(tmp_path)
    assert run.manifest.top_k == 2
    assert run.routes.height == 8
    assert sorted(run.routes["phase"].unique().to_list()) == ["decode", "prefill"]


def test_good_table_passes():
    validate(good_table(), top_k=2, num_experts=4)


@pytest.mark.parametrize("column,bad", [
    ("layer", pa.array([0], pa.int32())),
    ("weights", pa.array([[0.5, 0.5]], pa.list_(pa.float32()))),
    ("expert_ids", pa.array([[1, 2]], pa.list_(pa.int64()))),
])
def test_wrong_dtype_fails(column, bad):
    t = good_table()
    t = t.set_column(t.schema.get_field_index(column), column, bad)
    with pytest.raises(SchemaError, match=column):
        validate(t)


def test_wrong_list_length_fails():
    t = good_table()
    with pytest.raises(SchemaError, match="top_k"):
        validate(t, top_k=3)
    t = t.set_column(t.schema.get_field_index("weights"), "weights", pa.array([[1.0]], pa.list_(pa.float16())))
    with pytest.raises(SchemaError, match="different lengths"):
        validate(t)


def test_bad_phase_and_expert_range_fail():
    t = good_table()
    with pytest.raises(SchemaError, match="expert ids"):
        validate(t, num_experts=2)
    t = t.set_column(t.schema.get_field_index("phase"), "phase", pa.array(["train"]))
    with pytest.raises(SchemaError, match="phase"):
        validate(t)
