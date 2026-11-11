"""Small pyarrow helpers for the prepared parquet folders."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def read_table(path, columns=None):
    return pq.read_table(str(path), columns=columns)


def list_column(table, name):
    """Return a list of numpy arrays, one per row, from a list column."""
    array = table.column(name).combine_chunks()
    offsets = array.offsets.to_numpy()
    values = array.values.to_numpy()
    rows = []
    for i in range(len(offsets) - 1):
        rows.append(values[offsets[i] : offsets[i + 1]])
    return rows


def flat_column(table, name):
    """Concatenate a list column into one numpy array."""
    return table.column(name).combine_chunks().values.to_numpy()


def read_targets(path):
    table = read_table(path, ["uid", "item_ids"])
    return table.column("uid").to_numpy(), list_column(table, "item_ids")


def read_item_ids(path):
    return read_table(path, ["item_id"]).column("item_id").to_numpy()


def seen_item_sets(prepared):
    """Items seen under both cold-start definitions.

    cold_train: Listen+ items of the training period (item_map).
    cold_strict: Listen+ items of every history before the test day (history_test, which
    ends at VAL_END and covers training, first gap and validation day).
    """
    prepared = Path(prepared)
    strict = np.unique(flat_column(read_table(prepared / "history_test", ["item_ids"]), "item_ids"))
    return {"cold_train": read_item_ids(prepared / "item_map"), "cold_strict": strict}


def write_ranked(path, uids, item_ids, scores):
    """Parquet with one row per user: uid, item_ids list, scores list (same order)."""
    rows, width = item_ids.shape
    offsets = pa.array(np.arange(0, rows * width + 1, width, dtype=np.int32))
    table = pa.table(
        {
            "uid": pa.array(np.asarray(uids, dtype=np.uint32)),
            "item_ids": pa.ListArray.from_arrays(offsets, pa.array(item_ids.ravel().astype(np.int64))),
            "scores": pa.ListArray.from_arrays(offsets, pa.array(scores.ravel().astype(np.float32))),
        }
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))


def write_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)


def to_builtin(value):
    """Convert numpy scalars nested in dicts and lists for json."""
    if isinstance(value, dict):
        result = {}
        for key in value:
            result[key] = to_builtin(value[key])
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            result.append(to_builtin(item))
        return result
    if isinstance(value, np.generic):
        return value.item()
    return value
