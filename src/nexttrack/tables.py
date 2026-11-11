"""Small pyarrow helpers for the prepared parquet folders."""

import json
from pathlib import Path

import numpy as np
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
