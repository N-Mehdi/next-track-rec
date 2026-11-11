"""Item index sequences for SASRec: training windows and inference inputs.

Sequences hold item_idx values (1..N). Index 0 marks padding and items unseen in
training; those are dropped from the inputs. Windows are left padded so that the latest
event is always at the last position, during training and inference alike.
"""

import numpy as np

from nexttrack.tables import list_column, read_table


def read_histories(path):
    """uid array and list of item_idx arrays without the 0 entries."""
    table = read_table(path, ["uid", "item_idxs"])
    uids = table.column("uid").to_numpy()
    rows = list_column(table, "item_idxs")
    histories = []
    for row in rows:
        histories.append(row[row > 0].astype(np.int64))
    return uids, histories


def read_index_map(path):
    """item_idx to item_id lookup array, -1 at the padding position."""
    table = read_table(path, ["item_id", "item_idx"])
    item_ids = table.column("item_id").to_numpy().astype(np.int64)
    item_idxs = table.column("item_idx").to_numpy().astype(np.int64)
    lookup = np.full(item_idxs.max() + 1, -1, dtype=np.int64)
    lookup[item_idxs] = item_ids
    return lookup


def make_windows(histories, max_len, stride):
    """Windows of max_len + 1 events taken from the end of each history every stride events."""
    size = max_len + 1
    pieces = []
    for sequence in histories:
        if len(sequence) < 2:
            continue
        end = len(sequence)
        while True:
            start = max(0, end - size)
            pieces.append(sequence[start:end])
            if start == 0:
                break
            end -= stride
    windows = np.zeros((len(pieces), size), dtype=np.int64)
    for i in range(len(pieces)):
        windows[i, size - len(pieces[i]) :] = pieces[i]
    return windows


def last_inputs(histories, max_len):
    """Last max_len events of each history, left padded."""
    inputs = np.zeros((len(histories), max_len), dtype=np.int64)
    for i in range(len(histories)):
        tail = histories[i][-max_len:]
        if len(tail) > 0:
            inputs[i, max_len - len(tail) :] = tail
    return inputs


def align(uids, histories, wanted_uids):
    """Histories reordered to follow wanted_uids; missing users get an empty history."""
    position = {}
    for i in range(len(uids)):
        position[int(uids[i])] = i
    aligned = []
    for uid in wanted_uids:
        row = position.get(int(uid))
        if row is None:
            aligned.append(np.zeros(0, dtype=np.int64))
        else:
            aligned.append(histories[row])
    return aligned
