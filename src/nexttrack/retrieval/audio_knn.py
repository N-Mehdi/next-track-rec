"""Audio nearest neighbours for tracks never listened to during training, without training.

The query of a user is the mean of the normalized audio embeddings of the last listened
tracks. The search pool is every track of embeddings.parquet absent from item_map: the
tracks that SASRec cannot score. The pool does not depend on the test day, so it holds no
information about the targets. The number of averaged tracks is selected on the validation
day by ndcg@100 on its never-seen targets.
"""

from pathlib import Path

import faiss
import numpy as np
import pyarrow.parquet as pq

from nexttrack import constants
from nexttrack.config import parse_args
from nexttrack.eval.metrics import evaluate
from nexttrack.tables import (
    list_column,
    read_item_ids,
    read_table,
    seen_item_sets,
    to_builtin,
    write_json,
    write_ranked,
)

HISTORY_GRID = [1, 5, 10, 20, 50, 200]
EMBEDDING_DIM = 128
BATCH_ROWS = 200_000


def load_audio(raw_dir, cache_dir):
    """Item ids (sorted) and float32 normalized embeddings, cached as .npy files."""
    cache_dir = Path(cache_dir)
    ids_path = cache_dir / "item_ids.npy"
    vectors_path = cache_dir / "vectors.npy"
    if ids_path.exists() and vectors_path.exists():
        return np.load(ids_path), np.load(vectors_path, mmap_mode="r")
    source = pq.ParquetFile(str(Path(raw_dir) / constants.EMBEDDINGS_FILE))
    item_ids = source.read(columns=["item_id"]).column("item_id").to_numpy()
    order = np.argsort(item_ids, kind="stable")
    # Row of each file row in the sorted output.
    position = np.empty(len(order), dtype=np.int64)
    position[order] = np.arange(len(order))
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Written on disk batch by batch: the float32 matrix is about 4 GB.
    vectors = np.lib.format.open_memmap(
        vectors_path, mode="w+", dtype=np.float32, shape=(len(item_ids), EMBEDDING_DIM)
    )
    start = 0
    for batch in source.iter_batches(batch_size=BATCH_ROWS, columns=["normalized_embed"]):
        values = batch.column(0).flatten().to_numpy().reshape(-1, EMBEDDING_DIM)
        vectors[position[start : start + len(values)]] = values
        start += len(values)
    vectors.flush()
    del vectors
    np.save(ids_path, item_ids[order])
    return np.load(ids_path), np.load(vectors_path, mmap_mode="r")


def lookup(item_ids, wanted):
    """Row of each wanted id in the sorted item_ids, -1 when it has no embedding."""
    rows = np.searchsorted(item_ids, wanted)
    rows = np.minimum(rows, len(item_ids) - 1)
    found = item_ids[rows] == wanted
    return np.where(found, rows, -1)


def user_queries(histories, item_ids, vectors, num_items):
    """Mean embedding of the last num_items tracks with audio; zero when none."""
    queries = np.zeros((len(histories), vectors.shape[1]), dtype=np.float32)
    for i in range(len(histories)):
        rows = lookup(item_ids, histories[i].astype(item_ids.dtype))
        rows = rows[rows >= 0][-num_items:]
        if len(rows) > 0:
            mean = np.asarray(vectors[np.sort(rows)]).mean(axis=0)
            norm = np.linalg.norm(mean)
            if norm > 0:
                queries[i] = mean / norm
    return queries


def build_index(item_ids, vectors, excluded):
    """Inner product index over the tracks absent from excluded."""
    keep = np.flatnonzero(~np.isin(item_ids, excluded))
    index = faiss.IndexFlatIP(vectors.shape[1])
    step = 1_000_000
    for start in range(0, len(keep), step):
        index.add(np.ascontiguousarray(vectors[keep[start : start + step]]))
    return index, item_ids[keep].astype(np.int64)


def search(index, pool_ids, queries, k):
    scores, rows = index.search(queries, k)
    ranked = np.where(rows >= 0, pool_ids[np.maximum(rows, 0)], -1)
    empty = ~queries.any(axis=1)
    ranked[empty] = -1
    return ranked, scores


def load_split(prepared, split):
    """Target uids, target item ids and the matching histories."""
    history = read_table(prepared / ("history_" + split), ["uid", "item_ids"])
    targets = read_table(prepared / (split + "_targets"), ["uid", "item_ids"])
    position = {}
    history_uids = history.column("uid").to_numpy()
    for i in range(len(history_uids)):
        position[int(history_uids[i])] = i
    items = list_column(history, "item_ids")
    uids = targets.column("uid").to_numpy()
    histories = []
    for uid in uids:
        histories.append(items[position[int(uid)]])
    return uids, list_column(targets, "item_ids"), histories


def main():
    config, _ = parse_args("Audio nearest neighbour retrieval for never-seen tracks")
    paths = config["paths"]
    prepared = Path(paths["prepared"])
    k_values = config["eval"]["k_values"]
    item_ids, vectors = load_audio(paths["raw"], paths["audio"])
    index, pool_ids = build_index(item_ids, vectors, read_item_ids(prepared / "item_map"))
    print("pool", len(pool_ids), flush=True)
    seen = seen_item_sets(prepared)
    # The validation history ends at TRAIN_END: both definitions coincide there.
    val_seen = {"cold_train": seen["cold_train"]}

    val_uids, val_targets, val_histories = load_split(prepared, "val")
    validation = {}
    best = None
    best_value = -1.0
    for num_items in HISTORY_GRID:
        queries = user_queries(val_histories, item_ids, vectors, num_items)
        ranked, _ = search(index, pool_ids, queries, constants.NUM_RANKED)
        result = evaluate(ranked, val_targets, k_values, len(pool_ids), val_seen)
        validation[str(num_items)] = result
        value = result["cold_train"]["metrics"]["ndcg@100"]
        print("val last", num_items, "cold ndcg@100=%.4f" % value, flush=True)
        if value > best_value:
            best_value = value
            best = num_items

    outputs = Path(paths["outputs"]) / "audio_knn"
    queries = user_queries(val_histories, item_ids, vectors, best)
    ranked, scores = search(index, pool_ids, queries, constants.NUM_RANKED)
    write_ranked(outputs / "candidates_val.parquet", val_uids, ranked, scores)
    test_uids, test_targets, test_histories = load_split(prepared, "test")
    queries = user_queries(test_histories, item_ids, vectors, best)
    ranked, scores = search(index, pool_ids, queries, constants.NUM_RANKED)
    write_ranked(outputs / "candidates_test.parquet", test_uids, ranked, scores)
    test = evaluate(ranked, test_targets, k_values, len(pool_ids), seen)
    for name in ["cold_train", "cold_strict"]:
        print("test", name, test[name]["metrics"]["recall@100"], test[name]["ci95"]["recall@100"])

    report = {"pool_size": len(pool_ids), "best_history_items": best, "validation": validation, "test": test}
    write_json(to_builtin(report), Path(paths["reports"]) / "audio_knn.json")


if __name__ == "__main__":
    main()
