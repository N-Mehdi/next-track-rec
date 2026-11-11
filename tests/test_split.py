import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nexttrack import constants
from nexttrack.prep.split import prepare
from nexttrack.tables import list_column, read_table, seen_item_sets

T = constants
GAP_TS = T.TRAIN_END + 100
VAL_TS = T.VAL_START + 100
TEST_TS = T.TEST_START + 100

# uid, timestamp, item_id, played_ratio_pct
LISTENS = [
    (1, 10, 1, 100),
    (1, 20, 2, 80),
    (1, 30, 3, 30),
    (1, GAP_TS, 2, 100),
    (1, VAL_TS, 4, 100),
    (1, TEST_TS, 1, 100),
    (1, TEST_TS + 5, 5, 100),
    (2, TEST_TS, 1, 100),
    (3, 40, 2, 60),
    (3, VAL_TS, 2, 100),
]


def write_raw(raw_dir):
    flat = raw_dir / "flat" / "50m"
    flat.mkdir(parents=True)
    columns = {"uid": [], "timestamp": [], "item_id": [], "played_ratio_pct": []}
    for uid, ts, item, ratio in LISTENS:
        columns["uid"].append(uid)
        columns["timestamp"].append(ts)
        columns["item_id"].append(item)
        columns["played_ratio_pct"].append(ratio)
    n = len(LISTENS)
    table = pa.table(
        {
            "uid": pa.array(columns["uid"], pa.uint32()),
            "timestamp": pa.array(columns["timestamp"], pa.uint32()),
            "item_id": pa.array(columns["item_id"], pa.uint32()),
            "is_organic": pa.array([1] * n, pa.uint8()),
            "played_ratio_pct": pa.array(columns["played_ratio_pct"], pa.uint16()),
            "track_length_seconds": pa.array([200] * n, pa.uint32()),
        }
    )
    pq.write_table(table, flat / "listens.parquet")
    side = pa.table(
        {
            "uid": pa.array([1, 2], pa.uint32()),
            "item_id": pa.array([6, 1], pa.uint32()),
            "timestamp": pa.array([15, 15], pa.uint32()),
            "is_organic": pa.array([1, 0], pa.uint8()),
        }
    )
    for name in ["likes", "dislikes", "unlikes", "undislikes"]:
        pq.write_table(side, flat / (name + ".parquet"))
    vectors = []
    for item in [1, 2, 3, 4, 6, 99]:
        vectors.append([float(item)] * 4)
    embeddings = pa.table(
        {
            "item_id": pa.array([1, 2, 3, 4, 6, 99], pa.uint32()),
            "embed": pa.array(vectors, pa.large_list(pa.float64())),
            "normalized_embed": pa.array(vectors, pa.large_list(pa.float64())),
        }
    )
    pq.write_table(embeddings, raw_dir / "embeddings.parquet")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    pyspark = pytest.importorskip("pyspark")
    raw_dir = tmp_path_factory.mktemp("raw")
    out_dir = tmp_path_factory.mktemp("prepared")
    write_raw(raw_dir)
    spark = pyspark.sql.SparkSession.builder.master("local[1]").appName("test").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    prepare(spark, raw_dir, out_dir)
    spark.stop()
    return out_dir


def test_item_map_uses_train_only(prepared):
    table = read_table(prepared / "item_map").to_pydict()
    assert sorted(table["item_id"]) == [1, 2]
    assert sorted(table["item_idx"]) == [1, 2]


def test_targets_keep_train_users_and_cold_items(prepared):
    val = read_table(prepared / "val_targets")
    test = read_table(prepared / "test_targets")
    assert sorted(val.column("uid").to_pylist()) == [1, 3]
    assert test.column("uid").to_pylist() == [1]
    assert list_column(test, "item_ids")[0].tolist() == [1, 5]
    assert list_column(test, "item_idxs")[0].tolist() == [1, 0]
    assert list_column(val, "item_idxs")[0].tolist() == [0]


def test_histories(prepared):
    history_val = read_table(prepared / "history_val").to_pydict()
    history_test = read_table(prepared / "history_test").to_pydict()
    by_uid = {}
    for i in range(len(history_val["uid"])):
        by_uid[history_val["uid"][i]] = history_val["item_ids"][i]
    assert by_uid[1] == [1, 2]
    for i in range(len(history_test["uid"])):
        if history_test["uid"][i] == 1:
            assert history_test["item_ids"][i] == [1, 2, 2, 4]
            assert history_test["item_idxs"][i] == [1, 2, 2, 0]


def test_catalogue_and_stats(prepared):
    table = read_table(prepared / "catalogue")
    assert table.schema.field("embedding").type.value_type == pa.float32()
    catalogue = table.to_pydict()
    assert sorted(catalogue["item_id"]) == [1, 2, 3, 4, 6]
    assert len(catalogue["embedding"][0]) == 4
    assert np.asarray(list_column(table, "embedding")[0]).dtype == np.float32
    with open(prepared / "stats.json", encoding="utf-8") as handle:
        stats = json.load(handle)
    assert stats["item_pool"] == 6
    assert stats["items_without_embedding"] == 1
    assert stats["test_events_cold_rows"] == 1


def test_seen_item_sets(prepared):
    seen = seen_item_sets(prepared)
    assert sorted(seen["cold_train"].tolist()) == [1, 2]
    # Track 4 is first listened during the validation day: cold for training only.
    assert sorted(seen["cold_strict"].tolist()) == [1, 2, 4]
