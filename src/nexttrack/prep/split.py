"""Spark preparation: Listen+ filter, temporal split, item index, histories, targets.

Outputs (parquet folders under paths.prepared):
  item_map        item_id, item_idx (1..N, 0 is padding), train_count
  train_events    Listen+ events before TRAIN_END, with item_idx
  val_events      Listen+ events of the validation day, item_idx 0 when unseen in train
  test_events     Listen+ events of the test day, item_idx 0 when unseen in train
  history_val     per user arrays of events before TRAIN_END
  history_test    per user arrays of events before VAL_END (state frozen before the test day)
  val_targets     per user arrays of validation day events
  test_targets    per user arrays of test day events
  likes, dislikes, unlikes, undislikes   raw side events of the kept users
  catalogue       item_id, item_idx, embedding (float32[128]) for every item with audio
"""

import json
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

from nexttrack import constants
from nexttrack.config import parse_args

SIDE_EVENTS = ["likes", "dislikes", "unlikes", "undislikes"]


def build_spark(config):
    spark_conf = config["spark"]
    builder = SparkSession.builder.master(spark_conf["master"]).appName("nexttrack-prep")
    builder = builder.config("spark.driver.memory", spark_conf["driver_memory"])
    builder = builder.config("spark.sql.shuffle.partitions", spark_conf["shuffle_partitions"])
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def write(df, path, partitions):
    df.coalesce(partitions).write.mode("overwrite").parquet(str(path))


def pick_users(listens, max_users):
    """Smallest uids, used for smoke tests. None keeps every user."""
    if max_users is None:
        return None
    return listens.select("uid").distinct().orderBy("uid").limit(int(max_users))


def keep_users(df, users):
    if users is None:
        return df
    return df.join(F.broadcast(users), "uid")


def sequences(df):
    """One row per user with chronological arrays."""
    event = F.struct("timestamp", "item_idx", "item_id", "is_organic")
    grouped = df.groupBy("uid").agg(F.sort_array(F.collect_list(event)).alias("events"))
    return grouped.select(
        "uid",
        F.col("events.item_id").alias("item_ids"),
        F.col("events.item_idx").alias("item_idxs"),
        F.col("events.timestamp").alias("timestamps"),
        F.col("events.is_organic").alias("is_organic"),
    )


def prepare(spark, raw_dir, out_dir, max_users=None):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = F.col("timestamp")

    listens = spark.read.parquet(str(raw_dir / "flat" / "50m" / "listens.parquet"))
    users = pick_users(listens, max_users)
    listens = keep_users(listens, users)
    events = listens.filter(F.col("played_ratio_pct") >= constants.LISTEN_THRESHOLD_PCT)
    events = events.select("uid", "item_id", "timestamp", "is_organic")

    # Item index from training interactions only.
    train = events.filter(ts < constants.TRAIN_END)
    item_map = train.groupBy("item_id").agg(F.count("*").alias("train_count"))
    item_map = item_map.withColumn("item_idx", F.row_number().over(Window.orderBy("item_id")))
    write(item_map.select("item_id", "item_idx", "train_count"), out_dir / "item_map", 1)
    item_map = spark.read.parquet(str(out_dir / "item_map"))
    index = F.broadcast(item_map.select("item_id", "item_idx"))
    train_users = train.select("uid").distinct()

    def with_index(df):
        return df.join(index, "item_id", "left").fillna({"item_idx": 0})

    train = with_index(train)
    in_val = (ts >= constants.VAL_START) & (ts < constants.VAL_END)
    val = with_index(events.filter(in_val).join(train_users, "uid"))
    test = with_index(events.filter(ts >= constants.TEST_START).join(train_users, "uid"))
    before_test = with_index(events.filter(ts < constants.VAL_END).join(train_users, "uid"))

    write(train, out_dir / "train_events", 8)
    write(val, out_dir / "val_events", 1)
    write(test, out_dir / "test_events", 1)
    write(sequences(train), out_dir / "history_val", 4)
    write(sequences(before_test), out_dir / "history_test", 4)
    write(sequences(val), out_dir / "val_targets", 1)
    write(sequences(test), out_dir / "test_targets", 1)

    for name in SIDE_EVENTS:
        side = spark.read.parquet(str(raw_dir / "flat" / "50m" / (name + ".parquet")))
        write(keep_users(side, users), out_dir / name, 1)

    # Audio catalogue: every item listened or liked by the kept users.
    likes = spark.read.parquet(str(out_dir / "likes"))
    pool = listens.select("item_id").union(likes.select("item_id")).distinct()
    embeddings = spark.read.parquet(str(raw_dir / constants.EMBEDDINGS_FILE))
    to_float = F.transform("normalized_embed", lambda x: x.cast("float"))
    embeddings = embeddings.select("item_id", to_float.alias("embedding"))
    catalogue = with_index(embeddings.join(F.broadcast(pool), "item_id"))
    write(catalogue.select("item_id", "item_idx", "embedding"), out_dir / "catalogue", 4)

    stats = collect_stats(spark, out_dir, pool.count())
    with open(out_dir / "stats.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    return stats


def collect_stats(spark, out_dir, pool_size):
    stats = {"item_pool": pool_size}
    for name in ["item_map", "train_events", "val_events", "test_events", "catalogue"]:
        stats[name + "_rows"] = spark.read.parquet(str(out_dir / name)).count()
    for name in ["history_val", "history_test", "val_targets", "test_targets"]:
        stats[name + "_users"] = spark.read.parquet(str(out_dir / name)).count()
    for name in ["val_events", "test_events"]:
        df = spark.read.parquet(str(out_dir / name))
        stats[name + "_cold_rows"] = df.filter(F.col("item_idx") == 0).count()
        stats[name + "_cold_items"] = df.filter(F.col("item_idx") == 0).select("item_id").distinct().count()
    stats["items_without_embedding"] = pool_size - stats["catalogue_rows"]
    return stats


def main():
    config, _ = parse_args("Prepare Yambda-50M with Spark")
    spark = build_spark(config)
    stats = prepare(spark, config["paths"]["raw"], config["paths"]["prepared"], config["data"]["max_users"])
    print(json.dumps(stats, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
