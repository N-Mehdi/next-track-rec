"""Candidate union and reranker features for one day (validation or test).

Candidates per user: SASRec, own history, audio kNN and DecayPop, deduplicated in that
order and filled up to the total with the next SASRec items. Every feature only uses
events before the cutoff of the day: TRAIN_END for validation, VAL_END for test.

The production variant has no audio: candidates.audio = 0 and rerank.audio_features false
remove the audio source and the two audio features, so the service needs no audio file.
"""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from nexttrack import constants
from nexttrack.data.sequences import align, last_inputs, read_histories
from nexttrack.eval.baselines import decayed_scores, top_items
from nexttrack.retrieval.audio_knn import load_audio, lookup, user_queries
from nexttrack.tables import list_column, read_table

FEATURES = [
    "sasrec_score",
    "history_score",
    "user_item_count",
    "user_item_count_recent",
    "hours_since_last",
    "events_since_last",
    "audio_score",
    "audio_last_score",
    "pop_score",
    "item_count",
    "item_like_count",
    "user_liked",
    "user_disliked",
    "user_artist_count",
    "user_artist_share",
    "is_cold",
    "history_length",
]
AUDIO_FEATURES = ["audio_score", "audio_last_score"]
# Name of each candidate list, in the order of the union.
SOURCES = ["sasrec", "history", "audio", "decay_pop", "sasrec"]
RECENT_EVENTS = 200


def feature_names(config):
    """Features of the variant: all of them, or all but the audio ones."""
    names = []
    for name in FEATURES:
        if config["rerank"]["audio_features"] or name not in AUDIO_FEATURES:
            names.append(name)
    return names


def uses_audio(config):
    return config["candidates"]["audio"] > 0 or config["rerank"]["audio_features"]


def merge_sources(lists, total):
    """Union of the lists in order, first occurrence kept, -1 dropped, at most total items."""
    joined = np.concatenate(lists)
    joined = joined[joined >= 0]
    _, first = np.unique(joined, return_index=True)
    return joined[np.sort(first)][:total]


def find(keys, values):
    """Row of each value in the sorted keys, -1 when absent."""
    if len(keys) == 0:
        return np.full(len(values), -1, dtype=np.int64)
    return lookup(keys, np.asarray(values, dtype=keys.dtype))


def read_ranked(path):
    table = read_table(path, ["uid", "item_ids", "scores"])
    return table.column("uid").to_numpy(), list_column(table, "item_ids")


def net_state(prepared, positive, negative, cutoff):
    """Sorted uid-item keys whose latest positive or negative event before cutoff is positive."""
    keys = []
    signs = []
    times = []
    for name, sign in [(positive, 1), (negative, -1)]:
        table = read_table(prepared / name, ["uid", "item_id", "timestamp"])
        stamp = table.column("timestamp").to_numpy()
        kept = stamp < cutoff
        uid = table.column("uid").to_numpy()[kept].astype(np.int64)
        keys.append((uid << 32) | table.column("item_id").to_numpy()[kept].astype(np.int64))
        signs.append(np.full(kept.sum(), sign))
        times.append(stamp[kept])
    keys = np.concatenate(keys)
    signs = np.concatenate(signs)
    order = np.lexsort((np.concatenate(times), keys))
    keys = keys[order]
    signs = signs[order]
    last = np.ones(len(keys), dtype=bool)
    last[:-1] = keys[1:] != keys[:-1]
    return np.sort(keys[last & (signs > 0)])


def candidate_sources(candidates, lists):
    """Names of the candidate lists that hold each candidate."""
    out = []
    for item in candidates:
        names = []
        for j in range(len(lists)):
            if SOURCES[j] not in names and item in lists[j]:
                names.append(SOURCES[j])
        out.append(names)
    return out


def item_groups(raw_dir, filename, column, known=None):
    """Sorted item ids and their artist or album (the first one when there are several),
    limited to the known items when given."""
    table = pq.read_table(str(Path(raw_dir) / filename))
    items = table.column("item_id").to_numpy().astype(np.int64)
    groups = table.column(column).to_numpy().astype(np.int64)
    items, first = np.unique(items, return_index=True)
    groups = groups[first]
    if known is None:
        return items, groups
    kept = np.isin(items, known)
    return items[kept], groups[kept]


def load_sasrec(model_path):
    """SASRec on CPU in eval mode; torch is only needed offline, not by the service."""
    import torch

    from nexttrack.models.sasrec import SASRec

    saved = torch.load(model_path, map_location="cpu")
    cfg = saved["config"]
    model = SASRec(
        saved["num_items"], cfg["max_seq_len"], cfg["embedding_dim"], cfg["num_layers"], cfg["num_heads"], 0.0
    )
    model.load_state_dict(saved["state"])
    model.eval()
    return model, cfg["max_seq_len"]


def sasrec_vectors(model_path, prepared, split, target_uids):
    """User vectors on CPU and the item embedding table (row = item_idx)."""
    import torch

    model, max_len = load_sasrec(model_path)
    uids, histories = read_histories(prepared / ("history_" + split))
    inputs = last_inputs(align(uids, histories, target_uids), max_len)
    parts = []
    with torch.no_grad():
        for start in range(0, len(inputs), 256):
            parts.append(model.user_vectors(torch.from_numpy(inputs[start : start + 256])).numpy())
        return np.concatenate(parts), model.items.weight.numpy()


def item_tables(config, prepared, all_items, all_times, cutoff, known_only=False):
    """Item level arrays shared by all users of a day, also exported for the service.

    known_only keeps the artist and album of training and history tracks only: enough for a
    variant without audio source (smaller bundle), not for the audio kNN candidates."""
    pop_items, pop_scores = decayed_scores(all_items, all_times, config["candidates"]["decay_pop_hours"])
    count_items, counts = np.unique(all_items, return_counts=True)
    likes = read_table(prepared / "likes", ["item_id", "timestamp"])
    liked = likes.column("item_id").to_numpy()[likes.column("timestamp").to_numpy() < cutoff]
    like_items, like_counts = np.unique(liked, return_counts=True)
    item_map = read_table(prepared / "item_map", ["item_id", "item_idx"])
    map_order = np.argsort(item_map.column("item_id").to_numpy())
    map_ids = item_map.column("item_id").to_numpy()[map_order]
    known = np.union1d(count_items, map_ids) if known_only else None
    raw = config["paths"]["raw"]
    artist_items, artists = item_groups(raw, constants.ARTIST_FILE, "artist_id", known)
    album_items, albums = item_groups(raw, constants.ALBUM_FILE, "album_id", known)
    return {
        "pop_items": pop_items,
        "pop_scores": pop_scores,
        "pop_top": top_items(pop_items, pop_scores, config["candidates"]["decay_pop"]),
        "count_items": count_items,
        "counts": counts,
        "like_items": like_items,
        "like_counts": like_counts,
        "artist_items": artist_items,
        "artists": artists,
        "album_items": album_items,
        "albums": albums,
        "map_ids": map_ids,
        "map_idxs": item_map.column("item_idx").to_numpy()[map_order],
    }


def load_day(config, split):
    """Everything the features of one day need, read once."""
    paths = config["paths"]
    prepared = Path(paths["prepared"])
    bounds = constants.split_bounds(config["split"]["shift_days"])
    # Validation features use events before TRAIN_END, test features events before VAL_END.
    cutoff = bounds["train_end"] if split == "val" else bounds["val_end"]
    history = read_table(prepared / ("history_" + split), ["uid", "item_ids", "timestamps"])
    targets = read_table(prepared / (split + "_targets"), ["uid", "item_ids"])
    target_uids = targets.column("uid").to_numpy()
    history_uids = history.column("uid").to_numpy()
    items = list_column(history, "item_ids")
    times = list_column(history, "timestamps")
    day = item_tables(config, prepared, np.concatenate(items), np.concatenate(times), cutoff)
    sasrec_dir = Path(config["candidates"]["sasrec_dir"])
    day["cutoff"] = cutoff
    day["uids"] = target_uids
    day["histories"] = align(history_uids, items, target_uids)
    day["history_times"] = align(history_uids, times, target_uids)
    day["targets"] = list_column(targets, "item_ids")
    day["liked"] = net_state(prepared, "likes", "unlikes", cutoff)
    day["disliked"] = net_state(prepared, "dislikes", "undislikes", cutoff)
    sasrec_uids, sasrec_items = read_ranked(sasrec_dir / ("candidates_%s.parquet" % split))
    day["sasrec"] = align(sasrec_uids, sasrec_items, target_uids)
    vectors = sasrec_vectors(sasrec_dir / "model.pt", prepared, split, target_uids)
    day["user_vectors"], day["item_vectors"] = vectors
    if uses_audio(config):
        audio_ids, audio_vectors = load_audio(paths["raw"], paths["audio"])
        day["audio_ids"] = audio_ids
        day["audio_vectors"] = audio_vectors
        audio_dir = Path(config["candidates"]["audio_dir"])
        audio_uids, audio_items = read_ranked(audio_dir / ("candidates_%s.parquet" % split))
        day["audio_lists"] = align(audio_uids, audio_items, target_uids)
        num_items = config["candidates"]["audio_history_items"]
        day["audio_query"] = user_queries(day["histories"], audio_ids, audio_vectors, num_items)
        day["audio_last"] = user_queries(day["histories"], audio_ids, audio_vectors, 1)
    return day


def lookup_values(keys, values, wanted, missing):
    if len(keys) == 0:
        return np.full(len(wanted), missing)
    rows = find(keys, wanted)
    return np.where(rows >= 0, values[np.maximum(rows, 0)], missing)


HISTORY_FEATURES = [
    "history_score",
    "user_item_count",
    "user_item_count_recent",
    "hours_since_last",
    "events_since_last",
]


def history_features(items, times, cutoff, candidates, hours):
    """Columns of HISTORY_FEATURES for each candidate; recency is NaN for unseen items."""
    out = {}
    for name in HISTORY_FEATURES:
        out[name] = np.zeros(len(candidates))
    out["hours_since_last"][:] = np.nan
    out["events_since_last"][:] = np.nan
    if len(items) == 0:
        return out
    unique, scores = decayed_scores(items, times, hours)
    _, counts = np.unique(items, return_counts=True)
    _, reversed_first = np.unique(items[::-1], return_index=True)
    last_position = len(items) - 1 - reversed_first
    rows = find(unique, candidates)
    seen = rows >= 0
    safe = np.maximum(rows, 0)
    out["history_score"] = np.where(seen, scores[safe], 0.0)
    out["user_item_count"] = np.where(seen, counts[safe], 0.0)
    recent, recent_counts = np.unique(items[-RECENT_EVENTS:], return_counts=True)
    out["user_item_count_recent"] = lookup_values(recent, recent_counts, candidates, 0)
    # Hours keep values below 8192 (see the rounding in user_rows).
    out["hours_since_last"] = np.where(seen, (cutoff - times[last_position[safe]]) / 3600.0, np.nan)
    out["events_since_last"] = np.where(seen, len(items) - 1 - last_position[safe], np.nan)
    return out


def cosine(vectors, rows, query):
    """Inner product with the query, NaN without audio or without query."""
    if not query.any():
        return np.full(len(rows), np.nan)
    values = np.asarray(vectors[np.maximum(rows, 0)]) @ query
    return np.where(rows >= 0, values, np.nan)


def user_rows(day, i, config):
    """Candidates, feature matrix and candidate lists (see SOURCES) of the i-th target user."""
    quotas = config["candidates"]
    items = day["histories"][i]
    times = day["history_times"][i]
    sasrec = day["sasrec"][i]
    history_top = np.zeros(0, dtype=np.int64)
    if len(items) > 0:
        unique, scores = decayed_scores(items, times, quotas["history_hours"])
        history_top = top_items(unique, scores, quotas["history"])
    audio = np.zeros(0, dtype=np.int64)
    if quotas["audio"] > 0:
        audio = day["audio_lists"][i][: quotas["audio"]]
    lists = [sasrec[: quotas["sasrec"]], history_top, audio, day["pop_top"], sasrec[quotas["sasrec"] :]]
    candidates = merge_sources(lists, quotas["total"]).astype(np.int64)

    columns = {}
    idx = lookup_values(day["map_ids"], day["map_idxs"], candidates, 0)
    warm = idx > 0
    columns["sasrec_score"] = np.where(warm, day["item_vectors"][idx] @ day["user_vectors"][i], np.nan)
    columns.update(history_features(items, times, day["cutoff"], candidates, quotas["history_hours"]))
    if config["rerank"]["audio_features"]:
        audio_rows = find(day["audio_ids"], candidates)
        columns["audio_score"] = cosine(day["audio_vectors"], audio_rows, day["audio_query"][i])
        columns["audio_last_score"] = cosine(day["audio_vectors"], audio_rows, day["audio_last"][i])
    columns["pop_score"] = lookup_values(day["pop_items"], day["pop_scores"], candidates, 0.0)
    columns["item_count"] = lookup_values(day["count_items"], day["counts"], candidates, 0)
    columns["item_like_count"] = lookup_values(day["like_items"], day["like_counts"], candidates, 0)
    keys = (np.int64(day["uids"][i]) << 32) | candidates
    columns["user_liked"] = find(day["liked"], keys) >= 0
    columns["user_disliked"] = find(day["disliked"], keys) >= 0
    candidate_artists = lookup_values(day["artist_items"], day["artists"], candidates, -1)
    history_artists = lookup_values(day["artist_items"], day["artists"], items, -1)
    artist_ids, artist_counts = np.unique(history_artists[history_artists >= 0], return_counts=True)
    artist_count = lookup_values(artist_ids, artist_counts, candidate_artists, 0)
    columns["user_artist_count"] = np.where(candidate_artists >= 0, artist_count, np.nan)
    columns["user_artist_share"] = columns["user_artist_count"] / max(len(items), 1)
    columns["is_cold"] = ~warm
    columns["history_length"] = np.full(len(candidates), len(items))
    names = feature_names(config)
    matrix = np.zeros((len(candidates), len(names)), dtype=np.float32)
    for j in range(len(names)):
        # Three decimals: below 8192 two distinct values are never adjacent float32 numbers,
        # so the float32 thresholds of the ONNX trees split exactly like LightGBM.
        matrix[:, j] = np.round(columns[names[j]].astype(np.float64), 3)
    return candidates, matrix, lists


def build_day(config, split):
    """Candidates, features, labels and group sizes of every target user of the day."""
    day = load_day(config, split)
    all_items = []
    matrices = []
    labels = []
    sizes = np.zeros(len(day["uids"]), dtype=np.int64)
    for i in range(len(day["uids"])):
        candidates, matrix, _ = user_rows(day, i, config)
        all_items.append(candidates)
        matrices.append(matrix)
        labels.append(np.isin(candidates, day["targets"][i]).astype(np.float32))
        sizes[i] = len(candidates)
    return {
        "uids": day["uids"],
        "names": feature_names(config),
        "targets": day["targets"],
        "sizes": sizes,
        "items": np.concatenate(all_items),
        "features": np.concatenate(matrices),
        "labels": np.concatenate(labels),
        "audio_lists": day.get("audio_lists"),
        "histories": day["histories"],
        "history_times": day["history_times"],
    }
