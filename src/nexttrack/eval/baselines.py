"""Popularity and personal history baselines.

MostPop and DecayPop check the split and the metrics against the paper: on Listen+ 50M
the paper reports recall_paper@10 of 0.0064 and 0.0122, ndcg_paper@10 of 0.0186 and 0.0260.
History ranks the tracks of the user's own history by recency weighted listen counts.
Decay periods are tuned on the validation day, then models are scored on the test day
with every event before VAL_END, as the official baselines do.
"""

from pathlib import Path

import numpy as np

from nexttrack import constants
from nexttrack.config import parse_args
from nexttrack.eval.metrics import evaluate
from nexttrack.tables import flat_column, list_column, read_item_ids, read_table, write_json

POP_HOURS = [0.5, 1, 2, 3, 6, 12, 24]
HISTORY_HOURS = [6, 24, 72, 168, 720, 2160]


def decayed_scores(item_ids, timestamps, hours):
    """Sum of 0.9 ** (age / hours) per item; hours None gives plain counts."""
    unique, inverse = np.unique(item_ids, return_inverse=True)
    if hours is None:
        weights = np.ones(len(item_ids), dtype=np.float64)
    else:
        age = timestamps.max() - timestamps.astype(np.float64)
        weights = 0.9 ** (age / (hours * 3600.0))
    scores = np.bincount(inverse, weights=weights, minlength=len(unique))
    return unique, scores


def top_items(unique, scores, k):
    """Best k items, padded with -1 when fewer are available."""
    order = np.argsort(-scores, kind="stable")[:k]
    out = np.full(k, -1, dtype=np.int64)
    out[: len(order)] = unique[order]
    return out


def history_ranking(histories, history_times, hours, k):
    ranked = np.full((len(histories), k), -1, dtype=np.int64)
    for i in range(len(histories)):
        if len(histories[i]) > 0:
            unique, scores = decayed_scores(histories[i], history_times[i], hours)
            ranked[i] = top_items(unique, scores, k)
    return ranked


def load_split(prepared, split):
    """Targets and the matching histories, aligned on the target users."""
    history = read_table(prepared / ("history_" + split), ["uid", "item_ids", "timestamps"])
    targets = read_table(prepared / (split + "_targets"), ["uid", "item_ids"])
    position = {}
    history_uids = history.column("uid").to_numpy()
    for i in range(len(history_uids)):
        position[int(history_uids[i])] = i
    items = list_column(history, "item_ids")
    times = list_column(history, "timestamps")
    target_uids = targets.column("uid").to_numpy()
    user_items = []
    user_times = []
    for uid in target_uids:
        row = position[int(uid)]
        user_items.append(items[row])
        user_times.append(times[row])
    split_data = {
        "all_items": flat_column(history, "item_ids"),
        "all_times": flat_column(history, "timestamps"),
        "user_items": user_items,
        "user_times": user_times,
        "targets": list_column(targets, "item_ids"),
    }
    return split_data


def run_split(prepared, split, pop_hours, history_hours, k_values, train_items):
    data = load_split(prepared, split)
    targets = data["targets"]
    catalogue_size = len(np.unique(data["all_items"]))
    results = {}
    for hours in pop_hours:
        unique, scores = decayed_scores(data["all_items"], data["all_times"], hours)
        ranked = np.tile(top_items(unique, scores, constants.NUM_RANKED), (len(targets), 1))
        name = "most_pop" if hours is None else "decay_pop_%gh" % hours
        results[name] = evaluate(ranked, targets, k_values, catalogue_size, train_items)
        print(split, name, "recall@10=%.4f" % results[name]["all"]["recall@10"], flush=True)
    for hours in history_hours:
        ranked = history_ranking(data["user_items"], data["user_times"], hours, constants.NUM_RANKED)
        name = "history_count" if hours is None else "history_%gh" % hours
        results[name] = evaluate(ranked, targets, k_values, catalogue_size, train_items)
        print(split, name, "recall@10=%.4f" % results[name]["all"]["recall@10"], flush=True)
    return results


def best_hours(results, prefix, hours_list, metric):
    best = None
    best_value = -1.0
    for hours in hours_list:
        value = results["%s_%gh" % (prefix, hours)]["all"][metric]
        if value > best_value:
            best_value = value
            best = hours
    return best


def main():
    config, _ = parse_args("Popularity and history baselines")
    prepared = Path(config["paths"]["prepared"])
    k_values = config["eval"]["k_values"]
    train_items = read_item_ids(prepared / "item_map")

    history_grid = [None] + HISTORY_HOURS
    validation = run_split(prepared, "val", [None] + POP_HOURS, history_grid, k_values, train_items)
    pop_best = best_hours(validation, "decay_pop", POP_HOURS, "ndcg@100")
    history_best = best_hours(validation, "history", HISTORY_HOURS, "ndcg@100")
    test = run_split(prepared, "test", [None, pop_best], [history_best], k_values, train_items)

    report = {
        "best_decay_pop_hours": pop_best,
        "best_history_hours": history_best,
        "validation": validation,
        "test": test,
    }
    write_json(report, Path(config["paths"]["reports"]) / "baselines.json")


if __name__ == "__main__":
    main()
