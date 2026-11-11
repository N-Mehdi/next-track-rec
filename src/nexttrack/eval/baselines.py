"""MostPop and DecayPop baselines.

They also check the split and the metrics against the paper: on Listen+ 50M the paper
reports recall_paper@10 of 0.0064 (MostPop) and 0.0122 (DecayPop), and ndcg_paper@10 of
0.0186 and 0.0260.
The decay hour is tuned on the validation day, then both models are scored on the test
day with every event before VAL_END, as the official baselines do.
"""

from pathlib import Path

import numpy as np

from nexttrack import constants
from nexttrack.config import parse_args
from nexttrack.eval.metrics import evaluate
from nexttrack.tables import flat_column, read_item_ids, read_table, read_targets, write_json

HOURS = [0.5, 1, 2, 3, 6, 12, 24]


def popularity(item_ids, timestamps, hours):
    """Decayed interaction counts; hours None gives plain counts. Weight 0.9 per `hours`."""
    unique, inverse = np.unique(item_ids, return_inverse=True)
    if hours is None:
        weights = np.ones(len(item_ids), dtype=np.float64)
    else:
        age = timestamps.max() - timestamps.astype(np.float64)
        weights = 0.9 ** (age / (hours * 3600.0))
    scores = np.bincount(inverse, weights=weights, minlength=len(unique))
    return unique, scores


def top_items(unique, scores, k):
    order = np.argsort(-scores, kind="stable")[:k]
    return unique[order]


def run_split(prepared, split, hours_list, k_values, train_items):
    history = read_table(prepared / ("history_" + split), ["item_ids", "timestamps"])
    item_ids = flat_column(history, "item_ids")
    timestamps = flat_column(history, "timestamps")
    _, targets = read_targets(prepared / (split + "_targets"))
    catalogue_size = len(np.unique(item_ids))
    results = {}
    for hours in hours_list:
        unique, scores = popularity(item_ids, timestamps, hours)
        ranked = np.tile(top_items(unique, scores, constants.NUM_RANKED), (len(targets), 1))
        name = "most_pop" if hours is None else "decay_pop_%gh" % hours
        results[name] = evaluate(ranked, targets, k_values, catalogue_size, train_items)
        print(split, name, "recall@10=%.4f" % results[name]["all"]["recall@10"], flush=True)
    return results


def main():
    config, _ = parse_args("Popularity baselines")
    prepared = Path(config["paths"]["prepared"])
    k_values = config["eval"]["k_values"]
    train_items = read_item_ids(prepared / "item_map")

    validation = run_split(prepared, "val", [None] + HOURS, k_values, train_items)
    best_hours = None
    best_value = -1.0
    for hours in HOURS:
        value = validation["decay_pop_%gh" % hours]["all"]["ndcg@100"]
        if value > best_value:
            best_value = value
            best_hours = hours
    test = run_split(prepared, "test", [None, best_hours], k_values, train_items)

    report = {"best_decay_hours": best_hours, "validation": validation, "test": test}
    write_json(report, Path(config["paths"]["reports"]) / "baselines.json")


if __name__ == "__main__":
    main()
