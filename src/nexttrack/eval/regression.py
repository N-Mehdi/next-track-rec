"""Non-regression check of the reranker on a small frozen set.

The frozen set holds the candidates, features and targets of 600 users of the validation
day (never the test day). The reranker is trained with the current code on the first 400
users and scored on the last 200. The check fails when ndcg@10 falls more than the
tolerance below the stored reference, or when the ONNX copy ranks differently.

  python -m nexttrack.eval.regression            check, exit code 1 on regression
  python -m nexttrack.eval.regression --freeze   rebuild the frozen set (needs data/prepared)
  python -m nexttrack.eval.regression --update   store the current value as the reference
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from nexttrack.config import copy_config, parse_args
from nexttrack.eval.metrics import evaluate, lengths
from nexttrack.rerank.train_reranker import export_onnx, offsets_of, ranked_matrix, train

FROZEN = Path("tests/data/regression_day.npz")
REFERENCE = Path("tests/data/regression_reference.json")
NUM_USERS = 600
NUM_TRAIN = 400
TOLERANCE = 0.005


def freeze(config):
    """Sample users of the validation day with a fixed seed and store their rows."""
    from nexttrack.rerank.features import build_day

    day = build_day(config, "val")
    users = np.sort(np.random.default_rng(0).choice(len(day["sizes"]), NUM_USERS, replace=False))
    offsets = offsets_of(day["sizes"])
    rows = []
    targets = []
    for i in users:
        rows.append(np.arange(offsets[i], offsets[i + 1]))
        targets.append(day["targets"][i])
    rows = np.concatenate(rows)
    FROZEN.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FROZEN,
        sizes=day["sizes"][users],
        items=day["items"][rows],
        features=day["features"][rows],
        labels=day["labels"][rows],
        targets=np.concatenate(targets),
        target_sizes=lengths(targets),
    )


def part(data, start, stop):
    """Users start to stop of the frozen set, in the layout used by the reranker."""
    rows = offsets_of(data["sizes"])
    cuts = offsets_of(data["target_sizes"])
    targets = []
    for i in range(start, stop):
        targets.append(data["targets"][cuts[i] : cuts[i + 1]])
    out = {"sizes": data["sizes"][start:stop], "targets": targets}
    for name in ["items", "features", "labels"]:
        out[name] = data[name][rows[start] : rows[stop]]
    return out


def measure(data, cfg):
    """ndcg@10 of a reranker trained on the first users, with LightGBM and with ONNX."""
    cfg = copy_config(cfg)
    # One thread and fewer rounds: same trees on every machine, a few seconds.
    cfg["num_threads"] = 1
    cfg["num_rounds"] = 150
    held = part(data, NUM_TRAIN, len(data["sizes"]))
    booster, summary = train(part(data, 0, NUM_TRAIN), cfg)
    values = {"rounds": summary["rounds"]}
    with tempfile.TemporaryDirectory() as folder:
        booster.save_model(folder + "/model.txt")
        onnx_scores, scores = export_onnx(folder + "/model.txt", held["features"], folder + "/model.onnx")
    for name, column in [("ndcg@10", scores), ("onnx_ndcg@10", onnx_scores)]:
        ranked, _ = ranked_matrix(held, column, 10)
        values[name] = evaluate(ranked, held["targets"], [10], 1)["all"]["ndcg@10"]
    return values


def failures(values, reference):
    """Reasons why the measured values are a regression; empty when the check passes."""
    reasons = []
    if values["ndcg@10"] < reference["ndcg@10"] - TOLERANCE:
        reasons.append("ndcg@10 %.4f below reference %.4f" % (values["ndcg@10"], reference["ndcg@10"]))
    if abs(values["onnx_ndcg@10"] - values["ndcg@10"]) > 0.001:
        reasons.append("ONNX ndcg@10 %.4f differs from LightGBM" % values["onnx_ndcg@10"])
    return reasons


def add_arguments(parser):
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--update", action="store_true")


def main():
    config, args = parse_args("Reranker non-regression check", add_arguments)
    if args.freeze:
        freeze(config)
    values = measure(dict(np.load(FROZEN)), config["rerank"])
    print(values)
    if args.freeze or args.update:
        with open(REFERENCE, "w", encoding="utf-8") as handle:
            json.dump(values, handle, indent=2)
        return
    with open(REFERENCE, encoding="utf-8") as handle:
        reasons = failures(values, json.load(handle))
    for reason in reasons:
        print("REGRESSION:", reason)
    sys.exit(1 if reasons else 0)


if __name__ == "__main__":
    main()
