"""LightGBM reranker trained on the validation day, scored on the test day, exported to ONNX.

Labels are binary: the candidate is listened (Listen+) during the day. Users without any
positive candidate are left out of training. The number of boosting rounds maximizes the
mean ndcg@10 of a cross-validation over the validation users; the final model is then
trained on all of them. The test day is only used for the final evaluation.
"""

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import onnxruntime
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

from nexttrack import constants
from nexttrack.config import parse_args
from nexttrack.eval.baselines import history_ranking
from nexttrack.eval.metrics import cold_targets, evaluate, paired_bootstrap
from nexttrack.rerank.features import FEATURES, build_day
from nexttrack.tables import read_item_ids, seen_item_sets, to_builtin, write_json, write_ranked


def offsets_of(sizes):
    return np.concatenate([[0], np.cumsum(sizes)])


def ranked_matrix(data, scores, width):
    """Candidates of each user sorted by decreasing score, padded with -1."""
    offsets = offsets_of(data["sizes"])
    ranked = np.full((len(data["sizes"]), width), -1, dtype=np.int64)
    top_scores = np.full((len(data["sizes"]), width), np.nan, dtype=np.float32)
    for i in range(len(data["sizes"])):
        items = data["items"][offsets[i] : offsets[i + 1]]
        values = scores[offsets[i] : offsets[i + 1]]
        order = np.argsort(-values, kind="stable")[:width]
        ranked[i, : len(order)] = items[order]
        top_scores[i, : len(order)] = values[order]
    return ranked, top_scores


def subset(data, users):
    """Rows and group sizes of the selected users."""
    offsets = offsets_of(data["sizes"])
    rows = []
    for i in users:
        rows.append(np.arange(offsets[i], offsets[i + 1]))
    rows = np.concatenate(rows)
    return data["features"][rows], data["labels"][rows], data["sizes"][users]


def user_folds(data, num_folds, seed):
    """Users with a positive candidate, shuffled into num_folds folds."""
    offsets = offsets_of(data["sizes"])
    useful = []
    for i in range(len(data["sizes"])):
        if data["labels"][offsets[i] : offsets[i + 1]].any():
            useful.append(i)
    useful = np.array(useful)
    np.random.default_rng(seed).shuffle(useful)
    folds = []
    for fold in np.array_split(useful, num_folds):
        folds.append(np.sort(fold))
    return folds


def dataset(data, users, reference=None):
    features, labels, groups = subset(data, users)
    return lgb.Dataset(features, labels, group=groups, feature_name=FEATURES, reference=reference)


def train(data, cfg):
    """Rounds chosen on the mean cross-validation curve, then one model on every fold."""
    params = {
        "objective": cfg["objective"],
        "metric": "ndcg",
        "eval_at": [10],
        "num_leaves": cfg["num_leaves"],
        "learning_rate": cfg["learning_rate"],
        "seed": cfg["seed"],
        # Same trees on every run with the same thread count.
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }
    folds = user_folds(data, cfg["folds"], cfg["seed"])
    curves = []
    for f in range(len(folds)):
        others = []
        for g in range(len(folds)):
            if g != f:
                others.append(folds[g])
        train_set = dataset(data, np.sort(np.concatenate(others)))
        record = {}
        held = dataset(data, folds[f], train_set)
        lgb.train(
            params, train_set, cfg["num_rounds"], valid_sets=[held], callbacks=[lgb.record_evaluation(record)]
        )
        curves.append(record["valid_0"]["ndcg@10"])
    mean_curve = np.mean(curves, axis=0)
    rounds = int(mean_curve.argmax()) + 1
    all_users = np.sort(np.concatenate(folds))
    booster = lgb.train(params, dataset(data, all_users), rounds)
    summary = {"rounds": rounds, "cv_ndcg@10": float(mean_curve[rounds - 1]), "users": len(all_users)}
    summary["cv_curve_every_25"] = mean_curve[24::25].tolist()
    return booster, summary


def export_onnx(model_file, features, path):
    """ONNX copy of the saved booster; returns ONNX and LightGBM scores on the given rows."""
    booster = lgb.Booster(model_file=str(model_file))
    model = convert_lightgbm(booster, initial_types=[("features", FloatTensorType([None, len(FEATURES)]))])
    with open(path, "wb") as handle:
        handle.write(model.SerializeToString())
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    return session.run(None, {name: features})[0].ravel(), booster.predict(features)


def next_unused(source, start, used):
    """First item of source from start that is valid and not used, and the next start."""
    while start < len(source):
        item = int(source[start])
        start += 1
        if item >= 0 and item not in used:
            return item, start
    return -1, start


def reserve_audio(ranked, audio_lists, every, width):
    """Positions every, 2 * every, ... hold the next audio kNN tracks, the rest follows ranked."""
    out = np.full((len(ranked), width), -1, dtype=np.int64)
    for i in range(len(ranked)):
        sources = [ranked[i], audio_lists[i]]
        starts = [0, 0]
        used = set()
        for slot in range(width):
            pick = 1 if (slot + 1) % every == 0 else 0
            item, starts[pick] = next_unused(sources[pick], starts[pick], used)
            if item < 0:
                # No audio track left (no audio in the history): fall back to ranked.
                item, starts[0] = next_unused(sources[0], starts[0], used)
            out[i, slot] = item
            used.add(item)
    return out


def paired_comparisons(ranked, history, reserved, targets, k_values, strict_items):
    """Paired bootstrap differences on all users, and on the cold_strict targets for the variant."""
    users, subset, _ = cold_targets(targets, strict_items)
    return {
        "reranker_minus_history": paired_bootstrap(ranked, history, targets, k_values),
        "reserved_minus_reranker": paired_bootstrap(reserved, ranked, targets, k_values),
        "reserved_minus_reranker_cold_strict": paired_bootstrap(
            reserved[users], ranked[users], subset, k_values
        ),
    }


def main():
    config, _ = parse_args("Train the LightGBM reranker")
    cfg = config["rerank"]
    k_values = config["eval"]["k_values"]
    prepared = Path(config["paths"]["prepared"])
    out_dir = Path(config["paths"]["outputs"]) / "rerank"
    catalogue_size = len(read_item_ids(prepared / "item_map"))
    seen = seen_item_sets(prepared)
    width = config["candidates"]["total"]

    started = time.time()
    val = build_day(config, "val")
    test = build_day(config, "test")
    feature_seconds = time.time() - started
    print("features %.0fs, rows %d and %d" % (feature_seconds, len(val["items"]), len(test["items"])))

    # Union in source order: the upper bound the reranker can reach.
    union = {}
    for name, data, sets in [("validation", val, {"cold_train": seen["cold_train"]}), ("test", test, seen)]:
        order = -np.arange(len(data["items"]), dtype=np.float64)
        ranked, _ = ranked_matrix(data, order, width)
        union[name] = evaluate(ranked, data["targets"], [width], catalogue_size, sets)

    started = time.time()
    booster, training = train(val, cfg)
    train_seconds = time.time() - started
    scores = booster.predict(test["features"])
    ranked, top_scores = ranked_matrix(test, scores, constants.NUM_RANKED)
    test_metrics = evaluate(ranked, test["targets"], k_values, catalogue_size, seen)
    write_ranked(out_dir / "ranked_test.parquet", test["uids"], ranked, top_scores)
    booster.save_model(str(out_dir / "reranker.txt"))
    model_file = out_dir / "reranker.txt"
    onnx_scores, saved_scores = export_onnx(model_file, test["features"], out_dir / "reranker.onnx")
    # ONNX tree thresholds are float32: rows whose value falls between a LightGBM threshold
    # and its float32 rounding take the other branch; outputs are float32 (gaps near 1e-7).
    onnx_gap = np.abs(onnx_scores - scores)
    onnx_ranked, _ = ranked_matrix(test, onnx_scores, constants.NUM_RANKED)
    onnx_ndcg = evaluate(onnx_ranked, test["targets"], [10], catalogue_size)["all"]["ndcg@10"]

    reserved = reserve_audio(ranked, test["audio_lists"], cfg["audio_every"], constants.NUM_RANKED)
    reserved_metrics = evaluate(reserved, test["targets"], k_values, catalogue_size, seen)
    hours = config["candidates"]["history_hours"]
    history = history_ranking(test["histories"], test["history_times"], hours, constants.NUM_RANKED)
    paired = paired_comparisons(ranked, history, reserved, test["targets"], k_values, seen["cold_strict"])

    importance = {}
    gains = booster.feature_importance("gain")
    for j in range(len(FEATURES)):
        importance[FEATURES[j]] = float(gains[j])
    report = {
        "config": {"candidates": config["candidates"], "rerank": cfg},
        "features": FEATURES,
        "feature_seconds": feature_seconds,
        "train_seconds": train_seconds,
        "training": training,
        "label_share_val": float(val["labels"].mean()),
        "candidate_union": union,
        "test": test_metrics,
        "test_reserved_audio": reserved_metrics,
        "paired": paired,
        "importance_gain": importance,
        "onnx_max_abs_gap": float(onnx_gap.max()),
        "onnx_rows_differing": int((onnx_gap > 1e-4).sum()),
        "saved_model_max_abs_gap": float(np.abs(saved_scores - scores).max()),
        "onnx_test_ndcg@10": onnx_ndcg,
    }
    write_json(to_builtin(report), Path(config["paths"]["reports"]) / "reranker.json")
    print("test", test_metrics["all"], flush=True)
    print("onnx rows differing", int((onnx_gap > 1e-4).sum()), "onnx ndcg@10", onnx_ndcg)


if __name__ == "__main__":
    main()
