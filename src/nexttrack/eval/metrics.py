"""Ranking metrics following the Yambda protocol.

ranked: int array (users, n) of item ids sorted by decreasing score, -1 for empty slots.
targets: list of int arrays, one per user, duplicates kept as in the official code.

recall          hits / min(targets, k), current benchmarks/yambda/evaluation/metrics.py
recall_paper    hits / targets, formula behind the paper tables
ndcg            mean over users of DCG / ideal DCG
ndcg_paper      sum of DCG / sum of ideal DCG, formula behind the paper tables
ndcg_official   current metrics.py: ideal DCG computed from the hit mask, equals the hit rate
"""

import numpy as np

BOOTSTRAP_SAMPLES = 1000


def hit_matrix(ranked, targets):
    hits = np.zeros(ranked.shape, dtype=np.float32)
    for i in range(len(targets)):
        hits[i] = np.isin(ranked[i], targets[i])
    return hits


def lengths(arrays):
    out = np.zeros(len(arrays), dtype=np.int64)
    for i in range(len(arrays)):
        out[i] = len(arrays[i])
    return out


def per_user(hits, num_targets, k):
    """Per user values at cutoff k: recall, recall_paper, ndcg, hit, dcg, ideal DCG."""
    num_targets = np.asarray(num_targets, dtype=np.float64)
    discounts = 1.0 / np.log2(np.arange(2, hits.shape[1] + 2, dtype=np.float64))
    top = hits[:, :k]
    hit_count = top.sum(axis=1)
    dcg = (top * discounts[:k]).sum(axis=1)
    ideal_count = np.minimum(num_targets, k).astype(np.int64)
    ideal_cumulative = np.cumsum(discounts)
    ideal = np.where(ideal_count > 0, ideal_cumulative[np.maximum(ideal_count, 1) - 1], 0.0)
    values = {}
    values["recall"] = np.where(ideal_count > 0, hit_count / np.maximum(ideal_count, 1), 0.0)
    values["recall_paper"] = np.where(num_targets > 0, hit_count / np.maximum(num_targets, 1), 0.0)
    values["ndcg"] = np.where(ideal > 0, dcg / np.maximum(ideal, 1e-12), 0.0)
    values["ndcg_official"] = np.where(dcg > 0, 1.0, 0.0)
    values["dcg"] = dcg
    values["ideal"] = ideal
    return values


def summarize(values, rows=None):
    """Aggregate per user values, optionally on a subset of row indices."""
    if rows is None:
        rows = np.arange(len(values["dcg"]))
    ideal_sum = values["ideal"][rows].sum()
    out = {}
    for name in ["recall", "recall_paper", "ndcg", "ndcg_official"]:
        out[name] = float(values[name][rows].mean())
    out["ndcg_paper"] = float(values["dcg"][rows].sum() / ideal_sum) if ideal_sum > 0 else 0.0
    return out


def coverage(ranked, k, catalogue_size):
    top = ranked[:, :k]
    return len(np.unique(top[top >= 0])) / float(catalogue_size)


def ranking_metrics(ranked, hits, num_targets, k_values, catalogue_size):
    result = {}
    for k in k_values:
        summary = summarize(per_user(hits, num_targets, k))
        for name in summary:
            result["%s@%d" % (name, k)] = summary[name]
        result["coverage@%d" % k] = coverage(ranked, k, catalogue_size)
    return result


def bootstrap(hits, num_targets, k_values, samples=BOOTSTRAP_SAMPLES, seed=0):
    """95 % percentile intervals obtained by resampling users with replacement."""
    rng = np.random.default_rng(seed)
    num_users = hits.shape[0]
    draws = []
    for _ in range(samples):
        draws.append(rng.integers(0, num_users, size=num_users))
    result = {}
    for k in k_values:
        values = per_user(hits, num_targets, k)
        collected = {}
        for draw in draws:
            summary = summarize(values, draw)
            for name in summary:
                if name not in collected:
                    collected[name] = []
                collected[name].append(summary[name])
        for name in collected:
            low, high = np.percentile(collected[name], [2.5, 97.5])
            result["%s@%d" % (name, k)] = [float(low), float(high)]
    return result


def cold_targets(targets, train_items):
    """Targets never seen in training, with the indices of users that have at least one."""
    flat = np.concatenate(targets)
    unseen = ~np.isin(flat, train_items)
    offsets = np.concatenate([[0], np.cumsum(lengths(targets))])
    kept_users = []
    kept_targets = []
    for i in range(len(targets)):
        mask = unseen[offsets[i] : offsets[i + 1]]
        if mask.any():
            kept_users.append(i)
            kept_targets.append(targets[i][mask])
    return kept_users, kept_targets, float(unseen.mean())


def evaluate(ranked, targets, k_values, catalogue_size, train_items=None):
    """Metrics on all targets and, when train_items is given, on never-seen targets with CIs."""
    ranked = np.asarray(ranked)
    hits = hit_matrix(ranked, targets)
    result = {"users": len(targets)}
    result["all"] = ranking_metrics(ranked, hits, lengths(targets), k_values, catalogue_size)
    if train_items is not None:
        users, subset, share = cold_targets(targets, train_items)
        result["cold_target_share"] = share
        result["cold_users"] = len(users)
        if len(users) > 0:
            cold_ranked = ranked[users]
            cold_hits = hit_matrix(cold_ranked, subset)
            cold_lengths = lengths(subset)
            result["cold"] = ranking_metrics(cold_ranked, cold_hits, cold_lengths, k_values, catalogue_size)
            result["cold_ci95"] = bootstrap(cold_hits, cold_lengths, k_values)
    return result
