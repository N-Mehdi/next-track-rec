"""Ranking metrics following the Yambda protocol.

ranked: int array (users, n) of item ids sorted by decreasing score.
targets: list of int arrays, one per user, duplicates kept as in the official code.

recall          hits / min(targets, k), current benchmarks/yambda/evaluation/metrics.py
recall_paper    hits / targets, formula behind the paper tables
ndcg            mean over users of DCG / ideal DCG
ndcg_paper      sum of DCG / sum of ideal DCG, formula behind the paper tables
ndcg_official   current metrics.py: ideal DCG computed from the hit mask, equals the hit rate
"""

import numpy as np


def hit_matrix(ranked, targets):
    hits = np.zeros(ranked.shape, dtype=np.float32)
    for i in range(len(targets)):
        hits[i] = np.isin(ranked[i], targets[i])
    return hits


def ranking_metrics(ranked, hits, num_targets, k_values, catalogue_size):
    num_targets = np.asarray(num_targets, dtype=np.float64)
    discounts = 1.0 / np.log2(np.arange(2, hits.shape[1] + 2, dtype=np.float64))
    ideal_cumulative = np.cumsum(discounts)
    result = {}
    for k in k_values:
        top = hits[:, :k]
        hit_count = top.sum(axis=1)
        denominator = np.minimum(num_targets, k)
        recall = np.where(denominator > 0, hit_count / np.maximum(denominator, 1), 0.0)
        recall_paper = np.where(num_targets > 0, hit_count / np.maximum(num_targets, 1), 0.0)
        dcg = (top * discounts[:k]).sum(axis=1)
        ideal_count = np.minimum(num_targets, k).astype(np.int64)
        ideal = np.where(ideal_count > 0, ideal_cumulative[np.maximum(ideal_count, 1) - 1], 0.0)
        ndcg = np.where(ideal > 0, dcg / np.maximum(ideal, 1e-12), 0.0)
        ndcg_paper = dcg.sum() / ideal.sum() if ideal.sum() > 0 else 0.0
        ndcg_official = np.where(dcg > 0, 1.0, 0.0)
        result["recall@%d" % k] = float(recall.mean())
        result["recall_paper@%d" % k] = float(recall_paper.mean())
        result["ndcg@%d" % k] = float(ndcg.mean())
        result["ndcg_paper@%d" % k] = float(ndcg_paper)
        result["ndcg_official@%d" % k] = float(ndcg_official.mean())
        result["coverage@%d" % k] = len(np.unique(ranked[:, :k])) / float(catalogue_size)
    return result


def lengths(arrays):
    out = np.zeros(len(arrays), dtype=np.int64)
    for i in range(len(arrays)):
        out[i] = len(arrays[i])
    return out


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
    """Metrics on all targets and, when train_items is given, on never-seen targets."""
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
    return result
