import math

import numpy as np
import pytest

from nexttrack.eval.metrics import evaluate

RANKED = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
TARGETS = [np.array([2, 2, 9]), np.array([10])]


def test_metrics_by_hand():
    result = evaluate(RANKED, TARGETS, [2, 5], catalogue_size=10)["all"]
    d2 = 1.0 / math.log2(3)
    d3 = 1.0 / math.log2(4)
    d5 = 1.0 / math.log2(6)
    # user 0: one hit at rank 2 among 3 targets; user 1: one hit at rank 5 among 1 target.
    assert result["recall@2"] == pytest.approx((1 / 2 + 0) / 2)
    assert result["recall@5"] == pytest.approx((1 / 3 + 1) / 2)
    assert result["ndcg@2"] == pytest.approx((d2 / (1 + d2) + 0) / 2)
    assert result["ndcg@5"] == pytest.approx((d2 / (1 + d2 + d3) + d5 / 1) / 2)
    assert result["ndcg_official@2"] == pytest.approx(0.5)
    assert result["ndcg_official@5"] == pytest.approx(1.0)
    assert result["recall_paper@2"] == pytest.approx((1 / 3 + 0) / 2)
    assert result["recall_paper@5"] == pytest.approx((1 / 3 + 1) / 2)
    assert result["ndcg_paper@2"] == pytest.approx(d2 / ((1 + d2) + 1))
    assert result["ndcg_paper@5"] == pytest.approx((d2 + d5) / ((1 + d2 + d3) + 1))
    assert result["coverage@2"] == pytest.approx(0.4)
    assert result["coverage@5"] == pytest.approx(1.0)


def test_cold_subsets_per_definition():
    seen = {"cold_train": np.array([1, 2, 3, 4, 5, 6, 7, 8]), "cold_strict": np.arange(1, 10)}
    result = evaluate(RANKED, TARGETS, [5], catalogue_size=10, seen_sets=seen)
    loose = result["cold_train"]
    assert loose["target_share"] == pytest.approx(0.5)
    assert loose["users"] == 2
    assert loose["targets"] == 2
    assert loose["metrics"]["recall@5"] == pytest.approx(0.5)
    assert loose["metrics"]["ndcg_official@5"] == pytest.approx(0.5)
    strict = result["cold_strict"]
    assert strict["users"] == 1
    assert strict["targets"] == 1
    assert strict["metrics"]["recall@5"] == pytest.approx(1.0)


def test_bootstrap_interval_contains_estimate():
    rng = np.random.default_rng(1)
    ranked = rng.integers(0, 50, size=(200, 10))
    targets = []
    for _ in range(200):
        targets.append(rng.integers(0, 50, size=3))
    train_items = np.arange(0, 40)
    result = evaluate(ranked, targets, [10], catalogue_size=50, seen_sets={"cold_train": train_items})
    for name in ["recall@10", "ndcg@10", "recall_paper@10", "ndcg_paper@10"]:
        low, high = result["cold_train"]["ci95"][name]
        assert low <= result["cold_train"]["metrics"][name] <= high
        assert high - low > 0


def test_empty_slots_are_ignored():
    ranked = np.array([[3, -1, -1], [-1, -1, -1]])
    result = evaluate(ranked, [np.array([3]), np.array([4])], [3], catalogue_size=4)["all"]
    assert result["recall@3"] == pytest.approx(0.5)
    assert result["coverage@3"] == pytest.approx(0.25)


def test_no_cold_targets():
    result = evaluate(RANKED, TARGETS, [5], catalogue_size=10, seen_sets={"cold_train": np.arange(1, 11)})
    assert result["cold_train"]["users"] == 0
    assert "metrics" not in result["cold_train"]
