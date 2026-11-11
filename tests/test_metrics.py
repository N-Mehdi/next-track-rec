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


def test_cold_subset():
    train_items = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    result = evaluate(RANKED, TARGETS, [5], catalogue_size=10, train_items=train_items)
    assert result["cold_target_share"] == pytest.approx(0.5)
    assert result["cold_users"] == 2
    assert result["cold"]["recall@5"] == pytest.approx(0.5)
    assert result["cold"]["ndcg_official@5"] == pytest.approx(0.5)


def test_no_cold_targets():
    result = evaluate(RANKED, TARGETS, [5], catalogue_size=10, train_items=np.arange(1, 11))
    assert result["cold_users"] == 0
    assert "cold" not in result
