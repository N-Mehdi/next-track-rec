import numpy as np
import pytest

from nexttrack.eval.baselines import decayed_scores, history_ranking, top_items


def test_decayed_scores():
    unique, scores = decayed_scores(np.array([5, 7, 5]), np.array([0, 7200, 7200]), hours=2)
    assert unique.tolist() == [5, 7]
    assert scores.tolist() == pytest.approx([0.9 + 1.0, 1.0])
    _, counts = decayed_scores(np.array([5, 7, 5]), np.array([0, 7200, 7200]), hours=None)
    assert counts.tolist() == [2.0, 1.0]


def test_top_items_pads():
    assert top_items(np.array([4, 8]), np.array([1.0, 3.0]), 3).tolist() == [8, 4, -1]


def test_history_prefers_recent_tracks():
    hour = 3600
    items = [np.array([1, 1, 2]), np.array([], dtype=np.int64)]
    times = [np.array([0, 1 * hour, 1000 * hour]), np.array([], dtype=np.int64)]
    # With a short period the single recent listen of track 2 beats two old listens of 1.
    assert history_ranking(items, times, 6, 2)[0].tolist() == [2, 1]
    assert history_ranking(items, times, None, 2)[0].tolist() == [1, 2]
    assert history_ranking(items, times, 6, 2)[1].tolist() == [-1, -1]
