import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("torch")

from nexttrack.rerank import features  # noqa: E402
from nexttrack.rerank.train_reranker import ranked_matrix, subset, user_folds  # noqa: E402


def test_merge_sources_keeps_first_occurrence_and_total():
    lists = [np.array([5, 3]), np.array([3, -1, 7]), np.array([9, 5]), np.array([11, 12])]
    assert features.merge_sources(lists, 10).tolist() == [5, 3, 7, 9, 11, 12]
    assert features.merge_sources(lists, 4).tolist() == [5, 3, 7, 9]


def write_events(folder, rows):
    folder.mkdir()
    table = pa.table(
        {
            "uid": pa.array(column(rows, 0), pa.int64()),
            "item_id": pa.array(column(rows, 1), pa.int64()),
            "timestamp": pa.array(column(rows, 2), pa.int64()),
        }
    )
    pq.write_table(table, folder / "part-0.parquet")


def column(rows, j):
    out = []
    for row in rows:
        out.append(row[j])
    return out


def test_net_state_uses_latest_event_before_cutoff(tmp_path):
    # uid 1: item 10 liked then unliked, item 20 liked; uid 2: item 10 unliked after cutoff.
    write_events(tmp_path / "likes", [(1, 10, 5), (1, 20, 6), (2, 10, 7)])
    write_events(tmp_path / "unlikes", [(1, 10, 8), (2, 10, 50)])
    keys = features.net_state(tmp_path, "likes", "unlikes", 20)
    assert keys.tolist() == [(1 << 32) | 20, (2 << 32) | 10]


def test_history_features():
    items = np.array([4, 2, 4, 3])
    times = np.array([100, 200, 300, 400])
    out = features.history_features(items, times, 1000, np.array([4, 2, 9]), 24)
    assert out["user_item_count"].tolist() == [2, 1, 0]
    assert out["user_item_count_recent"].tolist() == [2, 1, 0]
    assert out["hours_since_last"][:2] * 3600 == pytest.approx([700, 800])
    assert out["events_since_last"][:2].tolist() == [1, 2]
    assert np.isnan(out["hours_since_last"][2])
    assert out["history_score"][2] == 0
    assert out["history_score"][0] > out["history_score"][1]
    empty = features.history_features(np.zeros(0, dtype=np.int64), np.zeros(0), 10, np.array([1]), 24)
    assert empty["user_item_count"].tolist() == [0]


def test_lookup_values_with_empty_keys():
    values = features.lookup_values(np.zeros(0, dtype=np.int64), np.zeros(0), np.array([1, 2]), -1)
    assert values.tolist() == [-1, -1]


def make_data():
    return {
        "sizes": np.array([3, 2, 2]),
        "items": np.array([10, 11, 12, 20, 21, 30, 31]),
        "labels": np.array([0, 1, 0, 0, 0, 1, 1], dtype=np.float32),
        "features": np.arange(14, dtype=np.float32).reshape(7, 2),
    }


def test_ranked_matrix_sorts_and_pads():
    ranked, scores = ranked_matrix(make_data(), np.array([0.1, 0.9, 0.5, 1.0, 2.0, 0.0, 0.3]), 3)
    assert ranked.tolist() == [[11, 12, 10], [21, 20, -1], [31, 30, -1]]
    assert np.isnan(scores[1, 2])


def test_users_without_positive_are_dropped():
    data = make_data()
    folds = user_folds(data, 2, 0)
    assert len(folds) == 2
    assert sorted(np.concatenate(folds).tolist()) == [0, 2]
    x, y, groups = subset(data, np.array([0, 2]))
    assert groups.tolist() == [3, 2]
    assert y.tolist() == [0, 1, 0, 1, 1]
    assert x.shape == (5, 2)


def test_onnx_export_matches_saved_booster(tmp_path):
    lgb = pytest.importorskip("lightgbm")
    pytest.importorskip("onnxruntime")
    from nexttrack.rerank.train_reranker import export_onnx

    rng = np.random.default_rng(0)
    x = np.round(rng.normal(size=(400, len(features.FEATURES))), 3).astype(np.float32)
    x[::7, 3] = np.nan
    y = (x[:, 0] + x[:, 1] > 0).astype(np.float32)
    data = lgb.Dataset(x, y, group=np.full(20, 20))
    booster = lgb.train({"objective": "lambdarank", "verbose": -1, "min_data_in_leaf": 5}, data, 20)
    booster.save_model(str(tmp_path / "model.txt"))
    onnx_scores, saved_scores = export_onnx(tmp_path / "model.txt", x, tmp_path / "model.onnx")
    assert onnx_scores == pytest.approx(saved_scores, abs=1e-5)
    assert saved_scores == pytest.approx(booster.predict(x), abs=1e-9)


def test_reserve_audio_every_third_position():
    from nexttrack.rerank.train_reranker import reserve_audio

    ranked = np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
    audio = [np.array([9, 2, 8]), np.array([-1, -1])]
    out = reserve_audio(ranked, audio, 3, 6)
    # Audio track 2 is already placed: the next audio track 8 takes position 6.
    assert out[0].tolist() == [1, 2, 9, 3, 4, 8]
    assert out[1].tolist() == [1, 2, 3, 4, 5, 6]
