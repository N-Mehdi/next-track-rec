import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("faiss")

from nexttrack.retrieval import audio_knn  # noqa: E402


def unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    return vector / np.linalg.norm(vector)


@pytest.fixture
def audio(monkeypatch):
    monkeypatch.setattr(audio_knn, "EMBEDDING_DIM", 2)
    item_ids = np.array([1, 3, 5, 7, 9], dtype=np.uint32)
    vectors = np.stack([unit([1, 0]), unit([0, 1]), unit([1, 0.1]), unit([0.1, 1]), unit([-1, 0])])
    return item_ids, vectors


def test_load_audio_sorts_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_knn, "EMBEDDING_DIM", 2)
    # One row per batch: rows land at their sorted position across batches.
    monkeypatch.setattr(audio_knn, "BATCH_ROWS", 1)
    table = pa.table(
        {
            "item_id": pa.array([5, 1], pa.uint32()),
            "embed": pa.array([[5.0, 0.0], [0.0, 1.0]], pa.large_list(pa.float64())),
            "normalized_embed": pa.array([[1.0, 0.0], [0.0, 1.0]], pa.large_list(pa.float64())),
        }
    )
    pq.write_table(table, tmp_path / "embeddings.parquet")
    item_ids, vectors = audio_knn.load_audio(tmp_path, tmp_path / "cache")
    assert item_ids.tolist() == [1, 5]
    assert vectors.dtype == np.float32
    assert vectors.tolist() == [[0.0, 1.0], [1.0, 0.0]]
    cached_ids, cached_vectors = audio_knn.load_audio(tmp_path / "missing", tmp_path / "cache")
    assert cached_ids.tolist() == [1, 5]
    assert np.asarray(cached_vectors).tolist() == [[0.0, 1.0], [1.0, 0.0]]


def test_lookup_marks_missing(audio):
    item_ids, _ = audio
    assert audio_knn.lookup(item_ids, np.array([3, 4, 9, 10], dtype=np.uint32)).tolist() == [1, -1, 4, -1]


def test_queries_average_last_tracks_with_audio(audio):
    item_ids, vectors = audio
    histories = [np.array([3, 1, 2]), np.array([2, 4]), np.array([7, 3, 1])]
    queries = audio_knn.user_queries(histories, item_ids, vectors, 2)
    # Track 2 has no embedding: the last two tracks with audio of user 0 are 3 and 1.
    assert queries[0] == pytest.approx(unit([1, 1]), abs=1e-6)
    assert not queries[1].any()
    assert queries[2] == pytest.approx(unit([1, 1]), abs=1e-6)


def test_search_excludes_seen_tracks(audio):
    item_ids, vectors = audio
    index, pool_ids = audio_knn.build_index(item_ids, vectors, np.array([1, 3]))
    assert pool_ids.tolist() == [5, 7, 9]
    queries = np.stack([unit([1, 0]), np.zeros(2, dtype=np.float32)])
    ranked, _ = audio_knn.search(index, pool_ids, queries, 4)
    assert ranked[0].tolist() == [5, 7, 9, -1]
    assert ranked[1].tolist() == [-1, -1, -1, -1]
