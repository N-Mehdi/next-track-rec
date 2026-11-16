import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
lgb = pytest.importorskip("lightgbm")
pytest.importorskip("fastapi")
pytest.importorskip("onnxruntime")
pytest.importorskip("faiss")
pytest.importorskip("prometheus_client")

from fastapi.testclient import TestClient  # noqa: E402

from nexttrack.models.sasrec import SASRec  # noqa: E402
from nexttrack.rerank.features import AUDIO_FEATURES, FEATURES  # noqa: E402
from nexttrack.rerank.train_reranker import export_onnx  # noqa: E402
from nexttrack.serve.app import Recommender  # noqa: E402
from nexttrack.serve.export import export_encoder  # noqa: E402

TRAIN_ITEMS = np.array([10, 11, 12, 13, 14])


def write_retrieval(folder, name="run_a"):
    folder.mkdir()
    torch.manual_seed(0)
    model = SASRec(len(TRAIN_ITEMS), 4, 8, 1, 2, 0.0).eval()
    with torch.no_grad():
        export_encoder(model, 4, folder / "sasrec_encoder.onnx")
        np.save(folder / "item_vectors.npy", model.items.weight.numpy())
    tables = {
        "map_ids": TRAIN_ITEMS,
        "map_idxs": np.arange(1, 6),
        "idx_to_id": np.concatenate([[-1], TRAIN_ITEMS]),
        "pop_items": np.array([10, 11]),
        "pop_scores": np.array([2.0, 1.0]),
        "pop_top": np.array([10]),
        "count_items": np.array([10, 11, 20]),
        "counts": np.array([5, 3, 1]),
        "like_items": np.array([11]),
        "like_counts": np.array([1]),
        "artist_items": np.array([10, 11, 20]),
        "artists": np.array([1, 1, 2]),
        "album_items": np.array([10, 12]),
        "albums": np.array([7, 8]),
    }
    np.savez(folder / "tables.npz", **tables)
    with open(folder / "retrieval.json", "w", encoding="utf-8") as handle:
        json.dump({"name": name, "max_seq_len": 4, "cutoff": 1000}, handle)
    return folder


def write_reranker(folder, audio, version):
    folder.mkdir()
    names = []
    for name in FEATURES:
        if audio or name not in AUDIO_FEATURES:
            names.append(name)
    x = np.random.default_rng(0).normal(size=(100, len(names))).astype(np.float32)
    data = lgb.Dataset(x, (x[:, 0] > 0).astype(np.float32), group=np.full(10, 10))
    booster = lgb.train({"objective": "lambdarank", "verbose": -1, "min_data_in_leaf": 5}, data, 5)
    booster.save_model(str(folder / "reranker.txt"))
    export_onnx(folder / "reranker.txt", x, folder / "reranker.onnx")
    quotas = {"sasrec": 3, "history": 2, "audio": 1 if audio else 0, "decay_pop": 1, "total": 5}
    quotas.update({"decay_pop_hours": 2, "history_hours": 24, "audio_history_items": 2})
    settings = {
        "candidates": quotas,
        "rerank": {"audio_features": audio},
        "features": names,
        "retrieval": "run_a",
    }
    with open(folder / "settings.json", "w", encoding="utf-8") as handle:
        json.dump(settings, handle)
    with open(folder / "version.json", "w", encoding="utf-8") as handle:
        json.dump({"model": "test", "version": version}, handle)
    return folder


def write_audio(folder):
    folder.mkdir()
    vectors = np.random.default_rng(1).normal(size=(7, 4)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    np.save(folder / "item_ids.npy", np.array([10, 11, 12, 13, 14, 20, 21], dtype=np.uint32))
    np.save(folder / "vectors.npy", vectors)
    return folder


HISTORY = [
    {"item_id": 12, "timestamp": 100},
    {"item_id": 20, "timestamp": 300},
    {"item_id": 10, "timestamp": 200},
]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("serve")
    patch = pytest.MonkeyPatch()
    patch.setenv("NEXTTRACK_RERANKER", str(write_reranker(root / "reranker", False, 3)))
    patch.setenv("NEXTTRACK_RETRIEVAL", str(write_retrieval(root / "retrieval")))
    patch.setenv("NEXTTRACK_COMMIT", "abc123")
    from nexttrack.serve.app import app

    with TestClient(app) as test_client:
        yield test_client
    patch.undo()


def test_health_reports_versions(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "model_version": 3, "retrieval": "run_a", "commit": "abc123"}


def test_recommend_describes_tracks(client):
    body = client.post("/recommend", json={"history": HISTORY, "liked": [11], "k": 3}).json()
    assert body["model_version"] == 3
    items = []
    for track in body["tracks"]:
        items.append(track["item_id"])
        # Without audio source the candidates are training or history tracks.
        assert track["item_id"] in [10, 11, 12, 13, 14, 20]
        assert set(track["sources"]) <= {"sasrec", "history", "decay_pop"}
        assert len(track["sources"]) > 0
        assert track["already_listened"] == (track["item_id"] in [10, 12, 20])
        assert track["label"].startswith("track %d, artist " % track["item_id"])
    assert len(set(items)) == 3
    by_item = {}
    for track in body["tracks"]:
        by_item[track["item_id"]] = track
    if 10 in by_item:
        assert by_item[10]["artist_id"] == 1 and by_item[10]["album_id"] == 7
        assert by_item[10]["listens"] == 5 and by_item[10]["user_listens"] == 1


def test_metrics_count_requests(client):
    client.post("/recommend", json={"history": HISTORY, "k": 2})
    client.post("/recommend", json={"history": [], "k": 0})
    text = client.get("/metrics/").text
    assert 'nexttrack_requests_total{status="200"}' in text
    assert 'nexttrack_requests_total{status="422"}' in text
    assert "nexttrack_request_seconds_bucket" in text
    assert 'nexttrack_model_info{retrieval="run_a",version="3"}' in text


def test_empty_history_and_invalid_k(client):
    assert len(client.post("/recommend", json={"history": [], "k": 2}).json()["tracks"]) == 2
    assert client.post("/recommend", json={"history": [], "k": 0}).status_code == 422


def test_audio_variant_and_bundle_check(tmp_path):
    retrieval = write_retrieval(tmp_path / "retrieval")
    model = Recommender(
        write_reranker(tmp_path / "audio", True, None), retrieval, write_audio(tmp_path / "vectors")
    )
    tracks = model.recommend([12, 20, 10], [100, 300, 200], [], [], 4)
    assert len(tracks) == 4
    with pytest.raises(RuntimeError):
        Recommender(write_reranker(tmp_path / "other", False, 1), write_retrieval(tmp_path / "b", "run_b"))
