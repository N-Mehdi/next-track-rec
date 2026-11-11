import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
lgb = pytest.importorskip("lightgbm")
pytest.importorskip("fastapi")
pytest.importorskip("onnxruntime")
pytest.importorskip("faiss")

from fastapi.testclient import TestClient  # noqa: E402

from nexttrack.models.sasrec import SASRec  # noqa: E402
from nexttrack.rerank.features import FEATURES  # noqa: E402
from nexttrack.rerank.train_reranker import export_onnx  # noqa: E402
from nexttrack.serve.export import export_encoder  # noqa: E402

TRAIN_ITEMS = np.array([10, 11, 12, 13, 14])


def write_artifacts(root):
    artifacts = root / "artifacts"
    audio = root / "audio"
    artifacts.mkdir()
    audio.mkdir()
    torch.manual_seed(0)
    model = SASRec(len(TRAIN_ITEMS), 4, 8, 1, 2, 0.0).eval()
    with torch.no_grad():
        export_encoder(model, 4, artifacts / "sasrec_encoder.onnx")
        np.save(artifacts / "item_vectors.npy", model.items.weight.numpy())
    tables = {
        "map_ids": TRAIN_ITEMS,
        "map_idxs": np.arange(1, 6),
        "idx_to_id": np.concatenate([[-1], TRAIN_ITEMS]),
        "pop_items": np.array([10, 11]),
        "pop_scores": np.array([2.0, 1.0]),
        "pop_top": np.array([10]),
        "count_items": np.array([10, 11]),
        "counts": np.array([5, 3]),
        "like_items": np.array([11]),
        "like_counts": np.array([1]),
        "artist_items": np.array([10, 11, 20]),
        "artists": np.array([1, 1, 2]),
    }
    np.savez(artifacts / "tables.npz", **tables)
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(7, 4)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    np.save(audio / "item_ids.npy", np.array([10, 11, 12, 13, 14, 20, 21], dtype=np.uint32))
    np.save(audio / "vectors.npy", vectors)
    x = rng.normal(size=(100, len(FEATURES))).astype(np.float32)
    booster = lgb.train(
        {"objective": "lambdarank", "verbose": -1, "min_data_in_leaf": 5},
        lgb.Dataset(x, (x[:, 0] > 0).astype(np.float32), group=np.full(10, 10)),
        5,
    )
    booster.save_model(str(root / "reranker.txt"))
    export_onnx(root / "reranker.txt", x, artifacts / "reranker.onnx")
    quotas = {"sasrec": 3, "history": 2, "audio": 1, "decay_pop": 1, "total": 5}
    quotas.update({"decay_pop_hours": 2, "history_hours": 24, "audio_history_items": 2})
    with open(artifacts / "settings.json", "w", encoding="utf-8") as handle:
        json.dump({"candidates": quotas, "max_seq_len": 4, "features": FEATURES}, handle)
    return artifacts, audio


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    artifacts, audio = write_artifacts(tmp_path_factory.mktemp("serve"))
    monkeypatch_module.setenv("NEXTTRACK_ARTIFACTS", str(artifacts))
    monkeypatch_module.setenv("NEXTTRACK_AUDIO", str(audio))
    from nexttrack.serve.app import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def monkeypatch_module():
    patch = pytest.MonkeyPatch()
    yield patch
    patch.undo()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_recommend_returns_distinct_tracks(client):
    history = [
        {"item_id": 12, "timestamp": 100},
        {"item_id": 20, "timestamp": 300},
        {"item_id": 10, "timestamp": 200},
    ]
    body = client.post("/recommend", json={"history": history, "liked": [11], "k": 3}).json()
    items = []
    for track in body["tracks"]:
        items.append(track["item_id"])
    assert len(items) == 3
    assert len(set(items)) == 3
    # Training tracks or audio pool tracks (20 and 21, never seen in training).
    assert set(items) <= {10, 11, 12, 13, 14, 20, 21}
    scores = []
    for track in body["tracks"]:
        scores.append(track["score"])
    assert scores == sorted(scores, reverse=True)


def test_empty_history_and_invalid_k(client):
    assert len(client.post("/recommend", json={"history": [], "k": 2}).json()["tracks"]) == 2
    assert client.post("/recommend", json={"history": [], "k": 0}).status_code == 422
