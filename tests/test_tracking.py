import json

import pytest

pytest.importorskip("mlflow")
pytest.importorskip("sqlalchemy")

from nexttrack import tracking  # noqa: E402


@pytest.fixture
def config(tmp_path, monkeypatch):
    # MLflow writes run artifacts under the working directory with a sqlite store.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    return {
        "tracking": {
            "uri": "sqlite:///%s" % (tmp_path / "mlflow.db"),
            "model_name": "test-reranker",
            "hf_repo": "someone/test-reranker",
            "hf_private": True,
        },
        "paths": {"prepared": str(tmp_path / "prepared")},
        "candidates": {"sasrec": 100, "audio": 0, "sasrec_dir": str(tmp_path / "sasrec")},
        "rerank": {"num_leaves": 63, "learning_rate": 0.05},
        "split": {"shift_days": 0},
    }


def model_dir(tmp_path, content):
    folder = tmp_path / ("model_" + content)
    folder.mkdir()
    (folder / "reranker.onnx").write_text(content)
    (folder / "reranker.txt").write_text(content)
    return folder


REPORT = {
    "training": {"rounds": 12, "cv_ndcg@10": 0.3},
    "onnx_rows_differing": 0,
    "features": ["a", "b"],
    "test": {
        "all": {"ndcg@10": 0.25, "recall@10": 0.2},
        "cold_strict": {"metrics": {"recall@10": 0.0, "recall@100": 0.01, "ndcg@10": 0.0}},
    },
}


def test_versions_aliases_and_download(config, tmp_path):
    first = tracking.log_reranker(config, REPORT, model_dir(tmp_path, "one"))
    second = tracking.log_reranker(config, REPORT, model_dir(tmp_path, "two"))
    assert (first, second) == (1, 2)
    tracking.set_alias(config, "staging", second)
    folder, version = tracking.download(config, "staging", tmp_path / "out")
    assert version == 2
    assert (folder / "reranker.onnx").read_text() == "two"
    settings = json.loads((folder / "settings.json").read_text())
    assert settings["features"] == ["a", "b"]
    run = tracking.connect(config).search_runs([tracking.experiment(tracking.connect(config), "reranker")])[0]
    assert run.data.metrics["test_ndcg_at_10"] == pytest.approx(0.25)
    assert run.data.tags["variant"] == "no_audio"


def test_promote_publishes_the_production_version(config, tmp_path, monkeypatch):
    tracking.log_reranker(config, REPORT, model_dir(tmp_path, "one"))
    published = []

    def fake_publish(config, folder, version):
        published.append((version, (folder / "reranker.onnx").read_text()))

    monkeypatch.setattr(tracking, "publish", fake_publish)
    tracking.promote(config, 1)
    assert published == [(1, "one")]
    _, version = tracking.download(config, "production", tmp_path / "prod")
    assert version == 1


def test_sasrec_import_is_idempotent(config, tmp_path):
    report = {
        "config": {"run_name": "a", "embedding_dim": 128},
        "device": "Tesla T4",
        "best_epoch": 2,
        "train_seconds": 10.0,
        "epochs": [
            {"epoch": 1, "loss": 3.0, "val_ndcg@10": 0.1},
            {"epoch": 2, "loss": 2.0, "val_ndcg@10": 0.2},
        ],
        "validation": {"all": {"ndcg@10": 0.2}},
        "test": {"all": {"ndcg@10": 0.19}},
    }
    path = tmp_path / "sasrec_a.json"
    path.write_text(json.dumps(report))
    assert tracking.import_sasrec(config, path) is not None
    assert tracking.import_sasrec(config, path) is None


def test_metric_and_param_names():
    assert tracking.metric_name("ndcg@10") == "ndcg_at_10"
    assert tracking.flat_params("c.", {"a": 1, "b": {"x": 2}}, {}) == {"c.a": 1, "c.b.x": 2}
