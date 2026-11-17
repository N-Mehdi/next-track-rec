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
            "k8s_manifest": str(tmp_path / "deployment.yaml"),
        },
        "paths": {"prepared": str(tmp_path / "prepared")},
        "candidates": {"sasrec": 100, "audio": 0, "sasrec_dir": str(tmp_path / "sasrec")},
        "rerank": {"num_leaves": 63, "audio_features": False, "name": "reranker_production"},
        "split": {"shift_days": 0},
        "production": {
            "overrides": [
                "candidates.audio=0",
                "rerank.audio_features=false",
                "rerank.name=reranker_production",
            ]
        },
    }


def model_dir(tmp_path, content):
    folder = tmp_path / ("model_" + content)
    folder.mkdir()
    (folder / "reranker.onnx").write_text(content)
    (folder / "reranker.txt").write_text(content)
    return folder


MANIFEST = """spec:
  containers:
    - env:
        - name: NEXTTRACK_RERANKER_REVISION
          value: "v0"
        - name: NEXTTRACK_RETRIEVAL_REVISION
          value: "run_a"
"""

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
    # Training again to the same ONNX file gives no new version.
    (tmp_path / "again").mkdir()
    (tmp_path / "again" / "reranker.onnx").write_text("two")
    (tmp_path / "again" / "reranker.txt").write_text("two")
    assert tracking.log_reranker(config, REPORT, tmp_path / "again") == 2
    tracking.set_alias(config, "staging", second)
    folder, version = tracking.download(config, "staging", tmp_path / "out")
    assert version == 2
    assert (folder / "reranker.onnx").read_text() == "two"
    settings = json.loads((folder / "settings.json").read_text())
    assert settings["features"] == ["a", "b"]
    assert settings["retrieval"] == "sasrec"
    run = tracking.connect(config).search_runs([tracking.experiment(tracking.connect(config), "reranker")])[0]
    assert run.data.metrics["test_ndcg_at_10"] == pytest.approx(0.25)
    assert run.data.tags["variant"] == "no_audio"


def test_promote_publishes_the_production_version(config, tmp_path, monkeypatch):
    tracking.log_reranker(config, REPORT, model_dir(tmp_path, "one"))
    published = []

    def fake_publish(repo, folder, tag, message, private, description):
        version = json.loads((folder / "version.json").read_text())["version"]
        published.append((repo, tag, version, (folder / "reranker.onnx").read_text(), private))

    monkeypatch.setattr(tracking, "publish", fake_publish)
    (tmp_path / "deployment.yaml").write_text(MANIFEST)
    tracking.promote(config, 1)
    # The new tag is written for the deployment, nothing else changes.
    assert (tmp_path / "deployment.yaml").read_text() == MANIFEST.replace('"v0"', '"v1"')
    assert published == [("someone/test-reranker", "v1", 1, "one", True)]
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


def test_offline_variant_is_not_registered_nor_promoted(config, tmp_path):
    offline = dict(config)
    offline["candidates"] = dict(config["candidates"], audio=30)
    offline["rerank"] = dict(config["rerank"], audio_features=True, name="reranker")
    assert tracking.log_reranker(offline, REPORT, model_dir(tmp_path, "audio")) is None
    client = tracking.connect(config)
    run = client.search_runs([tracking.experiment(client, "reranker")])[0]
    assert run.data.tags["variant"] == "audio"
    client.create_registered_model("test-reranker")
    client.create_model_version("test-reranker", "runs:/%s/model" % run.info.run_id, run_id=run.info.run_id)
    with pytest.raises(SystemExit):
        tracking.promote(config, 1, publish_model=False)


def test_set_revision_needs_the_variable(tmp_path):
    path = tmp_path / "deployment.yaml"
    path.write_text(MANIFEST)
    tracking.set_revision(path, "NEXTTRACK_RETRIEVAL_REVISION", "run_b")
    assert 'value: "run_b"' in path.read_text()
    assert 'value: "v0"' in path.read_text()
    with pytest.raises(SystemExit):
        tracking.set_revision(path, "MISSING", "x")
