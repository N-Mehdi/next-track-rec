from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nexttrack import pipeline


def spark_folder(folder, parts):
    """A folder laid out like a Spark output: part files and a _SUCCESS marker."""
    folder.mkdir()
    for i in range(len(parts)):
        uids, items = parts[i]
        table = pa.table(
            {"uid": pa.array(uids, pa.int64()), "item_ids": pa.array(items, pa.list_(pa.int64()))}
        )
        pq.write_table(table, folder / ("part-%05d.parquet" % i))
    (folder / "_SUCCESS").touch()
    return folder


def test_table_digest_ignores_layout_and_row_order(tmp_path):
    one = spark_folder(tmp_path / "one", [([1, 2, 3], [[1], [2, 3], []])])
    two = spark_folder(tmp_path / "two", [([3], [[]]), ([2, 1], [[2, 3], [1]])])
    other = spark_folder(tmp_path / "other", [([1, 2, 3], [[1], [3, 2], []])])
    assert pipeline.fingerprint(one, {}) == pipeline.fingerprint(two, {})
    assert pipeline.fingerprint(one, {}) != pipeline.fingerprint(other, {})


def test_file_digest_follows_content(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"abc")
    cache = {}
    first = pipeline.fingerprint(path, cache)
    assert pipeline.fingerprint(path, cache) == first
    path.write_bytes(b"abcd")
    assert pipeline.fingerprint(path, cache) != first
    assert pipeline.fingerprint(tmp_path / "missing", cache) is None


def test_code_digest_follows_imports(tmp_path, monkeypatch):
    package = tmp_path / "nexttrack"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "config.py").write_text("")
    (package / "a.py").write_text("from nexttrack.b import value\n")
    (package / "b.py").write_text("value = 1\n")
    (package / "c.py").write_text("other = 1\n")
    monkeypatch.setattr(pipeline, "SRC", tmp_path)
    before = pipeline.code_sha256("nexttrack.a")
    (package / "c.py").write_text("other = 2\n")
    assert pipeline.code_sha256("nexttrack.a") == before
    (package / "b.py").write_text("value = 2\n")
    assert pipeline.code_sha256("nexttrack.a") != before


def test_stage_reruns_when_input_or_output_changes(tmp_path):
    source = tmp_path / "in.txt"
    result = tmp_path / "out.txt"
    source.write_text("1")
    result.write_text("x")
    item = {
        "name": "s",
        "module": "nexttrack.config",
        "inputs": [source],
        "outputs": [result],
        "sections": ["eval"],
        "production": False,
    }
    config = {"eval": {"k_values": [10]}}
    cache = {}
    entry = pipeline.state(item, config, cache)
    entry["outputs"] = {str(result): pipeline.fingerprint(result, cache)}
    assert pipeline.outdated(item, pipeline.state(item, config, cache), entry, cache) is None
    config["eval"]["k_values"] = [5]
    assert pipeline.outdated(item, pipeline.state(item, config, cache), entry, cache) == "config changed"
    config["eval"]["k_values"] = [10]
    source.write_text("2")
    assert pipeline.outdated(item, pipeline.state(item, config, cache), entry, cache) == "inputs changed"
    source.write_text("1")
    Path(result).unlink()
    assert "missing" in pipeline.outdated(item, pipeline.state(item, config, cache), entry, cache)


def test_stage_list_follows_the_config():
    from nexttrack.config import load_config

    names = []
    for item in pipeline.stages(load_config("configs/default.yaml")):
        names.append(item["name"])
    expected = ["download", "split", "baselines", "audio_knn", "sasrec", "reranker", "reranker_production"]
    assert names == expected + ["export"]


def test_production_stage_uses_the_production_overrides(tmp_path):
    from nexttrack.config import load_config

    # Paths in an empty folder: the test does not hash the real data.
    overrides = []
    for name in ["raw", "prepared", "audio", "outputs", "reports", "retrieval"]:
        overrides.append("paths.%s=%s" % (name, tmp_path / name))
    overrides.append("candidates.sasrec_dir=%s" % (tmp_path / "sasrec"))
    config = load_config("configs/default.yaml", overrides)
    by_name = {}
    for item in pipeline.stages(config):
        by_name[item["name"]] = item
    offline = pipeline.state(by_name["reranker"], config, {})
    deployed = pipeline.state(by_name["reranker_production"], config, {})
    # Same code, other configuration: the two variants are tracked separately.
    assert offline["code"] == deployed["code"]
    assert offline["config"] != deployed["config"]
    assert str(by_name["reranker_production"]["outputs"][0]).endswith("reranker_production")
