"""One command for the offline pipeline, with a manifest of data, code and environment.

  python -m nexttrack.pipeline [--dry-run] [--force STAGE ...]

Stages: download, split (Spark), baselines, audio_knn, sasrec (trained on Kaggle, only
checked), reranker (features, training, evaluation), export. A stage runs when its inputs,
its code (the nexttrack modules it imports) or its config sections differ from
reports/manifest.json, or when an output is missing or changed. Spark folders are hashed
as tables sorted on their scalar columns, so the digest ignores file layout and row order;
other files with sha256, cached by size and modification time in .cache/fingerprints.json.
"""

import ast
import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nexttrack import constants
from nexttrack.config import parse_args, production_config
from nexttrack.tables import write_json

SRC = Path(__file__).resolve().parent.parent
CACHE = Path(".cache/fingerprints.json")
PACKAGES = ["numpy", "pyarrow", "pyspark", "torch", "faiss-cpu", "lightgbm", "onnxmltools", "onnxruntime"]


def stage(name, module, io, sections, clean=False, production=False):
    """clean: delete the output files before running (caches that are reused when present).
    production: run with the overrides of the production section of the config."""
    if module is not None:
        module = "nexttrack." + module
    return {
        "name": name,
        "module": module,
        "inputs": io[0],
        "outputs": io[1],
        "sections": sections,
        "clean": clean,
        "production": production,
    }


def stages(config):
    paths = config["paths"]
    raw = Path(paths["raw"])
    data = Path(paths["prepared"])
    reports = Path(paths["reports"])
    flat = []
    for name in constants.FLAT_FILES:
        flat.append(raw / name)
    embeddings = raw / constants.EMBEDDINGS_FILE
    artists = raw / constants.ARTIST_FILE
    groups = [artists, raw / constants.ALBUM_FILE]
    audio = [Path(paths["audio"]) / "item_ids.npy", Path(paths["audio"]) / "vectors.npy"]
    sasrec = Path(config["candidates"]["sasrec_dir"])
    knn = Path(paths["outputs"]) / "audio_knn"
    offline = config["rerank"]["name"]
    deployed = production_config(config)["rerank"]["name"]
    download = [], flat + [embeddings] + groups
    knn_io = [embeddings, data], audio + [knn, reports / "audio_knn.json"]
    rerank_io = (
        [data, sasrec, knn] + groups + audio,
        [Path(paths["outputs"]) / offline, reports / (offline + ".json")],
    )
    deployed_io = [data, sasrec] + groups, [Path(paths["outputs"]) / deployed, reports / (deployed + ".json")]
    export_io = [data, sasrec] + groups, [Path(paths["retrieval"])]
    sections = ["split", "candidates", "rerank", "eval"]
    return [
        stage("download", "prep.download", download, ["data"]),
        stage("split", "prep.split", (flat + [embeddings], [data]), ["data", "spark", "split"]),
        stage("baselines", "eval.baselines", ([data], [reports / "baselines.json"]), ["candidates", "eval"]),
        stage("audio_knn", "retrieval.audio_knn", knn_io, ["candidates", "eval"], clean=True),
        # Trained on Kaggle: only checked, never run here.
        stage("sasrec", None, ([data], [sasrec]), ["sasrec"]),
        stage("reranker", "rerank.train_reranker", rerank_io, sections),
        # The deployed variant, without audio; its versions go to the MLflow registry.
        stage("reranker_production", "rerank.train_reranker", deployed_io, sections, production=True),
        stage("export", "serve.export", export_io, ["split", "candidates", "production"]),
    ]


def cached(path, cache, files, compute):
    """compute(path) unless the files under it kept their size and modification time."""
    stamp = []
    for name in files:
        status = name.stat()
        stamp.append([name.name, status.st_size, status.st_mtime_ns])
    entry = cache.get(str(path))
    if entry is None or entry["stamp"] != stamp:
        entry = {"stamp": stamp, "sha256": compute(path)}
        cache[str(path)] = entry
    return entry["sha256"]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        block = handle.read(1 << 24)
        while block:
            digest.update(block)
            block = handle.read(1 << 24)
    return digest.hexdigest()


def table_sha256(folder):
    """Content of a Spark output folder, independent of file layout and row order."""
    table = pq.read_table(str(folder))
    keys = []
    for field in table.schema:
        if not pa.types.is_list(field.type):
            keys.append((field.name, "ascending"))
    table = table.sort_by(keys).combine_chunks().replace_schema_metadata(None)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue()).hexdigest()


def fingerprint(path, cache):
    """Digest of a file or folder; None when it does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    if path.is_file():
        return cached(path, cache, [path], file_sha256)
    if (path / "_SUCCESS").exists():
        return cached(path, cache, sorted(path.glob("part-*")), table_sha256)
    digest = hashlib.sha256()
    for child in sorted(path.iterdir()):
        if not child.name.startswith("."):
            digest.update((child.name + str(fingerprint(child, cache))).encode())
    return digest.hexdigest()


def code_sha256(module):
    """The module and every nexttrack module it imports, directly or not."""
    seen = {}
    todo = [module, "nexttrack.config"]
    while todo:
        name = todo.pop()
        base = SRC.joinpath(*name.split("."))
        path = base.with_suffix(".py") if base.with_suffix(".py").exists() else base / "__init__.py"
        if name in seen or not path.exists():
            continue
        seen[name] = path.read_bytes()
        for node in ast.walk(ast.parse(seen[name])):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("nexttrack"):
                todo.append(node.module)
                for alias in node.names:
                    todo.append(node.module + "." + alias.name)
    digest = hashlib.sha256()
    for name in sorted(seen):
        digest.update(name.encode() + seen[name])
    return digest.hexdigest()


def state(item, config, cache):
    """What a stage depends on: code, config sections and input fingerprints."""
    if item["production"]:
        config = production_config(config)
    sections = {}
    for name in item["sections"]:
        sections[name] = config.get(name)
    inputs = {}
    for path in item["inputs"]:
        inputs[str(path)] = fingerprint(path, cache)
    code = code_sha256(item["module"]) if item["module"] else None
    text = json.dumps(sections, sort_keys=True).encode()
    return {"code": code, "config": hashlib.sha256(text).hexdigest(), "inputs": inputs}


def outdated(item, current, entry, cache):
    """Reason to run the stage, or None when it is up to date."""
    if entry is None:
        return "not in the manifest"
    for key in ["code", "config", "inputs"]:
        if entry[key] != current[key]:
            return key + " changed"
    for path in item["outputs"]:
        if fingerprint(path, cache) != entry["outputs"].get(str(path)):
            return "output %s missing or changed" % path
    return None


def run_stage(item, args, current, entry):
    if item["module"] is None:
        if not item["outputs"][0].exists() or (entry is not None and entry["inputs"] != current["inputs"]):
            sys.exit(
                "SASRec outputs missing or trained on other prepared data: run notebooks/kaggle/02_sasrec "
                "and download its outputs into %s" % item["outputs"][0]
            )
        return
    for path in item["outputs"]:
        # The audio cache is reused when present: remove it so it follows the embeddings.
        if item["clean"] and path.is_file():
            path.unlink()
    command = [sys.executable, "-m", item["module"], "--config", args.config]
    overrides = list(args.set)
    if item["production"]:
        overrides += args.production
    if overrides:
        command += ["--set"] + overrides
    subprocess.run(command, check=True)


def environment():
    versions = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "src", "configs"], capture_output=True, text=True)
    return {"packages": versions, "git_commit": commit, "code_changed_since_commit": bool(dirty.stdout)}


def read_json(path, default):
    if not Path(path).exists():
        return default
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def add_arguments(parser):
    parser.add_argument("--force", nargs="*", default=[])
    parser.add_argument("--dry-run", action="store_true")


def main():
    config, args = parse_args("Offline pipeline", add_arguments)
    manifest_path = Path(config["paths"]["reports"]) / "manifest.json"
    manifest = read_json(manifest_path, {"stages": {}})
    cache = read_json(CACHE, {})
    args.production = config["production"]["overrides"]
    for item in stages(config):
        entry = manifest["stages"].get(item["name"])
        current = state(item, config, cache)
        reason = "forced" if item["name"] in args.force else outdated(item, current, entry, cache)
        print("%-10s %s" % (item["name"], reason or "up to date"), flush=True)
        if reason is None or args.dry_run:
            continue
        started = time.time()
        run_stage(item, args, current, entry)
        current["seconds"] = round(time.time() - started, 1)
        current["outputs"] = {}
        for path in item["outputs"]:
            current["outputs"][str(path)] = fingerprint(path, cache)
        manifest["stages"][item["name"]] = current
        manifest["environment"] = environment()
        write_json(manifest, manifest_path)
        write_json(cache, CACHE)


if __name__ == "__main__":
    main()
