"""MLflow tracking and model registry, on the local server of docker-compose.yml.

Experiments: "sasrec" (runs imported from the Kaggle reports) and "reranker" (one run per
training, logged by rerank.train_reranker). Each reranker run registers a version of the
model tracking.model_name. The aliases staging and production point to versions; the
registry is the source of truth. Promotion to production also publishes the model files
to the Hugging Face repository tracking.hf_repo, tagged with the MLflow version, which the
deployed service downloads at start.

  python -m nexttrack.tracking import-sasrec reports/sasrec_a_d128.json ...
  python -m nexttrack.tracking log-reranker            log the last training (outputs/rerank)
  python -m nexttrack.tracking alias staging 3         point an alias to a version
  python -m nexttrack.tracking promote 3 [--no-publish]
  python -m nexttrack.tracking download production DIR
"""

import json
import os
import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from nexttrack import pipeline
from nexttrack.config import parse_args


def connect(config):
    mlflow.set_tracking_uri(config["tracking"]["uri"])
    return MlflowClient()


def experiment(client, name):
    found = client.get_experiment_by_name(name)
    if found is not None:
        return found.experiment_id
    return client.create_experiment(name)


def metric_name(name):
    # MLflow metric names cannot hold "@".
    return name.replace("@", "_at_")


def flat_params(prefix, section, out):
    """section {"a": {"b": 1}} gives out["prefix.a.b"] = 1."""
    for key in section:
        if isinstance(section[key], dict):
            flat_params(prefix + key + ".", section[key], out)
        else:
            out[prefix + key] = section[key]
    return out


def data_tags(paths):
    """Fingerprints of the data a run used, from the pipeline cache when possible."""
    cache = pipeline.read_json(pipeline.CACHE, {})
    tags = {}
    for path in paths:
        tags["data." + Path(path).name] = str(pipeline.fingerprint(path, cache))
    return tags


def model_settings(config, features):
    """What the service needs next to the reranker: candidate quotas and feature order."""
    return {"candidates": config["candidates"], "features": features, "split": config["split"]}


def log_reranker(config, report, model_dir, client=None):
    """One run in the reranker experiment and a new registered model version; returns it."""
    client = client or connect(config)
    params = flat_params("candidates.", config["candidates"], {})
    flat_params("rerank.", config["rerank"], params)
    params["split.shift_days"] = config["split"]["shift_days"]
    params["rounds"] = report["training"]["rounds"]
    metrics = {
        "cv_ndcg_at_10": report["training"]["cv_ndcg@10"],
        "onnx_rows_differing": report["onnx_rows_differing"],
    }
    for name in report["test"]["all"]:
        metrics["test_" + metric_name(name)] = report["test"]["all"][name]
    for name in ["recall@10", "recall@100", "ndcg@10"]:
        metrics["test_cold_strict_" + metric_name(name)] = report["test"]["cold_strict"]["metrics"][name]
    tags = data_tags([config["paths"]["prepared"], config["candidates"]["sasrec_dir"]])
    tags["variant"] = "audio" if config["candidates"]["audio"] > 0 else "no_audio"
    with mlflow.start_run(experiment_id=experiment(client, "reranker"), tags=tags) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as folder:
            with open(Path(folder) / "reranker.json", "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
            with open(Path(folder) / "settings.json", "w", encoding="utf-8") as handle:
                json.dump(model_settings(config, report["features"]), handle, indent=2)
            mlflow.log_artifact(str(Path(folder) / "reranker.json"))
            mlflow.log_artifact(str(Path(folder) / "settings.json"), artifact_path="model")
        mlflow.log_artifact(str(Path(model_dir) / "reranker.onnx"), artifact_path="model")
        mlflow.log_artifact(str(Path(model_dir) / "reranker.txt"), artifact_path="model")
    name = config["tracking"]["model_name"]
    if not client.search_registered_models("name='%s'" % name):
        client.create_registered_model(name)
    version = client.create_model_version(name, "runs:/%s/model" % run.info.run_id, run_id=run.info.run_id)
    print("reranker run", run.info.run_id, "registered as", name, "version", version.version)
    return int(version.version)


def import_sasrec(config, report_path, client=None):
    """A SASRec report from Kaggle as a run of the sasrec experiment; skipped when already there."""
    client = client or connect(config)
    exp_id = experiment(client, "sasrec")
    source = Path(report_path).name
    if client.search_runs([exp_id], "tags.source = '%s'" % source):
        print(source, "already imported")
        return None
    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)
    tags = {"source": source, "device": report["device"], "trained_on": "kaggle"}
    with mlflow.start_run(experiment_id=exp_id, run_name=report["config"]["run_name"], tags=tags) as run:
        mlflow.log_params(flat_params("sasrec.", report["config"], {}))
        mlflow.log_params(
            {"best_epoch": report["best_epoch"], "train_seconds": round(report["train_seconds"])}
        )
        for entry in report["epochs"]:
            mlflow.log_metrics(
                {"loss": entry["loss"], "val_ndcg_at_10": entry["val_ndcg@10"]}, step=entry["epoch"]
            )
        for split in ["validation", "test"]:
            for name in report[split]["all"]:
                mlflow.log_metric(split + "_" + metric_name(name), report[split]["all"][name])
        mlflow.log_artifact(str(report_path))
    return run.info.run_id


def set_alias(config, alias, version, client=None):
    client = client or connect(config)
    client.set_registered_model_alias(config["tracking"]["model_name"], alias, str(version))
    print(config["tracking"]["model_name"], alias, "->", version)


def download(config, alias, destination, client=None):
    """Files of the version an alias points to; returns the folder and the version."""
    client = client or connect(config)
    name = config["tracking"]["model_name"]
    version = client.get_model_version_by_alias(name, alias).version
    folder = mlflow.artifacts.download_artifacts("models:/%s@%s" % (name, alias), dst_path=str(destination))
    return Path(folder), int(version)


def publish(config, folder, version):
    """Upload the model files to the Hugging Face repository and tag them with the version."""
    from huggingface_hub import HfApi

    # The token comes from the environment (HF_TOKEN) and is never printed.
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    repo = config["tracking"]["hf_repo"]
    api.create_repo(repo, private=config["tracking"]["hf_private"], exist_ok=True)
    with open(Path(folder) / "version.json", "w", encoding="utf-8") as handle:
        json.dump({"model": config["tracking"]["model_name"], "version": version}, handle)
    commit = api.upload_folder(
        repo_id=repo, folder_path=str(folder), commit_message="MLflow version %d" % version
    )
    api.create_tag(repo, tag="v%d" % version, revision=commit.oid, exist_ok=True)
    print("published version", version, "to", repo, "tag v%d" % version)


def promote(config, version, publish_model=True, client=None):
    """Point production to the version, then publish its files for the deployed service."""
    client = client or connect(config)
    set_alias(config, "production", version, client)
    if publish_model:
        with tempfile.TemporaryDirectory() as destination:
            folder, _ = download(config, "production", destination, client)
            publish(config, folder, version)


def add_arguments(parser):
    parser.add_argument("command", choices=["import-sasrec", "log-reranker", "alias", "promote", "download"])
    parser.add_argument("values", nargs="*")
    parser.add_argument("--no-publish", action="store_true")


def main():
    config, args = parse_args("MLflow tracking and registry", add_arguments)
    if args.command == "import-sasrec":
        for path in args.values:
            import_sasrec(config, path)
    elif args.command == "log-reranker":
        with open(Path(config["paths"]["reports"]) / "reranker.json", encoding="utf-8") as handle:
            report = json.load(handle)
        log_reranker(config, report, Path(config["paths"]["outputs"]) / "rerank")
    elif args.command == "alias":
        set_alias(config, args.values[0], int(args.values[1]))
    elif args.command == "promote":
        promote(config, int(args.values[0]), not args.no_publish)
    else:
        folder, version = download(config, args.values[0], args.values[1])
        print("version", version, "in", folder)


if __name__ == "__main__":
    main()
