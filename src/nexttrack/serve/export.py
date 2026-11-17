"""Retrieval bundle of the service: SASRec encoder in ONNX, item vectors and item tables.

Item tables (popularity, listen and like counts, artist and album of each known track) are
frozen at VAL_END, the state used to score the test day. The reranker comes from the MLflow
registry (nexttrack.tracking promote). --publish uploads the bundle to tracking.retrieval_repo,
tagged with the SASRec run name.
"""

import json
from pathlib import Path

import numpy as np
import torch

from nexttrack import constants
from nexttrack.config import parse_args, production_config
from nexttrack.data.sequences import read_index_map
from nexttrack.rerank.features import item_tables, load_sasrec
from nexttrack.tables import flat_column, read_table


class Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        return self.model.user_vectors(inputs)


def export_encoder(model, max_len, path):
    inputs = torch.zeros((2, max_len), dtype=torch.long)
    inputs[0, -3:] = torch.tensor([1, 2, 3])
    torch.onnx.export(
        Encoder(model),
        (inputs,),
        str(path),
        input_names=["inputs"],
        output_names=["user"],
        dynamic_axes={"inputs": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )


def retrieval_name(config):
    """The bundle is named after the SASRec run it comes from."""
    return Path(config["candidates"]["sasrec_dir"]).name


def add_arguments(parser):
    parser.add_argument("--publish", action="store_true")


def main():
    config, args = parse_args("Export the retrieval bundle of the service", add_arguments)
    config = production_config(config)
    paths = config["paths"]
    prepared = Path(paths["prepared"])
    out_dir = Path(paths["retrieval"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model, max_len = load_sasrec(Path(config["candidates"]["sasrec_dir"]) / "model.pt")
    with torch.no_grad():
        export_encoder(model, max_len, out_dir / "sasrec_encoder.onnx")
        np.save(out_dir / "item_vectors.npy", model.items.weight.numpy())
    history = read_table(prepared / "history_test", ["item_ids", "timestamps"])
    cutoff = constants.split_bounds(config["split"]["shift_days"])["val_end"]
    items = flat_column(history, "item_ids")
    # The deployed variant has no audio source: its candidates are training or history tracks.
    tables = item_tables(config, prepared, items, flat_column(history, "timestamps"), cutoff, known_only=True)
    tables["idx_to_id"] = read_index_map(prepared / "item_map")
    np.savez(out_dir / "tables.npz", **tables)
    info = {"name": retrieval_name(config), "max_seq_len": max_len, "cutoff": cutoff}
    with open(out_dir / "retrieval.json", "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)
    print("retrieval bundle", info["name"], "written to", out_dir)
    if args.publish:
        from nexttrack.tracking import publish, set_revision

        repo = config["tracking"]["retrieval_repo"]
        description = (
            "Retrieval bundle of next-track-rec: SASRec encoder in ONNX (last 200 Listen+ tracks to a "
            "user vector), item embeddings and item tables (popularity, counts, artist and album) "
            "frozen at the end of the validation day."
        )
        private = config["tracking"]["hf_private"]
        publish(repo, out_dir, info["name"], "Retrieval bundle " + info["name"], private, description)
        set_revision(config["tracking"]["k8s_manifest"], "NEXTTRACK_RETRIEVAL_REVISION", info["name"])


if __name__ == "__main__":
    main()
