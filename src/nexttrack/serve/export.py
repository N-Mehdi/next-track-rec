"""Write the artifacts of the service: SASRec encoder and reranker in ONNX, item tables.

Item tables (popularity, listen and like counts) are frozen at VAL_END, the state used to
score the test day. Audio embeddings stay in paths.audio (4 GB) and are mounted apart.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import torch

from nexttrack import constants
from nexttrack.config import parse_args
from nexttrack.data.sequences import read_index_map
from nexttrack.rerank.features import FEATURES, item_tables, load_sasrec
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


def main():
    config, _ = parse_args("Export the service artifacts")
    paths = config["paths"]
    prepared = Path(paths["prepared"])
    out_dir = Path(paths["artifacts"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model, max_len = load_sasrec(Path(config["candidates"]["sasrec_dir"]) / "model.pt")
    with torch.no_grad():
        export_encoder(model, max_len, out_dir / "sasrec_encoder.onnx")
        np.save(out_dir / "item_vectors.npy", model.items.weight.numpy())
    history = read_table(prepared / "history_test", ["item_ids", "timestamps"])
    items = flat_column(history, "item_ids")
    times = flat_column(history, "timestamps")
    tables = item_tables(config, prepared, items, times, constants.VAL_END)
    tables["idx_to_id"] = read_index_map(prepared / "item_map")
    np.savez(out_dir / "tables.npz", **tables)
    shutil.copy(Path(paths["outputs"]) / "rerank" / "reranker.onnx", out_dir / "reranker.onnx")
    settings = {"candidates": config["candidates"], "max_seq_len": max_len, "features": FEATURES}
    with open(out_dir / "settings.json", "w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2)
    print("artifacts written to", out_dir)


if __name__ == "__main__":
    main()
