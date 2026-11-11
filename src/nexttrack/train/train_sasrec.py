"""Train SASRec on the training period with early stopping on the validation day.

Validation inputs end at TRAIN_END, test inputs at VAL_END (state frozen before the test
day). Writes to paths.outputs/<run_name>: model.pt, report.json and the top
candidates.num_retrieved items per target user for both days (candidates_val.parquet,
candidates_test.parquet).
"""

import time
from pathlib import Path

import numpy as np
import torch

from nexttrack import constants
from nexttrack.config import parse_args
from nexttrack.data.sequences import align, last_inputs, make_windows, read_histories, read_index_map
from nexttrack.eval.metrics import evaluate
from nexttrack.models.sasrec import SASRec
from nexttrack.tables import read_targets, seen_item_sets, to_builtin, write_json, write_ranked


def pick_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast(device, fp16):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16)


@torch.no_grad()
def rank(model, inputs, k, batch_size, device, fp16):
    """Top k item_idx and scores over every training item, padding excluded."""
    model.eval()
    weights = model.items.weight
    top_idx = []
    top_scores = []
    for start in range(0, len(inputs), batch_size):
        batch = torch.from_numpy(inputs[start : start + batch_size]).to(device)
        with autocast(device, fp16):
            scores = model.user_vectors(batch) @ weights.T
        scores = scores.float()
        scores[:, 0] = float("-inf")
        best = scores.topk(k, dim=1)
        top_idx.append(best.indices.cpu().numpy())
        top_scores.append(best.values.cpu().numpy())
    return np.concatenate(top_idx), np.concatenate(top_scores)


def train_epoch(model, optimizer, scaler, windows, cfg, device, fp16):
    model.train()
    batch_size = cfg["batch_size"]
    order = torch.randperm(len(windows))
    total = 0.0
    steps = 0
    for start in range(0, len(windows) - batch_size + 1, batch_size):
        batch = windows[order[start : start + batch_size]].to(device, non_blocking=True)
        with autocast(device, fp16):
            loss = model.loss(batch[:, :-1], batch[:, 1:], cfg["num_negatives"])
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
        steps += 1
        if cfg["max_steps_per_epoch"] is not None and steps >= cfg["max_steps_per_epoch"]:
            break
    return total / max(steps, 1)


def score_split(model, inputs, targets, lookup, cfg, device, fp16, k_values, seen_sets):
    idx, scores = rank(model, inputs, cfg["num_retrieved"], cfg["eval_batch_size"], device, fp16)
    item_ids = lookup[idx]
    metrics = evaluate(item_ids[:, : constants.NUM_RANKED], targets, k_values, len(lookup) - 1, seen_sets)
    return metrics, item_ids, scores


def load_split(prepared, split, max_len):
    history_name = "history_val" if split == "val" else "history_test"
    uids, histories = read_histories(prepared / history_name)
    target_uids, targets = read_targets(prepared / (split + "_targets"))
    inputs = last_inputs(align(uids, histories, target_uids), max_len)
    return target_uids, targets, inputs


def main():
    config, _ = parse_args("Train SASRec")
    cfg = config["sasrec"]
    k_values = config["eval"]["k_values"]
    prepared = Path(config["paths"]["prepared"])
    out_dir = Path(config["paths"]["outputs"]) / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg["seed"])
    device = pick_device(cfg["device"])
    fp16 = bool(cfg["fp16"]) and device.type == "cuda"
    max_len = cfg["max_seq_len"]

    lookup = read_index_map(prepared / "item_map")
    train_items = lookup[1:]
    _, train_histories = read_histories(prepared / "history_val")
    windows = torch.from_numpy(make_windows(train_histories, max_len, cfg["window_stride"]))
    if device.type == "cuda":
        windows = windows.pin_memory()
    val_uids, val_targets, val_inputs = load_split(prepared, "val", max_len)
    test_uids, test_targets, test_inputs = load_split(prepared, "test", max_len)
    print(
        "items %d, windows %d, val users %d, test users %d, device %s, fp16 %s"
        % (len(train_items), len(windows), len(val_uids), len(test_uids), device, fp16),
        flush=True,
    )

    model = SASRec(
        len(train_items), max_len, cfg["embedding_dim"], cfg["num_layers"], cfg["num_heads"], cfg["dropout"]
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scaler = torch.amp.GradScaler(device.type, enabled=fp16)

    log = []
    best_value = -1.0
    best_state = None
    best_epoch = 0
    started = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        epoch_start = time.time()
        loss = train_epoch(model, optimizer, scaler, windows, cfg, device, fp16)
        val_metrics, _, _ = score_split(model, val_inputs, val_targets, lookup, cfg, device, fp16, [10], None)
        value = val_metrics["all"]["ndcg@10"]
        entry = {
            "epoch": epoch,
            "loss": loss,
            "val_ndcg@10": value,
            "val_recall@10": val_metrics["all"]["recall@10"],
            "seconds": time.time() - epoch_start,
        }
        log.append(entry)
        print(entry, flush=True)
        if value > best_value:
            best_value = value
            best_epoch = epoch
            best_state = {}
            for key, tensor in model.state_dict().items():
                best_state[key] = tensor.detach().cpu().clone()
        elif epoch - best_epoch >= cfg["patience"]:
            break
    train_seconds = time.time() - started

    model.load_state_dict(best_state)
    val_metrics, val_items, val_scores = score_split(
        model, val_inputs, val_targets, lookup, cfg, device, fp16, k_values, {"cold_train": train_items}
    )
    test_metrics, test_items, test_scores = score_split(
        model, test_inputs, test_targets, lookup, cfg, device, fp16, k_values, seen_item_sets(prepared)
    )
    write_ranked(out_dir / "candidates_val.parquet", val_uids, val_items, val_scores)
    write_ranked(out_dir / "candidates_test.parquet", test_uids, test_items, test_scores)
    torch.save({"state": best_state, "config": cfg, "num_items": len(train_items)}, out_dir / "model.pt")

    report = {
        "config": cfg,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "visible_gpus": torch.cuda.device_count(),
        "train_seconds": train_seconds,
        "best_epoch": best_epoch,
        "epochs": log,
        "validation": val_metrics,
        "test": test_metrics,
    }
    write_json(to_builtin(report), out_dir / "report.json")
    print("test", test_metrics["all"], flush=True)


if __name__ == "__main__":
    main()
