# next-track-rec

Next-track music recommendation on Yambda-50M: two-stage retrieval and reranking.

Given a listening history, the system returns the next tracks to play. Retrieval is a
SASRec encoder served through a FAISS index, with an audio-embedding nearest-neighbour
path for tracks that were never listened to during training. A LightGBM reranker scores
the merged candidates and is exported to ONNX for a CPU FastAPI service.

## Data

Dataset: [yandex/yambda](https://huggingface.co/datasets/yandex/yambda), Apache-2.0, size `50m`.

| File | Rows | Columns |
|---|---|---|
| `flat/50m/listens.parquet` | 46.5 M | `uid`, `item_id`, `timestamp`, `is_organic`, `played_ratio_pct`, `track_length_seconds` |
| `flat/50m/likes.parquet`, `dislikes`, `unlikes`, `undislikes` | 0.9 M, 0.1 M, 0.2 M, 0.02 M | `uid`, `item_id`, `timestamp`, `is_organic` |
| `embeddings.parquet` | 7.72 M tracks | `item_id`, `embed`, `normalized_embed` |
| `album_item_mapping.parquet`, `artist_item_mapping.parquet` | | item to album, item to artist |

10,000 users, 934,057 tracks. `timestamp` is in seconds since the first event, binned to 5
seconds. Files are sorted by (`uid`, `timestamp`).

The main event stream is Listen+ : listens with `played_ratio_pct >= 50`, the definition used
by the official benchmark. Likes and dislikes are only used as reranker features.

## Evaluation protocol

The split reproduces `benchmarks/yambda/constants.py` and
`benchmarks/yambda/processing/timesplit.py` of the dataset repository (global temporal split).

| Segment | Interval (seconds) |
|---|---|
| train | `[0, 25 823 600)` |
| gap | 30 minutes |
| validation | `[25 825 400, 25 911 800)` |
| gap | 30 minutes |
| test | `[25 913 600, +inf)` |

- The Listen+ filter is applied to the whole stream before splitting, so targets are
  Listen+ events as well.
- Test users must appear in the training set. Test items are kept even when they never
  appear in training; these items form the cold-start subset reported separately.
- Targets are all Listen+ events of the user during the test day, not deduplicated, as in
  the official code.
- Ranking is over the full catalogue of training items, top 100.
  `Recall@k = hits@k / min(number of targets, k)`. Coverage@k is the share of catalogue
  items that appear in at least one top-k list.
- User states are frozen 30 minutes before the test day: the history fed to the model stops
  at `25 911 800`.

Differences with the official baselines:

- The official baselines train the final model on `[0, 25 911 800)` without validation. Here
  the retriever is trained once on `[0, 25 823 600)` and the validation day is used for early
  stopping and for training the reranker. The retriever therefore sees one day less of data
  than the official baselines, which makes the comparison slightly conservative.
### Metric formulas

The current `benchmarks/yambda/evaluation/metrics.py` and the paper tables use different
formulas. Comments in that file describe two earlier bugs, and its NDCG has a third one.

| Name | Formula | Source |
|---|---|---|
| `recall@k` | mean of hits / min(targets, k) | current `metrics.py` |
| `recall_paper@k` | mean of hits / targets | paper tables (bug described in `metrics.py`) |
| `ndcg@k` | mean over users of DCG / ideal DCG | standard definition |
| `ndcg_paper@k` | sum of DCG / sum of ideal DCG | paper tables (bug described in `metrics.py`) |
| `ndcg_official@k` | DCG / DCG computed on the hit mask | current `metrics.py` |

The NDCG of the current `metrics.py` builds the ideal mask but then calls
`ideal_dcg = calc_dcg(target_mask)` on the hit mask, so each user scores 1 when the top-k
list holds at least one hit and 0 otherwise: it is a hit rate, not an NDCG. Users have 34
test targets on average, so `recall_paper@k` is much lower than `recall@k` for small k.

The popularity baselines below reproduce the paper tables to the fourth decimal with
`recall_paper` and `ndcg_paper`, which validates the split, the Listen+ filter and the
targets. `ndcg_official` does not match the paper. Comparisons with the paper therefore use
`recall_paper` and `ndcg_paper`; `recall@k` and `ndcg@k` are the main metrics of this project.

### Baselines

Test day, 4,599 users. Reproduce with `python -m nexttrack.eval.baselines`
(`reports/baselines.json`).

- MostPop: global listen counts.
- DecayPop: global counts, each event weighted by 0.9 per 2 hours of age.
- History: the tracks of the user's own history, each listen weighted by 0.9 per 24 hours
  of age before the end of the history. The paper has no repeat-aware baseline.

Decay periods are selected on the validation day by `ndcg@100` (0.5 to 24 hours for
DecayPop, 6 hours to 90 days and plain counts for History).

Comparison with the paper (paper formulas):

| Model | recall_paper@10 | recall_paper@100 | ndcg_paper@10 | ndcg_paper@100 | coverage@100 |
|---|---|---|---|---|---|
| MostPop (paper) | 0.0064 | 0.0321 | 0.0186 | 0.0249 | 0.0002 |
| MostPop (here) | 0.0064 | 0.0321 | 0.0186 | 0.0249 | 0.0002 |
| DecayPop (paper) | 0.0122 | 0.0479 | 0.0260 | 0.0323 | 0.0002 |
| DecayPop (here) | 0.0122 | 0.0479 | 0.0260 | 0.0322 | 0.0002 |
| ItemKNN (paper) | 0.0373 | 0.1297 | 0.0781 | 0.0934 | 0.0874 |
| SASRec (paper) | 0.0325 | 0.1026 | 0.0748 | 0.0764 | 0.0310 |
| History (here) | 0.1082 | 0.2640 | 0.2172 | 0.2104 | 0.1797 |
| SASRec A (here) | 0.0457 | 0.1672 | 0.0975 | 0.1214 | 0.1317 |

Main metrics of this project:

| Model | recall@10 | ndcg@10 | recall@100 | ndcg@100 | ndcg_official@10 |
|---|---|---|---|---|---|
| MostPop | 0.0159 | 0.0172 | 0.0327 | 0.0216 | 0.0931 |
| DecayPop | 0.0250 | 0.0250 | 0.0486 | 0.0313 | 0.1437 |
| History | 0.2054 | 0.2108 | 0.2676 | 0.2155 | 0.4708 |
| SASRec A | 0.0927 | 0.0944 | 0.1699 | 0.1171 | 0.3838 |

### Never-seen tracks

Two definitions are reported, each on the users that have at least one such target:

| Definition | Seen tracks | Test targets | Users |
|---|---|---|---|
| `cold_train`: no Listen+ event before `TRAIN_END` | 627,648 | 2,332 (1.48 %) | 688 |
| `cold_strict`: no Listen+ event before `VAL_END` (training, first gap and validation day) | 629,298 | 1,956 (1.24 %) | 602 |

`cold_strict` is computed from `history_test`; the same set is obtained from the raw
listens of all users. Values in brackets are 95 % bootstrap intervals over users (1,000
resamples).

| Model | Definition | recall@10 | recall@100 | ndcg@10 |
|---|---|---|---|---|
| MostPop, DecayPop, SASRec | both | 0 | 0 | 0 |
| History | `cold_train` | 0.0096 [0.0033, 0.0167] | 0.0325 [0.0210, 0.0444] | 0.0064 [0.0026, 0.0114] |
| History | `cold_strict` | 0 | 0 | 0 |
| Audio kNN | `cold_train` | 0.0051 [0.0015, 0.0096] | 0.0139 [0.0067, 0.0217] | 0.0058 [0.0019, 0.0107] |
| Audio kNN | `cold_strict` | 0.0018 [0.0000, 0.0043] | 0.0080 [0.0028, 0.0145] | 0.0019 [0.0000, 0.0046] |

History reaches `cold_train` tracks only because the test history ends at `VAL_END`: a track
first listened during the validation day is absent from training but present in the
history. Under `cold_strict` no collaborative model can score, which leaves the audio
embeddings as the only signal.

Audio kNN (`python -m nexttrack.retrieval.audio_knn`, `reports/audio_knn.json`), no training:

- Query: mean of the normalized audio embeddings of the user's last 5 listened tracks with
  audio, renormalized. The number of tracks (1, 5, 10, 20, 50 or 200) is selected on the
  validation day by `ndcg@100` on its never-seen targets; 1 to 20 give close values.
- Search pool: the 7,128,158 tracks of `embeddings.parquet` absent from training (exact
  inner product, FAISS). The pool does not depend on the test day; it holds 93.7 % of the
  `cold_strict` targets. A pool limited to tracks with any event before the test day
  would hold only 28.9 % of them.
- On 7.1 M tracks a random top 100 hits a given target with probability 1.4e-5;
  `recall@100` on `cold_strict` is about 570 times that.
- Reading the 13.8 GB parquet file once writes a 4 GB float32 copy to `data/audio`. One
  search over the pool for the 4,599 test users takes about 2 minutes on 12 CPU cores.

### Prepared data in numbers

| Quantity | Value |
|---|---|
| Listen+ training events (before `TRAIN_END`) | 29,135,186 |
| Training users, training items | 9,207, 627,648 |
| Validation day: users, events | 4,627, 143,557 |
| Test day: users, events | 4,599, 157,695 |
| Test targets already in the user history | 68.3 % (24.6 % within the last 200 events) |
| Test targets never seen in training | 1.48 % (2,332 events, 1,782 tracks, 688 users) |
| Never-seen test tracks with an audio embedding | 93.4 % |
| History length before the test day | median 1,787, p90 8,077 |

## Repository layout

```
configs/default.yaml        paths, split and model hyperparameters
src/nexttrack/
  constants.py              split boundaries, Listen+ threshold, k values
  config.py                 YAML loading with dotted overrides
  prep/                     download and Spark preparation (phase 1)
  data/                     torch datasets (phase 2)
  models/                   SASRec (phase 2)
  train/                    SASRec training and candidate export (phase 2)
  retrieval/                FAISS index, audio kNN, candidate union (phases 2 and 3)
  rerank/                   features, LightGBM, ONNX export (phase 4)
  eval/                     metrics and evaluation runs (phase 1)
  serve/                    FastAPI application (phase 5)
tests/
notebooks/kaggle/           thin runners that clone this repository
reports/                    metrics and model cards
```

## Plan

1. Skeleton, constants, protocol documentation.
2. Download, Spark preparation, metrics, popularity baselines (local, CPU).
3. SASRec training on Kaggle T4, at most two configurations.
4. Cold start: audio nearest neighbours without training, from the embeddings of the last
   listened tracks. A trained two-tower model is only added if it clearly beats this baseline
   on never-seen tracks.
5. Candidate union (200 candidates), LightGBM reranker, ONNX export.
6. FastAPI service, Docker image, final report.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[prep,dev]"       # local data preparation
pip install -e ".[train]"          # training and candidate generation
pip install -e ".[serve]"          # inference service
pytest
```

Spark runs locally (the 50M flat files total about 400 MB compressed). Java 17 or 21 is
required for Spark 4.

### Data preparation (local)

```bash
python -m nexttrack.prep.download                 # 14.4 GB into data/raw, embeddings included
python -m nexttrack.prep.split                    # Spark, writes data/prepared and stats.json
python -m nexttrack.eval.baselines                # MostPop and DecayPop, writes reports/baselines.json
python -m nexttrack.retrieval.audio_knn           # audio kNN on never-seen tracks, writes reports/audio_knn.json
```

Every command accepts `--config` and `--set section.key=value`. A smoke run on 300 users:

```bash
python -m nexttrack.prep.split --set data.max_users=300 paths.prepared=data/smoke
```

Prepared folders (parquet, one per table):

| Folder | Content |
|---|---|
| `item_map` | `item_id`, `item_idx` (1..N, 0 is padding), `train_count`; training items only |
| `train_events`, `val_events`, `test_events` | Listen+ events with `item_idx` (0 when unseen in training) |
| `history_val`, `history_test` | per user arrays, events before `TRAIN_END` and before `VAL_END` |
| `val_targets`, `test_targets` | per user arrays of the validation and test day |
| `likes`, `dislikes`, `unlikes`, `undislikes` | raw side events of the kept users |
| `catalogue` | `item_id`, `item_idx`, `embedding` float32[128] for every listened or liked item with audio |

Items whose `item_idx` is 0 in the targets have no Listen+ interaction before `TRAIN_END`;
they form the cold-start subset of the evaluation.

The prepared folders are published as the private Kaggle dataset
`mehdineji/yambda-50m-prepared`.

### SASRec training (Kaggle T4)

```bash
python -m nexttrack.train.train_sasrec --set sasrec.run_name=a_d128
```

- Training windows of 201 events are cut from the end of every training history every 100
  events (about 300,000 windows), left padded, so the whole history is used and not only
  the last 200 events.
- Loss: sampled softmax over 2,048 uniform negatives shared by the batch, item embeddings
  tied between input and output, fp16 autocast.
- Early stopping on `ndcg@10` of the validation day (patience 3, at most 20 epochs).
- Inputs: the last 200 events before `TRAIN_END` for the validation day, before `VAL_END`
  for the test day. Tracks unseen in training are dropped from the inputs.
- Outputs in `outputs/<run_name>/`: `model.pt`, `report.json` and the top 200 tracks per
  target user for both days (`candidates_val.parquet`, `candidates_test.parquet`).

Two configurations only, identical except for the embedding size: A (128) and B (256).
A first run trained both for 20 epochs. `notebooks/kaggle/02_sasrec` now clones this
repository and retrains A alone for up to 60 epochs (run `a_d128_e60`), since A was still
improving at epoch 20.
Results (one T4, fp16, reports in `reports/sasrec_a_d128.json` and `reports/sasrec_b_d256.json`):

| Config | Epochs run, best | Training time | val ndcg@10 | test recall@10 | test ndcg@10 | test recall@100 |
|---|---|---|---|---|---|---|
| A, dim 128 | 20, 20 | 57 min | 0.0980 | 0.0927 | 0.0944 | 0.1699 |
| B, dim 256 | 20, 19 | 88 min | 0.0944 | 0.0914 | 0.0930 | 0.1655 |

A is selected on the validation day. With the paper formulas it reaches
`recall_paper@10` 0.0457 and `ndcg_paper@10` 0.0975, against 0.0325 and 0.0748 for the paper
SASRec. It stays well below the History baseline: 68 % of test targets are repeats of the
user's own history, which a next-item model over the full catalogue ranks less sharply than
a recency-weighted count. SASRec cannot score tracks absent from training (both cold
definitions give 0). The candidate union and the reranker combine both sources.

A local smoke run on CPU:

```bash
python -m nexttrack.train.train_sasrec --set paths.prepared=data/smoke paths.outputs=data/smoke_outputs \
    sasrec.embedding_dim=32 sasrec.epochs=2 sasrec.max_steps_per_epoch=10 sasrec.num_negatives=256
```

## License

Apache-2.0, see `LICENSE`.
