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

### Popularity baselines

Test day, 4,599 users. DecayPop weights each event by 0.9 per 2 hours of age, the period
being selected on the validation day by `ndcg@100` among 0.5 to 24 hours. Reproduce with `python -m nexttrack.eval.baselines`.

| Model | recall_paper@10 | recall_paper@100 | ndcg_paper@10 | ndcg_paper@100 | recall@10 | ndcg@10 | ndcg_official@10 | coverage@100 |
|---|---|---|---|---|---|---|---|---|
| MostPop (paper) | 0.0064 | 0.0321 | 0.0186 | 0.0249 | | | | 0.0002 |
| MostPop (here) | 0.0064 | 0.0321 | 0.0186 | 0.0249 | 0.0159 | 0.0172 | 0.0931 | 0.0002 |
| DecayPop (paper) | 0.0122 | 0.0479 | 0.0260 | 0.0323 | | | | 0.0002 |
| DecayPop (here) | 0.0122 | 0.0479 | 0.0260 | 0.0322 | 0.0250 | 0.0250 | 0.1437 | 0.0002 |

Other reference results from the paper (Listen+, 50M), in paper formulas:

| Model | ndcg_paper@10 | recall_paper@10 | recall_paper@100 |
|---|---|---|---|
| ItemKNN | 0.0781 | 0.0373 | 0.1297 |
| SASRec | 0.0748 | 0.0325 | 0.1026 |

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

## License

Apache-2.0, see `LICENSE`.
