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
- The NDCG implementation in `benchmarks/yambda/evaluation/metrics.py` computes the ideal DCG
  from the hit mask instead of the ideal mask (`ideal_dcg = calc_dcg(target_mask)`), so the
  official NDCG equals 1 whenever the top-k list contains at least one hit and 0 otherwise.
  Both values are reported here: `ndcg_official@k` for comparison with the paper and
  `ndcg@k` with the standard ideal DCG. Recall@k is unaffected and is the primary metric for
  comparison with the paper.

Reference results from the paper (Yambda-50M, Listen+):

| Model | NDCG@10 | Recall@10 | Recall@100 |
|---|---|---|---|
| MostPop | 0.0186 | 0.0064 | 0.0321 |
| DecayPop | 0.0260 | 0.0122 | 0.0479 |
| ItemKNN | 0.0781 | 0.0373 | 0.1297 |
| SASRec | 0.0748 | 0.0325 | 0.1026 |

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

## License

Apache-2.0, see `LICENSE`.
