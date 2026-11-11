# next-track-rec

Next-track music recommendation on Yambda-50M: two-stage retrieval and reranking.

Given a listening history, the system returns the next tracks to play. Four sources
propose 200 candidates: a SASRec encoder searched with FAISS, the user's own history, an
audio-embedding nearest-neighbour search over tracks never listened to during training,
and recent popularity. A LightGBM reranker orders them. The SASRec encoder and the reranker
run in ONNX inside a CPU FastAPI service.

## Results

Test day of Yambda-50M (Listen+, 4,599 users, 157,695 target events), histories frozen
30 minutes before the day. `recall@k` and `ndcg@k` are the main metrics; the `_paper`
columns use the formulas behind the paper tables and `ndcg_official` the current official
code (a hit rate, see [Metric formulas](#metric-formulas)). The last column is
`recall@100` on the 1,956 targets never listened to before the test day (`cold_strict`),
with its 95 % bootstrap interval.

| Model | recall@10 | ndcg@10 | recall@100 | ndcg@100 | ndcg_official@10 | recall_paper@10 | ndcg_paper@10 | coverage@100 | cold_strict recall@100 |
|---|---|---|---|---|---|---|---|---|---|
| MostPop | 0.0159 | 0.0172 | 0.0327 | 0.0216 | 0.0931 | 0.0064 | 0.0186 | 0.0002 | 0 |
| DecayPop (2 h) | 0.0250 | 0.0250 | 0.0486 | 0.0313 | 0.1437 | 0.0122 | 0.0260 | 0.0002 | 0 |
| History (24 h) | 0.2054 | 0.2108 | 0.2676 | 0.2155 | 0.4708 | 0.1082 | 0.2172 | 0.1797 | 0 |
| SASRec A | 0.1012 | 0.1049 | 0.1797 | 0.1262 | 0.4068 | 0.0499 | 0.1085 | 0.1496 | 0 |
| Audio kNN (never-seen tracks only) | 0.0003 | 0.0005 | 0.0004 | 0.0004 | 0.0024 | 0.0002 | 0.0005 | (1) | 0.0080 [0.0028, 0.0145] |
| **Reranker** | **0.2363** | **0.2453** | **0.2872** | **0.2386** | **0.5888** | **0.1199** | **0.2540** | 0.1608 | 0.0013 [0.0000, 0.0040] |
| Reranker, audio every 10th position | 0.2212 | 0.2362 | 0.2810 | 0.2337 | 0.5740 | 0.1137 | 0.2441 | 0.2121 | 0.0024 [0.0000, 0.0059] |
| SASRec (paper) | | | | | | 0.0325 | 0.0748 | 0.0310 | |
| ItemKNN (paper) | | | | | | 0.0373 | 0.0781 | 0.0874 | |

(1) The audio kNN only searches the 7.1 M tracks absent from training; its top 100 lists
cover 3.8 % of that pool.

- The reranker beats the History baseline, the strongest single source: `ndcg@10` +0.0345,
  paired bootstrap interval [0.0302, 0.0387].
- Never-seen tracks: the audio kNN alone is the best model on `cold_strict`; the reranker
  keeps few of its candidates. Reserving one position in ten for them recovers part of it
  at a cost of 0.0092 in `ndcg@10` (details in [Reranker](#reranker-local-cpu)).
- The CPU service returns the same top 10 as the offline evaluation for 296 of 300 sampled
  test users, in about 0.5 s per request.

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
| SASRec A (here) | 0.0499 | 0.1769 | 0.1085 | 0.1292 | 0.1496 |
| Reranker (here) | 0.1199 | 0.2831 | 0.2540 | 0.2364 | 0.1608 |

Main metrics of this project:

| Model | recall@10 | ndcg@10 | recall@100 | ndcg@100 | ndcg_official@10 |
|---|---|---|---|---|---|
| MostPop | 0.0159 | 0.0172 | 0.0327 | 0.0216 | 0.0931 |
| DecayPop | 0.0250 | 0.0250 | 0.0486 | 0.0313 | 0.1437 |
| History | 0.2054 | 0.2108 | 0.2676 | 0.2155 | 0.4708 |
| SASRec A | 0.1012 | 0.1049 | 0.1797 | 0.1262 | 0.4068 |
| Reranker | 0.2363 | 0.2453 | 0.2872 | 0.2386 | 0.5888 |
| Reranker, audio every 10th position | 0.2212 | 0.2362 | 0.2810 | 0.2337 | 0.5740 |

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
| Reranker | `cold_train` | 0.0091 [0.0036, 0.0156] | 0.0225 [0.0131, 0.0330] | 0.0063 [0.0028, 0.0110] |
| Reranker | `cold_strict` | 0.0007 [0.0000, 0.0020] | 0.0013 [0.0000, 0.0040] | 0.0005 [0.0000, 0.0016] |
| Reranker, audio every 10th position | `cold_train` | 0.0118 [0.0052, 0.0190] | 0.0249 [0.0154, 0.0362] | 0.0072 [0.0033, 0.0120] |
| Reranker, audio every 10th position | `cold_strict` | 0.0015 [0.0000, 0.0038] | 0.0024 [0.0000, 0.0059] | 0.0008 [0.0000, 0.0022] |

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

Possible improvement, not implemented: a two-tower model that maps audio embeddings into
the SASRec item space, trained on the tracks seen in training, would give never-seen tracks
a collaborative vector instead of a pure audio similarity.

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
  data/                     SASRec training windows and inputs (phase 2)
  models/                   SASRec (phase 2)
  train/                    SASRec training and candidate export (phase 2)
  retrieval/                audio kNN over never-seen tracks (phase 3)
  rerank/                   candidate union, features, LightGBM, ONNX export (phase 4)
  eval/                     metrics, bootstrap, baselines (phase 1)
  serve/                    artifact export and FastAPI application (phase 5)
tests/
notebooks/kaggle/           thin runner that clones this repository (SASRec on a T4)
reports/                    JSON reports behind every table of this file
Dockerfile                  CPU image of the service
```

## Phases

All phases are done.

1. Skeleton, constants, protocol documentation.
2. Download, Spark preparation, metrics, popularity and history baselines (local, CPU).
3. SASRec training on Kaggle T4, two configurations, then the better one retrained longer.
4. Cold start: audio nearest neighbours without training, from the embeddings of the last
   listened tracks, over every track with audio absent from training. No two-tower model
   (listed as a possible improvement).
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
python -m nexttrack.eval.baselines                # MostPop, DecayPop, History, writes reports/baselines.json
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
- Early stopping on `ndcg@10` of the validation day (patience 3).
- Inputs: the last 200 events before `TRAIN_END` for the validation day, before `VAL_END`
  for the test day. Tracks unseen in training are dropped from the inputs.
- Outputs in `outputs/<run_name>/`: `model.pt`, `report.json` and the top 200 tracks per
  target user for both days (`candidates_val.parquet`, `candidates_test.parquet`).

Two configurations only, identical except for the embedding size: A (128) and B (256).
A first run trained both for at most 20 epochs; A was still improving at epoch 20, so
`notebooks/kaggle/02_sasrec` retrains A alone for at most 60 epochs (run `a_d128_e60`).
Results on one T4, fp16 (reports in `reports/sasrec_*.json`):

| Run | Epochs run, best | Training time | val ndcg@10 | test recall@10 | test ndcg@10 | test recall@100 |
|---|---|---|---|---|---|---|
| A, dim 128, 20 epochs max | 20, 20 | 57 min | 0.0980 | 0.0927 | 0.0944 | 0.1699 |
| B, dim 256, 20 epochs max | 20, 19 | 88 min | 0.0944 | 0.0914 | 0.0930 | 0.1655 |
| A, dim 128, 60 epochs max | 35, 32 | 100 min | 0.1089 | 0.1012 | 0.1049 | 0.1797 |

A with 60 epochs is selected on the validation day and feeds the reranker. With the paper
formulas it reaches `recall_paper@10` 0.0499 and `ndcg_paper@10` 0.1085, against 0.0325 and
0.0748 for the paper SASRec. It stays well below the History baseline: 68 % of test
targets are repeats of the user's own history, which a next-item model over the full
catalogue ranks less sharply than a recency-weighted count. SASRec cannot score tracks
absent from training (both cold definitions give 0).

A local smoke run on CPU:

```bash
python -m nexttrack.train.train_sasrec --set paths.prepared=data/smoke paths.outputs=data/smoke_outputs \
    sasrec.embedding_dim=32 sasrec.epochs=2 sasrec.max_steps_per_epoch=10 sasrec.num_negatives=256
```

### Reranker (local, CPU)

```bash
python -m nexttrack.rerank.train_reranker         # writes reports/reranker.json and outputs/rerank
```

Candidates, 200 per user, deduplicated in this order: SASRec top 100, History top 50, audio
kNN top 30, DecayPop top 20, then the next SASRec tracks until 200. The union holds
30.6 % of the test targets (`recall@200`), 3.2 % of the `cold_train` ones and 0.33 % of
the `cold_strict` ones.

Features (17), computed from events before the cutoff of the day only:

- SASRec score (user vector times item embedding, for every candidate seen in training);
- own history: recency-weighted score, listen count, count in the last 200 events, hours
  and events since the last listen, history length;
- audio: similarity with the mean of the last 5 tracks and with the last track;
- popularity: DecayPop score, Listen+ count, like count;
- user like or dislike of the track (latest of like and unlike, dislike and undislike);
- listens of the track's artist by the user, count and share of the history;
- never seen in training.

Training: LightGBM lambdarank on the validation day (histories end at `TRAIN_END`), binary
label "listened during the day", users without a positive candidate left out (3,978
users kept), deterministic mode (same trees on every run). The number of rounds (129) maximizes the mean `ndcg@10` of a 5-fold
cross-validation over these users, then one model is trained on all of them. An earlier
version used early stopping on a single 20 % holdout; its holdout curve was noisy (best
round between 18 and 110 depending on the split) and it stopped at round 13. The change
was made after seeing its test result (`ndcg@10` 0.2354), which is reported here for
completeness.

The test day is scored with histories ending at `VAL_END`. Paired bootstrap over test
users (1,000 resamples, same draws for both models):

| Difference | recall@10 | ndcg@10 | recall@100 | ndcg@100 |
|---|---|---|---|---|
| Reranker minus History | +0.0309 [0.0263, 0.0354] | +0.0345 [0.0302, 0.0387] | +0.0196 [0.0153, 0.0240] | +0.0231 [0.0203, 0.0258] |
| Audio every 10th position minus reranker, all users | -0.0151 [-0.0163, -0.0138] | -0.0092 [-0.0099, -0.0085] | -0.0062 [-0.0072, -0.0053] | -0.0049 [-0.0051, -0.0046] |
| Same, `cold_strict` targets | +0.0008 [0.0000, 0.0025] | +0.0003 [0.0000, 0.0009] | +0.0011 [0.0000, 0.0033] | +0.0004 [0.0000, 0.0012] |

Never-seen tracks: the reranker recovers almost none of the `cold_strict` targets
(`recall@100` 0.0013, against 0.0080 for the audio kNN alone). The union only holds 30 audio
tracks per user, and the validation day has too few never-seen positives for the model to
learn to rank them (gain of the "never seen" feature close to 0). The post-processing
variant gives positions 10, 20, ..., 100 to the next audio kNN tracks: it costs 0.0092 of
`ndcg@10` on all users and adds 0.0011 of `recall@100` on `cold_strict`. Both results are
published; the reranker without the variant is the main model.

Most used features (gain): count in the last 200 events, history score, events since the
last listen, artist listens, SASRec score. Features take 71 s for both days, the 6
trainings (5 folds and the final model) 2 minutes on 12 cores.

ONNX export: `outputs/rerank/reranker.onnx`, converted with onnxmltools. Its tree
thresholds are float32 while LightGBM compares in float64, so a value lying between a
threshold and its float32 rounding takes the other branch. Features are rounded to three
decimals, which keeps two distinct values from being adjacent float32 numbers, but
LightGBM can still place a threshold within float32 precision of a frequent value: 20 of
919,800 test rows differ by more than 1e-4, and the test `ndcg@10` computed from the ONNX
scores is equal to the LightGBM one to the fourth decimal (0.2453).

## Service (CPU, Docker)

```bash
python -m nexttrack.serve.export                  # writes artifacts/ (790 MB), needs the train extra
docker build -t next-track-rec .
docker run -p 8000:8000 -v "$PWD/artifacts:/artifacts:ro" -v "$PWD/data/audio:/audio:ro" next-track-rec
```

`artifacts/` holds the SASRec encoder (`sasrec_encoder.onnx`, last 200 events to a user
vector), the item embeddings for the exact FAISS search, the reranker
(`reranker.onnx`), item tables frozen at `VAL_END` (popularity, listen and like counts,
artist of each track) and `settings.json` (candidate quotas). The float32 audio embeddings
(`data/audio`, 4 GB, written by the audio kNN step) are mounted separately. The image
(764 MB) holds no torch and no model file.

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/recommend -H 'Content-Type: application/json' -d '{
  "history": [{"item_id": 6566295, "timestamp": 25900000}, {"item_id": 2562990, "timestamp": 25900200}],
  "liked": [2562990],
  "disliked": [],
  "k": 10
}'
```

- `history`: Listen+ events (`played_ratio_pct >= 50`) as item id and timestamp, in any
  order. Tracks unknown to SASRec are kept for the history, audio and artist features.
- `liked`, `disliked`: the user's current likes and dislikes (optional).
- `k`: 1 to 100. `now` (optional): the request time, by default the last event.
- Response: `{"tracks": [{"item_id": ..., "score": ...}, ...]}`, best first.

The service builds the 200 candidates and the 17 features with the same functions as the
offline evaluation (`rerank.features.user_rows`). On 300 random test users it returns the
same top 10 as the offline ranking for 296 of them; `ndcg@10` on these users is 0.2714
against 0.2713 offline. The remaining differences come from the SASRec candidates, computed
in fp16 on the GPU offline and in fp32 in the service. One request takes about 0.5 s
(p50 0.49 s, p95 0.59 s on 12 cores), most of it in the exact search over 7.1 M audio
vectors. The container starts in a few seconds and uses about 4.7 GB of memory.

## Models

| Model | Input | Output | Trained on |
|---|---|---|---|
| SASRec A | last 200 Listen+ tracks seen in training | user vector (128) | Listen+ events before `TRAIN_END` |
| Audio kNN | last 5 tracks with an audio embedding | 30 tracks absent from training | no training |
| Reranker | 17 features per candidate | score | validation day, 3,978 users |

Intended use: ranking the next tracks of a listener of the Yambda-50M population, for
research and demonstration. Limits:

- The models know the 627,648 tracks listened before `TRAIN_END` and the 10,000 Yambda-50M
  users' behaviour only; item tables are frozen at `VAL_END`.
- Never-seen tracks are reached through audio similarity only, and the reranker rarely
  keeps them (see the table above).
- 68 % of test targets are repeats: the reranker relies mostly on the user's own history
  (main features: listens of the track in the last 200 events, recency-weighted history
  score, events since the last listen). On average 4.7 % of its top 10 are tracks absent
  from the user's history, while 34 % of a user's test targets are.
- Metrics are offline, on one test day.

## License

Apache-2.0, see `LICENSE`.
