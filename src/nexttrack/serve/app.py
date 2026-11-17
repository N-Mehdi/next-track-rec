"""FastAPI service: a listening history in, the next tracks out, with a readable description.

At start the service reads two folders, local or downloaded from Hugging Face at the given
revision:

  NEXTTRACK_RERANKER, NEXTTRACK_RERANKER_REVISION     reranker.onnx, settings.json, version.json
  NEXTTRACK_RETRIEVAL, NEXTTRACK_RETRIEVAL_REVISION   SASRec encoder (ONNX), item vectors, tables

Same pipeline as the offline evaluation: SASRec with an exact FAISS search, own history,
DecayPop (and the audio kNN when the model uses it, from NEXTTRACK_AUDIO) give the
candidates, rerank.features.user_rows the features, the LightGBM reranker runs in ONNX.
History events are expected to be Listen+ events (played_ratio_pct >= 50). Yambda has no
track, artist or album names: tracks are described by their identifiers and statistics.
/metrics exposes Prometheus counters and a latency histogram of /recommend.
"""

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import faiss
import numpy as np
import onnxruntime
from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, Info, make_asgi_app
from pydantic import BaseModel, Field

from nexttrack.data.sequences import last_inputs
from nexttrack.rerank.features import candidate_sources, lookup_values, user_rows, uses_audio
from nexttrack.retrieval.audio_knn import build_index, load_audio, search, user_queries

REQUESTS = Counter("nexttrack_requests_total", "Requests to /recommend by HTTP status", ["status"])
LATENCY = Histogram(
    "nexttrack_request_seconds",
    "Duration of /recommend requests",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0],
)
MODEL = Info("nexttrack_model", "Reranker version and retrieval bundle in use")


def session(path):
    return onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def resolve(source, revision, attempts=4):
    """A local folder, or a Hugging Face model repository downloaded at the revision.

    Network errors of the Hub are retried; files already in the cache are not fetched again."""
    if Path(source).is_dir():
        return Path(source)
    from huggingface_hub import snapshot_download

    for attempt in range(1, attempts + 1):
        try:
            return Path(snapshot_download(repo_id=source, revision=revision))
        except Exception as error:
            if attempt == attempts:
                raise
            print(
                "download of %s failed (%s), attempt %d of %d" % (source, error, attempt, attempts),
                flush=True,
            )
            time.sleep(15 * attempt)


def read_json(path, default=None):
    if not Path(path).exists():
        return default
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


class Recommender:
    def __init__(self, reranker, retrieval, audio_dir=None):
        reranker = Path(reranker)
        retrieval = Path(retrieval)
        self.settings = read_json(reranker / "settings.json")
        self.version = read_json(reranker / "version.json", {"version": None})["version"]
        self.retrieval = read_json(retrieval / "retrieval.json")
        expected = self.settings["retrieval"]
        if expected != self.retrieval["name"]:
            found = self.retrieval["name"]
            raise RuntimeError("reranker built on retrieval %s, bundle is %s" % (expected, found))
        self.tables = dict(np.load(retrieval / "tables.npz"))
        self.encoder = session(retrieval / "sasrec_encoder.onnx")
        self.reranker = session(reranker / "reranker.onnx")
        self.item_vectors = np.load(retrieval / "item_vectors.npy")
        self.sasrec_index = faiss.IndexFlatIP(self.item_vectors.shape[1])
        self.sasrec_index.add(np.ascontiguousarray(self.item_vectors[1:]))
        self.audio = uses_audio(self.settings)
        if self.audio:
            # The cached copy is read without the raw parquet file.
            self.audio_ids, self.audio_vectors = load_audio(None, audio_dir)
            known = self.tables["map_ids"]
            self.audio_index, self.audio_pool = build_index(self.audio_ids, self.audio_vectors, known)

    def candidates(self, items, times, liked, disliked, now):
        """Candidates, feature matrix and candidate lists for one sorted history."""
        quotas = self.settings["candidates"]
        idx = lookup_values(self.tables["map_ids"], self.tables["map_idxs"], items, 0)
        inputs = last_inputs([idx[idx > 0]], self.retrieval["max_seq_len"])
        user = self.encoder.run(None, {"inputs": inputs})[0].astype(np.float32)
        _, rows = self.sasrec_index.search(user, quotas["total"])
        day = dict(self.tables)
        day["cutoff"] = now
        day["uids"] = np.zeros(1, dtype=np.int64)
        day["histories"] = [items]
        day["history_times"] = [times]
        day["sasrec"] = [self.tables["idx_to_id"][rows[0] + 1]]
        day["user_vectors"] = user
        day["item_vectors"] = self.item_vectors
        # Keys of user 0 are the item ids themselves.
        day["liked"] = np.unique(np.asarray(liked, dtype=np.int64))
        day["disliked"] = np.unique(np.asarray(disliked, dtype=np.int64))
        if self.audio:
            query = user_queries([items], self.audio_ids, self.audio_vectors, quotas["audio_history_items"])
            audio, _ = search(self.audio_index, self.audio_pool, query, quotas["audio"])
            day["audio_lists"] = [audio[0]]
            day["audio_ids"] = self.audio_ids
            day["audio_vectors"] = self.audio_vectors
            day["audio_query"] = query
            day["audio_last"] = user_queries([items], self.audio_ids, self.audio_vectors, 1)
        return user_rows(day, 0, self.settings)

    def describe(self, top, scores, lists, items):
        """Identifiers, statistics and sources of the recommended tracks."""
        tables = self.tables
        artists = lookup_values(tables["artist_items"], tables["artists"], top, -1)
        albums = lookup_values(tables["album_items"], tables["albums"], top, -1)
        listens = lookup_values(tables["count_items"], tables["counts"], top, 0)
        history_artists = lookup_values(tables["artist_items"], tables["artists"], items, -1)
        sources = candidate_sources(top, lists)
        tracks = []
        for i in range(len(top)):
            artist = int(artists[i]) if artists[i] >= 0 else None
            album = int(albums[i]) if albums[i] >= 0 else None
            label = "track %d, artist %s, album %s" % (top[i], artist or "unknown", album or "unknown")
            user_listens = int((items == top[i]).sum())
            tracks.append(
                {
                    "item_id": int(top[i]),
                    "score": float(scores[i]),
                    "label": label,
                    "artist_id": artist,
                    "album_id": album,
                    "listens": int(listens[i]),
                    "user_listens": user_listens,
                    "already_listened": user_listens > 0,
                    "artist_already_listened": artist is not None and bool((history_artists == artist).any()),
                    "sources": sources[i],
                }
            )
        return tracks

    def recommend(self, items, times, liked, disliked, k, now=None):
        """The k best tracks for a history given as item ids and timestamps, any order."""
        order = np.argsort(times, kind="stable")
        items = np.asarray(items, dtype=np.int64)[order]
        times = np.asarray(times, dtype=np.int64)[order]
        if now is None:
            now = int(times[-1]) if len(times) > 0 else self.retrieval["cutoff"]
        candidates, matrix, lists = self.candidates(items, times, liked, disliked, now)
        scores = self.reranker.run(None, {"features": matrix})[0].ravel()
        best = np.argsort(-scores, kind="stable")[:k]
        return self.describe(candidates[best], scores[best], lists, items)


class Event(BaseModel):
    item_id: int
    timestamp: int


class RecommendRequest(BaseModel):
    history: list[Event]
    liked: list[int] = []
    disliked: list[int] = []
    k: int = Field(10, ge=1, le=100)
    now: int | None = None


class Track(BaseModel):
    item_id: int
    score: float
    label: str
    artist_id: int | None
    album_id: int | None
    # Listen+ events of the track by all users before the cutoff of the item tables.
    listens: int
    user_listens: int
    already_listened: bool
    artist_already_listened: bool
    # Candidate lists that proposed the track: sasrec, history, decay_pop (audio).
    sources: list[str]


class RecommendResponse(BaseModel):
    model_version: int | None
    tracks: list[Track]


STATE = {}


@asynccontextmanager
async def lifespan(app):
    environ = os.environ
    reranker = resolve(environ["NEXTTRACK_RERANKER"], environ.get("NEXTTRACK_RERANKER_REVISION"))
    retrieval = resolve(environ["NEXTTRACK_RETRIEVAL"], environ.get("NEXTTRACK_RETRIEVAL_REVISION"))
    model = Recommender(reranker, retrieval, environ.get("NEXTTRACK_AUDIO"))
    MODEL.info({"version": str(model.version), "retrieval": model.retrieval["name"]})
    STATE["model"] = model
    yield
    STATE.clear()


app = FastAPI(title="next-track-rec", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def observe(request: Request, call_next):
    if request.url.path != "/recommend":
        return await call_next(request)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        REQUESTS.labels(status="500").inc()
        raise
    LATENCY.observe(time.perf_counter() - started)
    REQUESTS.labels(status=str(response.status_code)).inc()
    return response


@app.get("/health")
def health():
    model = STATE["model"]
    return {
        "status": "ok",
        "model_version": model.version,
        "retrieval": model.retrieval["name"],
        "commit": os.environ.get("NEXTTRACK_COMMIT"),
    }


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    items = []
    times = []
    for event in request.history:
        items.append(event.item_id)
        times.append(event.timestamp)
    model = STATE["model"]
    tracks = model.recommend(items, times, request.liked, request.disliked, request.k, request.now)
    return RecommendResponse(model_version=model.version, tracks=tracks)
