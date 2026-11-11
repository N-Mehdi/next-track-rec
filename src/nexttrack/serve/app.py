"""FastAPI service: a listening history in, the next tracks out.

Same pipeline as the offline evaluation: SASRec encoder (ONNX) and exact FAISS search,
own history, audio kNN over never-seen tracks and DecayPop give 200 candidates, the
features come from rerank.features.user_rows, the LightGBM reranker runs in ONNX.
History events are expected to be Listen+ events (played_ratio_pct >= 50).
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import faiss
import numpy as np
import onnxruntime
from fastapi import FastAPI
from pydantic import BaseModel, Field

from nexttrack.data.sequences import last_inputs
from nexttrack.rerank.features import lookup_values, user_rows
from nexttrack.retrieval.audio_knn import build_index, load_audio, search, user_queries


def session(path):
    return onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])


class Recommender:
    def __init__(self, artifacts, audio_dir):
        artifacts = Path(artifacts)
        with open(artifacts / "settings.json", encoding="utf-8") as handle:
            self.settings = json.load(handle)
        self.tables = dict(np.load(artifacts / "tables.npz"))
        self.encoder = session(artifacts / "sasrec_encoder.onnx")
        self.reranker = session(artifacts / "reranker.onnx")
        self.item_vectors = np.load(artifacts / "item_vectors.npy")
        self.sasrec_index = faiss.IndexFlatIP(self.item_vectors.shape[1])
        self.sasrec_index.add(np.ascontiguousarray(self.item_vectors[1:]))
        # The cached copy is read without the raw parquet file.
        self.audio_ids, self.audio_vectors = load_audio(None, audio_dir)
        known = self.tables["map_ids"]
        self.audio_index, self.audio_pool = build_index(self.audio_ids, self.audio_vectors, known)

    def recommend(self, items, times, liked, disliked, k, now=None):
        """Top k item ids and reranker scores for one history (item ids and timestamps)."""
        quotas = self.settings["candidates"]
        order = np.argsort(times, kind="stable")
        items = np.asarray(items, dtype=np.int64)[order]
        times = np.asarray(times, dtype=np.int64)[order]
        if now is None:
            now = int(times[-1]) if len(times) > 0 else 0
        idx = lookup_values(self.tables["map_ids"], self.tables["map_idxs"], items, 0)
        inputs = last_inputs([idx[idx > 0]], self.settings["max_seq_len"])
        user = self.encoder.run(None, {"inputs": inputs})[0].astype(np.float32)
        _, rows = self.sasrec_index.search(user, quotas["total"])
        sasrec = self.tables["idx_to_id"][rows[0] + 1]
        query = user_queries([items], self.audio_ids, self.audio_vectors, quotas["audio_history_items"])
        audio, _ = search(self.audio_index, self.audio_pool, query, quotas["audio"])
        day = dict(self.tables)
        day["cutoff"] = now
        day["uids"] = np.zeros(1, dtype=np.int64)
        day["histories"] = [items]
        day["history_times"] = [times]
        day["sasrec"] = [sasrec]
        day["audio_lists"] = [audio[0]]
        day["user_vectors"] = user
        day["item_vectors"] = self.item_vectors
        day["audio_ids"] = self.audio_ids
        day["audio_vectors"] = self.audio_vectors
        day["audio_query"] = query
        day["audio_last"] = user_queries([items], self.audio_ids, self.audio_vectors, 1)
        # Keys of user 0 are the item ids themselves.
        day["liked"] = np.unique(np.asarray(liked, dtype=np.int64))
        day["disliked"] = np.unique(np.asarray(disliked, dtype=np.int64))
        candidates, matrix = user_rows(day, 0, self.settings)
        scores = self.reranker.run(None, {"features": matrix})[0].ravel()
        best = np.argsort(-scores, kind="stable")[:k]
        return candidates[best], scores[best]


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


class RecommendResponse(BaseModel):
    tracks: list[Track]


STATE = {}


@asynccontextmanager
async def lifespan(app):
    artifacts = os.environ.get("NEXTTRACK_ARTIFACTS", "artifacts")
    STATE["model"] = Recommender(artifacts, os.environ.get("NEXTTRACK_AUDIO", "data/audio"))
    yield
    STATE.clear()


app = FastAPI(title="next-track-rec", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    items = []
    times = []
    for event in request.history:
        items.append(event.item_id)
        times.append(event.timestamp)
    model = STATE["model"]
    top, scores = model.recommend(items, times, request.liked, request.disliked, request.k, request.now)
    tracks = []
    for i in range(len(top)):
        tracks.append(Track(item_id=int(top[i]), score=float(scores[i])))
    return RecommendResponse(tracks=tracks)
