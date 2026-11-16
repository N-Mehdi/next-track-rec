import json

import numpy as np
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("onnxruntime")

from nexttrack.config import load_config  # noqa: E402
from nexttrack.eval import regression  # noqa: E402
from nexttrack.rerank.features import AUDIO_FEATURES  # noqa: E402

HISTORY = [
    "history_score",
    "user_item_count",
    "user_item_count_recent",
    "hours_since_last",
    "events_since_last",
]


@pytest.fixture(scope="module")
def frozen():
    return dict(np.load(regression.FROZEN))


@pytest.fixture(scope="module")
def reference():
    with open(regression.REFERENCE, encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def cfg():
    return load_config("configs/default.yaml")["rerank"]


def test_frozen_set_layout(frozen):
    assert len(frozen["sizes"]) == regression.NUM_USERS
    assert frozen["features"].shape == (frozen["sizes"].sum(), len(frozen["names"]))
    # The frozen set is the production variant: no audio feature.
    for name in AUDIO_FEATURES:
        assert name not in list(frozen["names"])
    assert frozen["target_sizes"].sum() == len(frozen["targets"])


def test_current_code_matches_reference(frozen, reference, cfg):
    values = regression.measure(frozen, cfg)
    assert regression.failures(values, reference) == []
    assert values["ndcg@10"] == pytest.approx(reference["ndcg@10"], abs=1e-9)


def test_degraded_features_fail_the_check(frozen, reference, cfg):
    broken = dict(frozen)
    broken["features"] = frozen["features"].copy()
    for name in HISTORY:
        broken["features"][:, list(frozen["names"]).index(name)] = 0.0
    reasons = regression.failures(regression.measure(broken, cfg), reference)
    assert len(reasons) == 1
    assert "below reference" in reasons[0]


def test_onnx_gap_is_reported(reference):
    values = {"ndcg@10": reference["ndcg@10"], "onnx_ndcg@10": reference["ndcg@10"] - 0.01}
    assert "ONNX" in regression.failures(values, reference)[0]
