"""Tests for the inference service HTTP API (TestClient, no server, no network)."""
import numpy as np
from fastapi.testclient import TestClient

import inference.app as inf
from model.registry import ModelRegistry
from model.network import GazeNet
from model.schemas import WINDOW_LEN


def _client(tmp_registry):
    inf.reg = ModelRegistry(tmp_registry)   # point the service at the temp registry
    inf._cache.clear()
    return TestClient(inf.app)


def test_health(tmp_registry):
    r = _client(tmp_registry).get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_returns_valid_schema(tmp_registry, rand_window):
    r = _client(tmp_registry).post("/predict",
                                   json={"user_id": "u1", "window": rand_window.tolist()})
    assert r.status_code == 200
    body = r.json()
    assert set(body["gaze"]) == {"x", "y"}
    assert isinstance(body["blink"], bool)
    assert body["model"] == "base"          # no personal model yet


def test_predict_rejects_wrong_shape(tmp_registry):
    bad = [[0.0] * 10, [0.0] * 10]           # (2, 10) not (2, 320)
    r = _client(tmp_registry).post("/predict", json={"user_id": "u1", "window": bad})
    assert r.status_code == 400


def test_reload_switches_to_personal(tmp_registry, rand_window):
    client = _client(tmp_registry)
    # first call serves base and caches it
    assert client.post("/predict", json={"user_id": "u2", "window": rand_window.tolist()}).json()["model"] == "base"
    # a personal model appears; /reload evicts the stale cache
    inf.reg.save_personal("u2", GazeNet().state_dict())
    assert client.post("/reload/u2").status_code == 200
    assert client.post("/predict", json={"user_id": "u2", "window": rand_window.tolist()}).json()["model"] == "personal"
