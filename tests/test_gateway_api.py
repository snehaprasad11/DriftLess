"""Tests for the gateway HTTP API. Redis and the inference service are faked, so
no external services are needed."""
from fastapi.testclient import TestClient

import gateway.app as gw
from model.schemas import WINDOW_LEN


class FakeRedis:
    def __init__(self):
        self.q = []
    def rpush(self, key, val):
        self.q.append(val); return len(self.q)
    def llen(self, key):
        return len(self.q)


class FakeResponse:
    status_code = 200
    def __init__(self, payload):
        self._p = payload
    def json(self):
        return self._p


def test_health(monkeypatch):
    monkeypatch.setattr(gw, "r", FakeRedis())
    r = TestClient(gw.app).get("/health")
    assert r.status_code == 200 and r.json()["redis"] is True


def test_calibrate_enqueues(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(gw, "r", fake)
    trials = [{"window": [[0.0] * WINDOW_LEN, [0.0] * WINDOW_LEN], "gaze": [1.0, 2.0]}
              for _ in range(3)]
    r = TestClient(gw.app).post("/calibrate", json={"user_id": "u1", "trials": trials})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "queued" and body["n_trials"] == 3
    assert len(fake.q) == 1                       # exactly one job enqueued


def test_predict_forwards_to_inference(monkeypatch):
    monkeypatch.setattr(gw, "r", FakeRedis())
    served = {"gaze": {"x": 1.0, "y": -2.0}, "blink": False, "model": "base"}
    monkeypatch.setattr(gw.httpx, "post", lambda *a, **k: FakeResponse(served))
    win = [[0.0] * WINDOW_LEN, [0.0] * WINDOW_LEN]
    r = TestClient(gw.app).post("/predict", json={"user_id": "u1", "window": win})
    assert r.status_code == 200 and r.json() == served
