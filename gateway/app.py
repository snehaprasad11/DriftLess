"""API gateway — the single front door (Phase 5).

Receives requests, and either forwards a prediction to the inference service or
drops a calibration job on the Redis queue and answers immediately (it never
blocks on the ~20 s calibration). Deliberately lightweight: no torch here.
"""
import os
import json
import time
import uuid
from pathlib import Path

import httpx
import redis
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from model.schemas import (PredictRequest, PredictResponse,
                           CalibrateRequest, CalibrateResponse)

INFERENCE_URL = os.environ.get("INFERENCE_URL", "http://localhost:8001")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
QUEUE = "calib_jobs"

r = redis.from_url(REDIS_URL)
app = FastAPI(title="DriftLess gateway")


@app.get("/health")
def health():
    try:
        depth = r.llen(QUEUE)
        redis_ok = True
    except Exception:
        depth, redis_ok = None, False
    return {"status": "ok", "redis": redis_ok, "queue_depth": depth}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Forward to the inference service (fast path)."""
    try:
        resp = httpx.post(f"{INFERENCE_URL}/predict", json=req.model_dump(), timeout=10)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"inference service unreachable: {e}")
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, resp.text)
    return resp.json()


@app.post("/calibrate", response_model=CalibrateResponse)
def calibrate(req: CalibrateRequest):
    """Enqueue a calibration job and return immediately — never wait for it."""
    job_id = uuid.uuid4().hex[:8]
    r.rpush(QUEUE, json.dumps({"job_id": job_id, **req.model_dump()}))
    return CalibrateResponse(status="queued", job_id=job_id, n_trials=len(req.trials))


@app.websocket("/ws")
async def ws_stream(websocket: WebSocket):
    """Live streaming demo: the client sends {user_id, window} frames; we forward
    each to the inference service and stream back the prediction + round-trip
    latency (so the page can show cold-start vs warm behaviour)."""
    await websocket.accept()
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            while True:
                msg = await websocket.receive_json()
                t0 = time.perf_counter()
                resp = await client.post(f"{INFERENCE_URL}/predict",
                                         json={"user_id": msg["user_id"], "window": msg["window"]})
                latency_ms = round((time.perf_counter() - t0) * 1000, 1)
                if resp.status_code != 200:
                    await websocket.send_json({"error": resp.text, "latency_ms": latency_ms})
                    continue
                data = resp.json()
                data["latency_ms"] = latency_ms
                await websocket.send_json(data)
    except WebSocketDisconnect:
        pass


# Serve the streaming web client (Phase 7) at /demo (same origin as /ws, /calibrate).
_CLIENT_DIR = Path(__file__).resolve().parent.parent / "client"
if _CLIENT_DIR.is_dir():
    app.mount("/demo", StaticFiles(directory=str(_CLIENT_DIR), html=True), name="demo")
