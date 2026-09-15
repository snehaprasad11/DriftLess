"""Self-contained public demo (Phase 7 client, deployable as ONE service).

The full system is microservices (gateway + inference + worker + Redis); for a
single public URL this app collapses them into one process: it serves the
streaming web client, predicts over a WebSocket, and runs per-user calibration
in a background thread (instead of a Redis queue + separate worker). Same model
code, same result — just packaged for easy hosting.
"""
import os
import time
import uuid
import threading
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from model.registry import ModelRegistry
from model.windowing import apply_normaliser
from model.calibrate import calibrate
from model.schemas import CalibrateRequest, CalibrateResponse

reg = ModelRegistry(os.environ.get("MODELS_DIR", "models"))
CAL_STEPS = int(os.environ.get("CAL_STEPS", "60"))  # fewer on tiny free CPUs
app = FastAPI(title="DriftLess demo")
_cache: dict = {}          # cache key -> (model, which)
_calibrating: set = set()  # user_ids with a calibration in flight


def _get_model(user_id: str):
    key = user_id if reg.has_personal(user_id) else "__base__"
    if key not in _cache:
        _cache[key] = reg.load_model(user_id)
    return _cache[key]


def _predict(user_id: str, window) -> dict:
    x = np.asarray(window, dtype=np.float32)
    n = reg.norm
    xn = apply_normaliser(x[None], n["in_mean"], n["in_std"]).astype(np.float32)
    model, which = _get_model(user_id)
    with torch.no_grad():
        gaze, blink_logit = model(torch.from_numpy(xn))
    gaze_deg = gaze.numpy()[0] * n["g_std"] + n["g_mean"]
    return {"gaze": {"x": float(gaze_deg[0]), "y": float(gaze_deg[1])},
            "blink": bool(torch.sigmoid(blink_logit).item() > 0.5), "model": which}


def _run_calibration(user_id: str, trials: list):
    try:
        X = np.asarray([t["window"] for t in trials], dtype=np.float32)
        gaze = np.asarray([t["gaze"] for t in trials], dtype=np.float32)
        n = reg.norm
        Xn = apply_normaliser(X, n["in_mean"], n["in_std"]).astype(np.float32)
        gz = ((gaze - n["g_mean"]) / n["g_std"]).astype(np.float32)
        state = calibrate(reg.base_state(), torch.from_numpy(Xn), torch.from_numpy(gz), steps=CAL_STEPS)
        reg.save_personal(user_id, state)
        _cache.pop(user_id, None)
        _cache.pop("__base__", None)   # so the next predict picks up the personal model
    finally:
        _calibrating.discard(user_id)


@app.get("/")
def root():
    return RedirectResponse(url="/demo/")


@app.get("/health")
def health():
    return {"status": "ok", "cached_models": len(_cache)}


@app.websocket("/ws")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_json()
            t0 = time.perf_counter()
            out = _predict(msg["user_id"], msg["window"])
            out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            await websocket.send_json(out)
    except WebSocketDisconnect:
        pass


@app.post("/calibrate", response_model=CalibrateResponse)
def calibrate_endpoint(req: CalibrateRequest):
    """Kick off calibration in the background and return immediately (like the queue)."""
    if req.user_id not in _calibrating:
        _calibrating.add(req.user_id)
        threading.Thread(target=_run_calibration,
                         args=(req.user_id, [t.model_dump() for t in req.trials]),
                         daemon=True).start()
    return CalibrateResponse(status="queued", job_id=uuid.uuid4().hex[:8], n_trials=len(req.trials))


# serve the streaming web client at /demo
_CLIENT_DIR = Path(__file__).resolve().parent.parent / "client"
app.mount("/demo", StaticFiles(directory=str(_CLIENT_DIR), html=True), name="demo")
