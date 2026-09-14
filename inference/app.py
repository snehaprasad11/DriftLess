"""Inference service — fast, stateless gaze prediction (Phase 5).

Given a window + user_id, loads that user's model from the registry (cached in
memory) and returns predicted gaze + blink. Stateless: the model always comes
from the registry, so many copies can run and scale to zero.
"""
import os

import numpy as np
import torch
from fastapi import FastAPI, HTTPException

from model.registry import ModelRegistry
from model.windowing import apply_normaliser
from model.schemas import PredictRequest, PredictResponse, Gaze, WINDOW_LEN

reg = ModelRegistry(os.environ.get("MODELS_DIR", "models"))
app = FastAPI(title="DriftLess inference")
_cache: dict = {}  # cache key -> (model, which)


def _get_model(user_id: str):
    """Return (model, which); personal if the user has one, else the base model."""
    key = user_id if reg.has_personal(user_id) else "__base__"
    if key not in _cache:
        _cache[key] = reg.load_model(user_id)
    return _cache[key]


@app.get("/health")
def health():
    return {"status": "ok", "cached_models": len(_cache)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    x = np.asarray(req.window, dtype=np.float32)
    if x.shape != (2, WINDOW_LEN):
        raise HTTPException(400, f"window must be (2, {WINDOW_LEN}); got {list(x.shape)}")

    n = reg.norm
    xn = apply_normaliser(x[None], n["in_mean"], n["in_std"]).astype(np.float32)
    model, which = _get_model(req.user_id)
    with torch.no_grad():
        gaze, blink_logit = model(torch.from_numpy(xn))
    gaze_deg = gaze.numpy()[0] * n["g_std"] + n["g_mean"]
    blink = bool(torch.sigmoid(blink_logit).item() > 0.5)
    return PredictResponse(gaze=Gaze(x=float(gaze_deg[0]), y=float(gaze_deg[1])),
                           blink=blink, model=which)


@app.post("/reload/{user_id}")
def reload_user(user_id: str):
    """Evict a user's cached model so the next request picks up a freshly
    calibrated personal model. Called by the worker after calibration."""
    _cache.pop(user_id, None)
    _cache.pop("__base__", None)  # base entry may have masked a now-personal user
    return {"status": "reloaded", "user_id": user_id}
