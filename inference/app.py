"""Inference service — fast, stateless gaze prediction (Phase 5).

Given a window + user_id, loads that user's model from the registry (cached in
memory) and returns predicted gaze + blink. Stateless: the model always comes
from the registry, so many copies can run and scale to zero.
"""
import os

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

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


@app.get("/", response_class=HTMLResponse)
def root():
    """Friendly landing page so the bare URL shows the service is alive."""
    return """<!doctype html><html><head><meta charset="utf-8">
<title>DriftLess Inference API</title>
<style>body{font-family:system-ui,sans-serif;max-width:640px;margin:56px auto;padding:0 20px;
color:#1a2b32;line-height:1.6}code{background:#eef3f5;padding:2px 6px;border-radius:4px}
a{color:#2a9d8f}h1{margin-bottom:4px}</style></head><body>
<h1>&#128065; DriftLess &mdash; Inference API</h1>
<p><b>Status: live.</b> Cloud auto-calibrating EOG gaze estimation.</p>
<h3>Endpoints</h3>
<ul>
<li><code>GET</code> <a href="/health">/health</a> &mdash; service status</li>
<li><code>POST</code> <code>/predict</code> &mdash; body <code>{"user_id": "...", "window": [[...],[...]]}</code>
(a 2&times;320 EOG window) &rarr; <code>{"gaze": {"x","y"}, "blink", "model"}</code></li>
</ul>
<p>Project &amp; source: <a href="https://github.com/snehaprasad11/DriftLess">github.com/snehaprasad11/DriftLess</a></p>
</body></html>"""


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
