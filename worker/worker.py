"""Calibration worker — the novel core as a background job (Phase 5).

Blocks on the Redis queue; for each job it few-shot fine-tunes the drift adapter
on the user's ~10 trials (model.calibrate), writes the personal model to the
registry, and tells the inference service to reload it. Running a second copy of
this worker when the queue grows is the horizontal-scaling demo.
"""
import os
import json
import time

import numpy as np
import torch
import redis
import httpx

from model.registry import ModelRegistry
from model.calibrate import calibrate
from model.windowing import apply_normaliser

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
INFERENCE_URL = os.environ.get("INFERENCE_URL", "http://localhost:8001")
QUEUE = "calib_jobs"
CAL_STEPS = int(os.environ.get("CAL_STEPS", "60"))

reg = ModelRegistry(os.environ.get("MODELS_DIR", "models"))
r = redis.from_url(REDIS_URL)


def run_job(job: dict) -> str:
    user_id = job["user_id"]
    trials = job["trials"]
    X = np.asarray([t["window"] for t in trials], dtype=np.float32)      # (n, 2, 320)
    gaze_deg = np.asarray([t["gaze"] for t in trials], dtype=np.float32)  # (n, 2)

    n = reg.norm
    Xn = apply_normaliser(X, n["in_mean"], n["in_std"]).astype(np.float32)
    gaze_std = ((gaze_deg - n["g_mean"]) / n["g_std"]).astype(np.float32)

    t0 = time.time()
    state = calibrate(reg.base_state(), torch.from_numpy(Xn), torch.from_numpy(gaze_std),
                      steps=CAL_STEPS)
    reg.save_personal(user_id, state)
    dt = time.time() - t0

    try:  # ask inference to pick up the new personal model
        httpx.post(f"{INFERENCE_URL}/reload/{user_id}", timeout=5)
    except httpx.HTTPError:
        pass
    return f"job {job['job_id']}: calibrated {user_id} on {len(trials)} trials in {dt:.1f}s"


def main():
    print(f"[worker] waiting for jobs on '{QUEUE}' ({REDIS_URL})", flush=True)
    while True:
        item = r.blpop(QUEUE, timeout=5)
        if item is None:
            continue
        try:
            print(run_job(json.loads(item[1])), flush=True)
        except Exception as e:  # a bad job must not kill the worker
            print(f"[worker] job failed: {e}", flush=True)


if __name__ == "__main__":
    main()
