"""Shared API request/response schemas (Pydantic).

Imported by both the gateway and the inference service so their contracts stay
in sync. Pure Pydantic — no torch — so the lightweight gateway can import it.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

WINDOW_LEN = 320  # samples per window (must match model.windowing.WINDOW_LEN)


class Gaze(BaseModel):
    x: float  # horizontal displacement, degrees
    y: float  # vertical displacement, degrees


class PredictRequest(BaseModel):
    user_id: str
    window: list[list[float]] = Field(..., description="raw EOG window, shape (2, 320)")


class PredictResponse(BaseModel):
    gaze: Gaze
    blink: bool
    model: str  # "personal" or "base" — which model served the request


class Trial(BaseModel):
    window: list[list[float]] = Field(..., description="raw EOG saccade window (2, 320)")
    gaze: list[float] = Field(..., description="known (x, y) displacement in degrees")


class CalibrateRequest(BaseModel):
    user_id: str
    trials: list[Trial] = Field(..., description="~10 labelled saccade windows from the new user")


class CalibrateResponse(BaseModel):
    status: str    # "queued"
    job_id: str
    n_trials: int
