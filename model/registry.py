"""The model registry — one versioned source of truth for models.

Every service reads models from here instead of holding its own copy: the
inference service loads a user's model, the calibration worker writes a new
personal model, and both share the base model + normalisers.

This is the *local filesystem* implementation (Phase 5, "run locally first").
It has the same small interface an S3/object-storage version would (Phase 6),
so swapping the backend later touches only this file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .network import GazeNet


class ModelRegistry:
    def __init__(self, models_dir: Path | str):
        self.dir = Path(models_dir)
        self._norm = None  # cached normalisers

    # --- normalisers (input + gaze target scaling) ---
    @property
    def norm(self) -> dict:
        if self._norm is None:
            d = np.load(self.dir / "base_norm.npz", allow_pickle=True)
            self._norm = {k: d[k] for k in ("in_mean", "in_std", "g_mean", "g_std")}
        return self._norm

    # --- model state dicts ---
    def base_state(self) -> dict:
        return torch.load(self.dir / "base_model.pt", map_location="cpu", weights_only=True)

    def has_personal(self, user_id: str) -> bool:
        return (self.dir / f"personal_{user_id}.pt").exists()

    def personal_path(self, user_id: str) -> Path:
        return self.dir / f"personal_{user_id}.pt"

    def save_personal(self, user_id: str, state_dict: dict) -> Path:
        p = self.personal_path(user_id)
        torch.save(state_dict, p)
        return p

    def load_model(self, user_id: str | None = None) -> tuple[GazeNet, str]:
        """Return (model, which) — the user's personal model if it exists, else base."""
        net = GazeNet()
        if user_id and self.has_personal(user_id):
            net.load_state_dict(torch.load(self.personal_path(user_id),
                                           map_location="cpu", weights_only=True))
            which = "personal"
        else:
            net.load_state_dict(self.base_state())
            which = "base"
        net.eval()
        return net, which
