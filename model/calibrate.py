"""Phase 4 — the novel core: per-user few-shot calibration.

For a brand-new user we take only ~10 of their trials, FREEZE the shared
backbone, and fine-tune just the tiny drift adapter (+ the two heads) for a few
steps. That locks the model onto *this person's* baseline-drift signature
without touching the shared knowledge — a fast, cheap, ~20 s cloud job.

    calibrate(base_state, cal_X, cal_gaze) -> personal state_dict

`cal_gaze` must be the STANDARDISED gaze targets (same scaling used in training),
and only saccade windows should be passed (blinks have no gaze target).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .network import GazeNet, count_parameters


def calibrate(base_state: dict,
              cal_X: torch.Tensor,
              cal_gaze: torch.Tensor,
              steps: int = 60,
              lr: float = 1e-3,
              device: str = "cpu") -> dict:
    """Few-shot fine-tune the drift adapter (+heads) on one user's trials.

    Parameters
    ----------
    base_state : shared base-model state_dict.
    cal_X      : (n, 2, L) the user's saccade windows (input-normalised).
    cal_gaze   : (n, 2) standardised gaze targets for those windows.
    steps      : gradient steps (report default 60).

    Returns the personalised state_dict.
    """
    net = GazeNet().to(device)
    net.load_state_dict(base_state)
    net.eval()               # keep frozen BatchNorm on its running stats
    net.freeze_backbone()    # only the adapter + heads remain trainable

    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)

    X, g = cal_X.to(device), cal_gaze.to(device)
    for _ in range(steps):
        opt.zero_grad()
        gaze_pred, _blink = net(X)
        loss = F.mse_loss(gaze_pred, g)   # drift shows up in gaze; calibrate on gaze
        loss.backward()
        opt.step()

    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def n_trainable(base_state: dict, device: str = "cpu") -> int:
    """How many parameters calibration actually updates (adapter + heads)."""
    net = GazeNet().to(device)
    net.load_state_dict(base_state)
    net.freeze_backbone()
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
