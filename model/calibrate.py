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
              weight_decay: float = 1e-3,
              device: str = "cpu") -> dict:
    """Few-shot fine-tune the drift adapter on one user's trials.

    Only the small **drift adapter** is trained — the backbone AND the output
    heads stay frozen. Training the heads on ~10 examples overfits and can
    diverge; restricting to the adapter (plus mild weight decay) keeps few-shot
    calibration stable, and matches the design: the adapter is the per-user part.

    Parameters
    ----------
    base_state   : shared base-model state_dict.
    cal_X        : (n, 2, L) the user's saccade windows (input-normalised).
    cal_gaze     : (n, 2) standardised gaze targets for those windows.
    steps        : gradient steps (report default 60).
    weight_decay : L2 regularisation on the adapter (the report's "mild
                   regularisation" to prevent overfitting).

    Returns the personalised state_dict.
    """
    net = GazeNet().to(device)
    net.load_state_dict(base_state)
    net.eval()                                   # frozen BatchNorm on running stats
    for p in net.parameters():
        p.requires_grad_(False)
    for p in net.drift_adapter.parameters():     # only the drift adapter adapts
        p.requires_grad_(True)

    opt = torch.optim.Adam(net.drift_adapter.parameters(), lr=lr, weight_decay=weight_decay)

    X, g = cal_X.to(device), cal_gaze.to(device)
    # The backbone is frozen, so its features are CONSTANT across steps — compute
    # them once and iterate only the tiny adapter. Same result, far less compute
    # (crucial on small CPUs), since we no longer re-run the CNN+BiLSTM every step.
    with torch.no_grad():
        feat = net.backbone(X)
    for _ in range(steps):
        opt.zero_grad()
        gaze_pred = net.gaze_head(net.drift_adapter(feat))   # drift shows up in gaze
        loss = F.mse_loss(gaze_pred, g)
        loss.backward()
        opt.step()

    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def n_trainable(base_state: dict, device: str = "cpu") -> int:
    """How many parameters calibration actually updates (the drift adapter)."""
    net = GazeNet().to(device)
    net.load_state_dict(base_state)
    return sum(p.numel() for p in net.drift_adapter.parameters())
