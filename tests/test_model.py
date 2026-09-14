"""Unit tests for the ML core: windowing helpers, the network, and calibration."""
import numpy as np
import torch

from model.windowing import (segment_runs, _slice, fit_normaliser, apply_normaliser,
                              WINDOW_LEN)
from model.network import GazeNet, gaze_blink_loss, count_parameters
from model.calibrate import calibrate, n_trainable


# --- windowing -------------------------------------------------------------
def test_segment_runs():
    ctrl = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3])
    runs = segment_runs(ctrl)
    assert runs == [(1, 0, 3), (2, 3, 5), (3, 5, 9)]


def test_slice_edge_pads_at_bounds():
    eog = np.arange(20, dtype=np.float32).reshape(2, 10)
    w = _slice(eog, start=-3, length=WINDOW_LEN)      # starts before 0
    assert w.shape == (2, WINDOW_LEN)
    assert w[0, 0] == eog[0, 0]                        # left edge-padded


def test_normaliser_is_drift_preserving():
    # two windows with very different baselines (simulated drift)
    a = np.full((1, 2, WINDOW_LEN), 100.0, dtype=np.float32)
    b = np.full((1, 2, WINDOW_LEN), -400.0, dtype=np.float32)
    X = np.concatenate([a, b]) + np.random.randn(2, 2, WINDOW_LEN).astype(np.float32)
    mean, std = fit_normaliser(X)
    Xn = apply_normaliser(X, mean, std)
    per_window_mean = Xn.mean(axis=2)                  # (2 windows, 2 ch)
    # the baseline gap between windows must survive (NOT recentred to ~0)
    assert abs(per_window_mean[0, 0] - per_window_mean[1, 0]) > 1.0


# --- network ---------------------------------------------------------------
def test_forward_shapes_and_masked_loss():
    torch.manual_seed(0)
    net = GazeNet()
    X = torch.randn(8, 2, WINDOW_LEN)
    gaze, blink = net(X)
    assert gaze.shape == (8, 2) and blink.shape == (8, 1)

    y_gaze = torch.randn(8, 2)
    y_blink = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    y_gaze[y_blink == 1] = float("nan")                # blink rows have no gaze
    total, gl, bl = gaze_blink_loss(gaze, blink, y_gaze, y_blink)
    assert torch.isfinite(total)                        # NaN gaze must be masked out


def test_adapter_is_small_fraction():
    net = GazeNet()
    frac = count_parameters(net.drift_adapter) / count_parameters(net)
    assert 0.0 < frac < 0.15                            # the per-user part is tiny


# --- calibration -----------------------------------------------------------
def test_calibrate_fits_the_user_and_freezes_backbone():
    torch.manual_seed(0)
    base = GazeNet()
    base_state = {k: v.clone() for k, v in base.state_dict().items()}

    X = torch.randn(24, 2, WINDOW_LEN)
    with torch.no_grad():
        target = base(X)[0] + 0.5      # a shifted target the adapter must learn
    before = torch.nn.functional.mse_loss(base(X)[0], target).item()

    personal = calibrate(base_state, X, target, steps=60)
    net = GazeNet(); net.load_state_dict(personal); net.eval()
    after = torch.nn.functional.mse_loss(net(X)[0], target).item()
    assert after < before                               # calibration reduced the error

    # backbone must be unchanged (frozen); adapter must have moved
    assert torch.equal(base_state["backbone.cnn.0.net.0.weight"],
                       personal["backbone.cnn.0.net.0.weight"])
    assert not torch.equal(base_state["drift_adapter.down.weight"],
                          personal["drift_adapter.down.weight"])


def test_n_trainable_excludes_backbone():
    base = GazeNet()
    total = count_parameters(base)
    trainable = n_trainable(base.state_dict())
    assert 0 < trainable < total                        # only adapter + heads train
