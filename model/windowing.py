"""Phase 2 — turn raw EOG recordings into fixed-length, labelled windows.

The dataset gives, per subject, a continuous ~20 min recording of 2 EOG channels
(horizontal, vertical) plus a per-sample control signal marking forward saccades
(1), return saccades (2) and blinks (3), and a table of target gaze angles.

We cut one fixed-length window around each saccade (and each blink) and attach:
  * a **gaze** label  — the (x, y) displacement of that saccade, in degrees
  * a **blink** flag  — 1 for blink windows, 0 for saccade windows

Design decisions (see notebooks/02_windowing.ipynb for the reasoning):
  * WINDOW_LEN = 320 samples = 64-sample pre-roll + 256 samples (~1.0 s) from
    the saccade onset. The pre-roll lets the model see the *starting baseline*,
    which is where drift shows up.
  * Gaze label = displacement: forward saccade = +T, return saccade = -T, where
    T is the trial's target angle. (TargetGA stores absolute targets: forward
    rows = T, return rows = (0, 0); displacement = target_after - target_before.)
  * Normalisation is deliberately NOT per-window (that would subtract each
    window's own baseline = the drift we want to keep). Use fixed global
    per-channel stats fitted on the TRAIN subjects only; see fit_normaliser().
  * Blink windows get gaze = NaN so the gaze loss can mask them.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np

FS = 256                      # Hz (from the Data Description)
PRE_ROLL = 64                 # samples before saccade onset (~0.25 s)
POST = 256                    # samples from onset (~1.0 s)
WINDOW_LEN = PRE_ROLL + POST  # 320 samples per window

FWD, RET, BLINK = 1, 2, 3     # control-signal labels


# --------------------------------------------------------------------------- #
# Loading & segmenting
# --------------------------------------------------------------------------- #
def load_subject(data_dir: Path | str, subject: str):
    """Return (eog (2, N), control (N,), target_ga (K, 2)) for e.g. subject='S1'."""
    from scipy.io import loadmat  # lazy: only notebooks need SciPy, not the services
    d = Path(data_dir) / subject
    eog = loadmat(d / "EOG.mat")["EOG"].astype(np.float64)          # (2, N)
    ctrl = loadmat(d / "ControlSignal.mat")["ControlSignal"].ravel()  # (N,)
    tga = loadmat(d / "TargetGA.mat")["TargetGA"].astype(np.float64)  # (K, 2)
    return eog, ctrl, tga


def segment_runs(ctrl: np.ndarray):
    """Run-length encode the control signal.

    Returns a list of (label, start, end) for each consecutive run.
    """
    changes = np.where(np.diff(ctrl) != 0)[0] + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(ctrl)]))
    return [(int(ctrl[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def _slice(eog: np.ndarray, start: int, length: int = WINDOW_LEN) -> np.ndarray:
    """Extract a (2, length) window starting at `start`, edge-padding at bounds."""
    n = eog.shape[1]
    s, e = start, start + length
    left, right = max(0, -s), max(0, e - n)
    w = eog[:, max(0, s):min(n, e)]
    if left or right:
        w = np.pad(w, ((0, 0), (left, right)), mode="edge")
    return w


# --------------------------------------------------------------------------- #
# Windowing one subject
# --------------------------------------------------------------------------- #
@dataclass
class SubjectWindows:
    X: np.ndarray        # (n, 2, WINDOW_LEN) float32 — raw (un-normalised) windows
    y_gaze: np.ndarray   # (n, 2) float32 — (dx, dy) degrees; NaN for blink windows
    y_blink: np.ndarray  # (n,) int64 — 1 for blink, 0 for saccade
    subject: str

    def __len__(self):
        return len(self.X)


def windows_for_subject(data_dir: Path | str, subject: str) -> SubjectWindows:
    """Extract saccade + blink windows for one subject."""
    eog, ctrl, tga = load_subject(data_dir, subject)
    runs = segment_runs(ctrl)

    X, y_gaze, y_blink = [], [], []
    sacc_idx = 0  # index into TargetGA rows: j-th saccade run <-> TargetGA[j]
    for label, start, _end in runs:
        if label in (FWD, RET):
            target_after = tga[sacc_idx]
            target_before = tga[sacc_idx - 1] if sacc_idx > 0 else np.zeros(2)
            displacement = target_after - target_before  # fwd=+T, ret=-T
            X.append(_slice(eog, start - PRE_ROLL))
            y_gaze.append(displacement)
            y_blink.append(0)
            sacc_idx += 1
        elif label == BLINK:
            X.append(_slice(eog, start - PRE_ROLL))
            y_gaze.append([np.nan, np.nan])   # gaze undefined during a blink
            y_blink.append(1)
        # any other label (e.g. the single stray '30' sample) is ignored

    return SubjectWindows(
        X=np.asarray(X, dtype=np.float32),
        y_gaze=np.asarray(y_gaze, dtype=np.float32),
        y_blink=np.asarray(y_blink, dtype=np.int64),
        subject=subject,
    )


# --------------------------------------------------------------------------- #
# Normalisation — drift-preserving (fit on TRAIN subjects only)
# --------------------------------------------------------------------------- #
def fit_normaliser(X: np.ndarray):
    """Fit per-channel (mean, std) over all windows/timesteps of the TRAIN set.

    This rescales channels to a comparable range while PRESERVING the between-
    window baseline differences (the drift). It does not recentre each window.
    Returns arrays of shape (2, 1) for broadcasting over (n, 2, L).
    """
    mean = X.mean(axis=(0, 2), keepdims=False).reshape(2, 1)
    std = X.std(axis=(0, 2), keepdims=False).reshape(2, 1)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_normaliser(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply fixed (mean, std) from fit_normaliser to windows (n, 2, L)."""
    return (X - mean[None]) / std[None]


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_subject(out_dir: Path | str, sw: SubjectWindows) -> Path:
    """Save one subject's raw windows to data/processed/<subject>_windows.npz."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{sw.subject}_windows.npz"
    np.savez_compressed(path, X=sw.X, y_gaze=sw.y_gaze,
                        y_blink=sw.y_blink, subject=sw.subject)
    return path


def load_subject_windows(path: Path | str) -> SubjectWindows:
    d = np.load(path, allow_pickle=True)
    return SubjectWindows(X=d["X"], y_gaze=d["y_gaze"],
                          y_blink=d["y_blink"], subject=str(d["subject"]))
