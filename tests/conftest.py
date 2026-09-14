"""Shared test fixtures. Everything here is synthetic — no dataset or trained
model required — so the suite runs anywhere (including CI)."""
import sys
from pathlib import Path

import numpy as np
import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.network import GazeNet
from model.schemas import WINDOW_LEN


@pytest.fixture
def tmp_registry(tmp_path):
    """A models/ dir holding a fresh (untrained) base model + plausible normalisers."""
    torch.manual_seed(0)
    net = GazeNet()
    torch.save(net.state_dict(), tmp_path / "base_model.pt")
    np.savez(
        tmp_path / "base_norm.npz",
        in_mean=np.zeros((2, 1), dtype=np.float32),
        in_std=np.ones((2, 1), dtype=np.float32),
        g_mean=np.zeros(2, dtype=np.float32),
        g_std=np.array([13.6, 7.7], dtype=np.float32),
        train_subjects=np.array(["A"]),
        heldout_subjects=np.array(["B"]),
        sanity=True,
    )
    return tmp_path


@pytest.fixture
def rand_window():
    """One raw EOG window, shape (2, WINDOW_LEN)."""
    rng = np.random.default_rng(1)
    return (rng.standard_normal((2, WINDOW_LEN)) * 100).astype(np.float32)
