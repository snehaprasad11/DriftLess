"""Tests for the model registry (base vs per-user model resolution)."""
import torch

from model.registry import ModelRegistry
from model.network import GazeNet


def test_base_then_personal_resolution(tmp_registry):
    reg = ModelRegistry(tmp_registry)

    # normalisers load
    assert set(reg.norm) == {"in_mean", "in_std", "g_mean", "g_std"}

    # a brand-new user has no personal model -> base is served
    assert reg.has_personal("newuser") is False
    _model, which = reg.load_model("newuser")
    assert which == "base"

    # after calibration writes a personal model, it takes over
    personal_state = GazeNet().state_dict()
    reg.save_personal("newuser", personal_state)
    assert reg.has_personal("newuser") is True
    _model, which = reg.load_model("newuser")
    assert which == "personal"


def test_base_state_is_loadable_into_gazenet(tmp_registry):
    reg = ModelRegistry(tmp_registry)
    net = GazeNet()
    net.load_state_dict(reg.base_state())   # must not raise
