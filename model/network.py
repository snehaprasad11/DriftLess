"""Phase 3 — the shared two-headed model (GazeNet).

Data flow (report Figure 2):

    raw window (2, 320)
        -> 1D-CNN            local shapes: saccade steps, blink spikes
        -> BiLSTM            how those shapes unfold over time
        -> drift adapter     the ONLY per-user part (fine-tuned in Phase 4)
        -> gaze head         (x, y) displacement in degrees   [regression]
        -> blink head        blink? yes/no                    [classification]

The backbone (CNN + BiLSTM) is shared across everyone and frozen during
per-user calibration; only the small bottleneck `DriftAdapter` moves, which is
what makes calibration a fast, few-shot, ~20 s job.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CHANNELS = 2      # horizontal, vertical EOG
FEATURE_DIM = 128   # backbone output width


class _ConvBlock(nn.Module):
    """Conv1d -> BatchNorm -> ReLU -> MaxPool(2)."""

    def __init__(self, c_in: int, c_out: int, kernel: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=kernel, padding=kernel // 2),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

    def forward(self, x):
        return self.net(x)


class CNNBiLSTMBackbone(nn.Module):
    """Shared feature extractor: 1D-CNN over the window, then a BiLSTM over time."""

    def __init__(self, feature_dim: int = FEATURE_DIM):
        super().__init__()
        self.cnn = nn.Sequential(
            _ConvBlock(N_CHANNELS, 32, kernel=7),   # 320 -> 160
            _ConvBlock(32, 64, kernel=5),           # 160 -> 80
            _ConvBlock(64, 128, kernel=3),          # 80  -> 40
        )
        # BiLSTM: hidden*2 (both directions) == feature_dim
        assert feature_dim % 2 == 0
        self.lstm = nn.LSTM(input_size=128, hidden_size=feature_dim // 2,
                            batch_first=True, bidirectional=True)
        self.feature_dim = feature_dim

    def forward(self, x):                 # x: (B, 2, 320)
        c = self.cnn(x)                   # (B, 128, 40)
        c = c.permute(0, 2, 1)            # (B, 40, 128) -> (batch, seq, feat)
        _out, (h_n, _c_n) = self.lstm(c)  # h_n: (2, B, feat/2)
        feat = torch.cat([h_n[0], h_n[1]], dim=1)  # (B, feature_dim)
        return feat


class DriftAdapter(nn.Module):
    """Small residual bottleneck adapter — the per-user part.

    feat -> down(feat) -> ReLU -> up(...) + feat. Tiny by design so it can be
    fine-tuned from only ~10 of a new user's trials without overfitting.
    """

    def __init__(self, feature_dim: int = FEATURE_DIM, bottleneck: int = 32):
        super().__init__()
        self.down = nn.Linear(feature_dim, bottleneck)
        self.up = nn.Linear(bottleneck, feature_dim)

    def forward(self, feat):
        return feat + self.up(F.relu(self.down(feat)))


class GazeNet(nn.Module):
    """Backbone -> drift adapter -> (gaze head, blink head)."""

    def __init__(self, feature_dim: int = FEATURE_DIM, adapter_bottleneck: int = 32):
        super().__init__()
        self.backbone = CNNBiLSTMBackbone(feature_dim)
        self.drift_adapter = DriftAdapter(feature_dim, adapter_bottleneck)
        self.gaze_head = nn.Linear(feature_dim, 2)   # x, y displacement
        self.blink_head = nn.Linear(feature_dim, 1)  # blink logit

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.drift_adapter(feat)
        return self.gaze_head(feat), self.blink_head(feat)

    # --- helpers for Phase 4 (per-user calibration) ---
    def freeze_backbone(self):
        """Freeze the shared backbone so only the adapter (and heads) adapt."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def adapter_parameters(self):
        return self.drift_adapter.parameters()


def gaze_blink_loss(gaze_pred, blink_logit, y_gaze, y_blink, lambda_blink: float = 1.0):
    """Combined loss.

    Gaze = MSE on saccade windows only (blink windows have NaN gaze and are
    masked out). Blink = binary cross-entropy on all windows.
    Returns (total, gaze_loss, blink_loss).
    """
    blink_target = y_blink.float().view(-1, 1)
    blink_loss = F.binary_cross_entropy_with_logits(blink_logit, blink_target)

    sacc = (y_blink == 0)
    if sacc.any():
        gaze_loss = F.mse_loss(gaze_pred[sacc], y_gaze[sacc])
    else:
        gaze_loss = gaze_pred.new_tensor(0.0)

    return gaze_loss + lambda_blink * blink_loss, gaze_loss, blink_loss


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
