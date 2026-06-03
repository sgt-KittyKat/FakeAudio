from __future__ import annotations

import torch
import torch.nn as nn


class MFM(nn.Module):
    """Max-Feature-Map activation: splits channels in half, takes element-wise max."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return torch.max(x1, x2)


class LCNN(nn.Module):
    """
    Light CNN with Max-Feature-Map activations for deepfake audio detection.

    Input:  [B, 3, 20, 400]
    Output: [B] logits  (positive → fake)

    Spatial trace:
      [B,  3, 20, 400]
      [B, 32, 20, 400]  after conv1 + MFM
      [B, 32, 10, 200]  after pool1
      [B, 48, 10, 200]  after conv2/3 + MFM
      [B, 48,  5, 100]  after pool2
      [B, 64,  5, 100]  after conv4/5 + MFM
      [B, 64,  2,  50]  after pool3
      [B, 32,  2,  50]  after conv6/7 + MFM
      [B, 32,  1,  25]  after pool4  → flatten → 800-d
    """

    def __init__(self, dropout: float = 0.75) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            MFM(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=1),
            MFM(),
            nn.Conv2d(32, 96, kernel_size=3, padding=1),
            MFM(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(48, 96, kernel_size=1),
            MFM(),
            nn.Conv2d(48, 128, kernel_size=3, padding=1),
            MFM(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=1),
            MFM(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # index -3: GradCAM target
            MFM(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(800, 256),
            MFM(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x).squeeze(1)
