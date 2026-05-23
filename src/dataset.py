from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class AudioFeatureDataset(Dataset):
    def __init__(self, feature_dir: Path | str, split: str, normalize: bool = True):
        feature_dir = Path(feature_dir)

        x = np.load(feature_dir / f"{split}_features.npz")["X"]  # [N, 3, 20, 400]
        self.x = torch.from_numpy(x)

        meta = pd.read_csv(feature_dir / f"{split}_metadata.csv")
        self.labels = torch.tensor(meta["label"].values, dtype=torch.float32)

        if normalize:
            stats = json.loads((feature_dir / "feature_stats.json").read_text())
            mean = torch.tensor(stats["mean"], dtype=torch.float32).unsqueeze(-1)  # [3, 20, 1]
            std = torch.tensor(stats["std"], dtype=torch.float32).unsqueeze(-1)    # [3, 20, 1]
            self.x = (self.x - mean) / (std + 1e-8)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.labels[idx]
