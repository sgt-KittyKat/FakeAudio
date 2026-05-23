from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

_cache = Path(tempfile.gettempdir()) / "matplotlib-cache"
_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache))

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from dataset import AudioFeatureDataset
from model import LCNN


class GradCAM:
    """GradCAM for LCNN — hooks the last Conv2d layer (features[-3])."""

    def __init__(self, model: LCNN) -> None:
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        target = model.features[-3]  # Conv2d(64, 64, 3, padding=1)
        target.register_forward_hook(self._fwd_hook)
        target.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out) -> None:
        self._activations = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out) -> None:
        self._gradients = grad_out[0].detach()

    def compute(self, x: torch.Tensor) -> tuple[np.ndarray, float]:
        """Returns (cam [20, 400] normalized 0-1, sigmoid score)."""
        self.model.eval()
        logit = self.model(x.unsqueeze(0))
        score = torch.sigmoid(logit).item()

        self.model.zero_grad()
        logit.backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # [1, C, 1, 1]
        cam = F.relu((weights * self._activations).sum(dim=1, keepdim=True))  # [1, 1, H, W]
        cam = F.interpolate(cam, size=(20, 400), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        lo, hi = cam.min(), cam.max()
        if hi > lo:
            cam = (cam - lo) / (hi - lo)
        return cam, score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-type", choices=["mfcc", "lfcc", "cqcc"], required=True)
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--out-dir", type=Path, default=Path("xai_outputs"))
    parser.add_argument("--n-samples", type=int, default=8, help="Total samples (split evenly real/fake)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    checkpoint_path = args.model_dir / f"lcnn_{args.feature_type}_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path}. Run train.py first.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = LCNN(dropout=0.0).to(device)
    model.load_state_dict(checkpoint["model_state"])

    gradcam = GradCAM(model)

    dataset = AudioFeatureDataset(args.processed_root / args.feature_type, args.split)
    labels_np = dataset.labels.numpy()
    half = args.n_samples // 2
    real_idx = np.where(labels_np == 0)[0][:half]
    fake_idx = np.where(labels_np == 1)[0][:half]
    indices = np.concatenate([real_idx, fake_idx])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(indices), 2, figsize=(14, 3.2 * len(indices)), constrained_layout=True)

    for row, idx in enumerate(indices):
        x, label = dataset[int(idx)]
        x = x.to(device)
        cam, score = gradcam.compute(x)
        static = x[0].cpu().numpy()  # static coefficients channel
        label_name = "fake" if int(label) == 1 else "real"

        ax_feat, ax_cam = axes[row, 0], axes[row, 1]

        ax_feat.imshow(static, aspect="auto", origin="lower", cmap="viridis")
        ax_feat.set_title(f"[{label_name}] Static coefficients", fontsize=9)
        ax_feat.set_ylabel("Coeff index")
        ax_feat.set_xlabel("Time frame")

        ax_cam.imshow(static, aspect="auto", origin="lower", cmap="viridis", alpha=0.55)
        ax_cam.imshow(cam, aspect="auto", origin="lower", cmap="jet", alpha=0.5)
        ax_cam.set_title(f"GradCAM overlay  (score={score:.3f})", fontsize=9)
        ax_cam.set_ylabel("Coeff index")
        ax_cam.set_xlabel("Time frame")

    fig.suptitle(
        f"GradCAM — {args.feature_type.upper()} | {args.split} split | "
        f"left=feature, right=attention heatmap",
        fontsize=12,
    )
    out_path = args.out_dir / f"{args.feature_type}_gradcam.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
