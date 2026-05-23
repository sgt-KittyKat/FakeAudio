from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from dataset import AudioFeatureDataset
from model import LCNN


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thresholds[idx])


@torch.no_grad()
def run_inference(model: LCNN, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_scores, all_labels = [], []
    for x, y in loader:
        logits = model(x.to(device))
        all_scores.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(y.numpy())
    return np.concatenate(all_scores), np.concatenate(all_labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-type", choices=["mfcc", "lfcc", "cqcc"], required=True)
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    checkpoint_path = args.model_dir / f"lcnn_{args.feature_type}_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}. Run train.py first.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = LCNN(dropout=0.0).to(device)
    model.load_state_dict(checkpoint["model_state"])

    dataset = AudioFeatureDataset(args.processed_root / args.feature_type, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    scores, labels = run_inference(model, loader, device)

    auc = roc_auc_score(labels, scores)
    eer, eer_threshold = compute_eer(labels, scores)
    preds = (scores >= eer_threshold).astype(int)
    f1 = f1_score(labels, preds)

    print(f"\nResults — {args.feature_type.upper()} | split={args.split} | best ckpt epoch={checkpoint['epoch']}")
    print(f"  ROC-AUC : {auc:.4f}")
    print(f"  EER     : {eer:.4f}  (threshold={eer_threshold:.4f})")
    print(f"  F1      : {f1:.4f}")


if __name__ == "__main__":
    main()
