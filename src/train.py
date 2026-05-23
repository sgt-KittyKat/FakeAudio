from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AudioFeatureDataset
from model import LCNN


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in tqdm(loader, desc="  train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * len(y)
        all_scores.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(y.cpu().numpy())
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    return (
        total_loss / len(loader.dataset),
        compute_eer(labels, scores),
        float(roc_auc_score(labels, scores)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-type", choices=["mfcc", "lfcc", "cqcc"], required=True)
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.75)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device} | Feature: {args.feature_type.upper()}")

    feature_dir = args.processed_root / args.feature_type
    train_ds = AudioFeatureDataset(feature_dir, "train")
    val_ds = AudioFeatureDataset(feature_dir, "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = LCNN(dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    args.model_dir.mkdir(parents=True, exist_ok=True)
    best_eer = float("inf")
    best_path = args.model_dir / f"lcnn_{args.feature_type}_best.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_eer, val_auc = evaluate(model, val_loader, criterion, device)
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_eer": round(val_eer, 6),
            "val_auc": round(val_auc, 6),
        })

        marker = ""
        if val_eer < best_eer:
            best_eer = val_eer
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_eer": val_eer,
                "val_auc": val_auc,
                "args": vars(args),
            }, best_path)
            marker = "  ← best"

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_EER={val_eer:.4f} | "
            f"val_AUC={val_auc:.4f}{marker}"
        )

    (args.model_dir / f"lcnn_{args.feature_type}_history.json").write_text(
        json.dumps(history, indent=2)
    )
    print(f"\nBest val EER: {best_eer:.4f}  →  {best_path}")


if __name__ == "__main__":
    main()
