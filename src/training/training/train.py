from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import FeatureDataset
from src.models.cnn_baseline import CNNBaseline


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, loader, optimizer, criterion):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader)

    for batch in progress:

        x = batch["features"].to(DEVICE)
        y = batch["label"].float().to(DEVICE)

        optimizer.zero_grad()

        logits = model(x).squeeze(1)

        loss = criterion(logits, y)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        preds = (torch.sigmoid(logits) >= 0.5).long()

        correct += (preds == y.long()).sum().item()

        total += y.size(0)

        progress.set_postfix(
            loss=loss.item(),
            acc=100 * correct / total
        )

    return running_loss / len(loader)


def validate(model, loader, criterion):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for batch in loader:

            x = batch["features"].to(DEVICE)
            y = batch["label"].float().to(DEVICE)

            logits = model(x).squeeze(1)

            loss = criterion(logits, y)

            running_loss += loss.item()

            preds = (torch.sigmoid(logits) >= 0.5).long()

            correct += (preds == y.long()).sum().item()

            total += y.size(0)

    acc = 100 * correct / total

    return running_loss / len(loader), acc


def main():

    feature_dir = "data/processed/mfcc"

    train_dataset = FeatureDataset(
        feature_dir=feature_dir,
        split="train",
        normalize=True,
    )

    val_dataset = FeatureDataset(
        feature_dir=feature_dir,
        split="val",
        normalize=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
    )

    model = CNNBaseline().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    epochs = 10

    for epoch in range(epochs):

        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

    torch.save(model.state_dict(), "cnn_baseline.pth")

    print("\nModel saved: cnn_baseline.pth")


if __name__ == "__main__":
    main()