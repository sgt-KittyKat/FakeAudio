from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List

_cache_dir = Path(tempfile.gettempdir()) / "matplotlib-cache"
_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


LABEL_NAMES = {0: "real", 1: "fake"}


def load_feature_store(feature_dir: Path, splits: Iterable[str]) -> tuple[np.ndarray, pd.DataFrame, dict]:
    arrays: list[np.ndarray] = []
    metadata_frames: list[pd.DataFrame] = []

    for split in splits:
        feature_path = feature_dir / f"{split}_features.npz"
        metadata_path = feature_dir / f"{split}_metadata.csv"
        if not feature_path.exists():
            raise FileNotFoundError(f"Missing feature file: {feature_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

        x = np.load(feature_path)["X"]
        metadata = pd.read_csv(metadata_path)
        if x.shape[0] != len(metadata):
            raise ValueError(
                f"Row mismatch for split '{split}': feature rows={x.shape[0]} metadata rows={len(metadata)}"
            )

        arrays.append(x)
        metadata_frames.append(metadata)

    if not arrays:
        raise ValueError("No splits were loaded.")

    config_path = feature_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    return np.concatenate(arrays, axis=0), pd.concat(metadata_frames, ignore_index=True), config


def build_chunk_summary_vectors(x: np.ndarray) -> np.ndarray:
    chunk_mean = x.mean(axis=-1)
    chunk_std = x.std(axis=-1)
    return np.concatenate([chunk_mean, chunk_std], axis=1).reshape(x.shape[0], -1).astype(np.float32)


def build_file_level_dataset(x: np.ndarray, metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    chunk_vectors = build_chunk_summary_vectors(x)
    grouped = metadata.groupby(["file", "speaker", "label"], sort=True).indices

    file_vectors: list[np.ndarray] = []
    file_labels: list[int] = []
    for (_, _, label), row_indices in grouped.items():
        file_vectors.append(chunk_vectors[row_indices].mean(axis=0))
        file_labels.append(int(label))

    return np.vstack(file_vectors).astype(np.float32), np.asarray(file_labels, dtype=np.int64)


def coefficient_class_means(x: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
    static_coeffs = x[:, 0, :, :].mean(axis=-1)
    means: Dict[int, np.ndarray] = {}
    for cls in sorted(np.unique(labels).tolist()):
        means[int(cls)] = static_coeffs[labels == cls]
    return means


def save_boxplot(per_class_means: Dict[int, np.ndarray], output_path: Path, top_k: int) -> None:
    class_ids = sorted(per_class_means.keys())
    if not class_ids:
        raise ValueError("No classes available for boxplot.")

    n_coeffs = per_class_means[class_ids[0]].shape[1]
    if len(class_ids) >= 2:
        diff = np.abs(per_class_means[class_ids[0]].mean(axis=0) - per_class_means[class_ids[1]].mean(axis=0))
        top_idx = np.argsort(diff)[::-1][: min(top_k, n_coeffs)]
    else:
        top_idx = np.arange(min(top_k, n_coeffs))

    rows = max(1, int(np.ceil(len(top_idx) / 4)))
    fig, axes = plt.subplots(rows, 4, figsize=(18, max(4, rows * 3.3)), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax_i, coeff_idx in enumerate(top_idx):
        ax = axes[ax_i]
        box_data = [per_class_means[c][:, coeff_idx] for c in class_ids]
        labels = [LABEL_NAMES.get(c, str(c)) for c in class_ids]
        ax.boxplot(box_data, tick_labels=labels, showfliers=False)
        ax.set_title(f"Static coeff {coeff_idx}", fontsize=10)
        ax.tick_params(axis="x", labelrotation=20)

    for ax in axes[len(top_idx) :]:
        ax.axis("off")

    fig.suptitle("Top Static-Coefficient Distributions by Class", fontsize=14)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def save_pca_scatter(file_vectors: np.ndarray, file_labels: np.ndarray, output_path: Path) -> None:
    pca = PCA(n_components=2, random_state=42)
    z = pca.fit_transform(file_vectors)

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    class_ids = sorted(np.unique(file_labels).tolist())
    cmap = plt.get_cmap("tab10")

    for k, cls in enumerate(class_ids):
        idx = np.where(file_labels == cls)[0]
        ax.scatter(
            z[idx, 0],
            z[idx, 1],
            s=14,
            alpha=0.65,
            label=f"{LABEL_NAMES.get(int(cls), cls)} (n={len(idx)})",
            color=cmap(k % 10),
            edgecolors="none",
        )

    var1 = pca.explained_variance_ratio_[0] * 100.0
    var2 = pca.explained_variance_ratio_[1] * 100.0
    ax.set_xlabel(f"PC1 ({var1:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var2:.1f}% var)")
    ax.set_title("PCA of File-Level Feature Summaries")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)

    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def print_top_diffs(per_class_means: Dict[int, np.ndarray], top_k: int) -> None:
    class_ids = sorted(per_class_means.keys())
    if len(class_ids) < 2:
        print("Need at least 2 classes for coefficient-difference ranking.")
        return

    c0, c1 = class_ids[0], class_ids[1]
    diff = per_class_means[c0].mean(axis=0) - per_class_means[c1].mean(axis=0)
    order = np.argsort(np.abs(diff))[::-1][:top_k]

    print(f"Top {top_k} static-coefficient mean differences: {LABEL_NAMES.get(c0, c0)} - {LABEL_NAMES.get(c1, c1)}")
    for idx in order:
        print(f"  coeff[{idx:02d}] diff={diff[idx]:+.4f}")


def print_dataset_summary(feature_type: str, x: np.ndarray, metadata: pd.DataFrame, config: dict) -> None:
    print(f"Feature type: {feature_type}")
    print(f"Loaded chunks: {x.shape[0]}")
    print(f"Tensor shape: {x.shape}")
    print(f"Unique files: {metadata['file'].nunique()}")
    print(f"Unique speakers: {metadata['speaker'].nunique()}")
    print("Chunk counts by split:", metadata["split"].value_counts().sort_index().to_dict())
    print("Chunk counts by label:", metadata["label"].value_counts().sort_index().to_dict())
    if config:
        keep = {
            "sample_rate": config.get("sample_rate"),
            "chunk_seconds": config.get("chunk_seconds"),
            "include_deltas": config.get("include_deltas"),
            "target_frames": config.get("target_frames"),
        }
        print("Config summary:", keep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect preprocessed MFCC/LFCC/CQCC feature stores.")
    parser.add_argument("--feature-type", choices=["mfcc", "lfcc", "cqcc"], required=True)
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"])
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10, help="Number of top coefficients to print and plot.")
    args = parser.parse_args()

    feature_dir = args.processed_root / args.feature_type
    if args.out_dir is None:
        out_dir = Path("feature_inspection") / args.feature_type
    else:
        out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    x, metadata, config = load_feature_store(feature_dir, args.splits)
    print_dataset_summary(args.feature_type, x, metadata, config)

    labels = metadata["label"].to_numpy(dtype=np.int64)
    per_class_means = coefficient_class_means(x, labels)
    file_vectors, file_labels = build_file_level_dataset(x, metadata)

    boxplot_path = out_dir / f"{args.feature_type}_coeff_boxplots.png"
    pca_path = out_dir / f"{args.feature_type}_pca_scatter.png"

    save_boxplot(per_class_means, boxplot_path, top_k=args.top_k)
    save_pca_scatter(file_vectors, file_labels, pca_path)
    print_top_diffs(per_class_means, top_k=args.top_k)

    print(f"Saved: {boxplot_path}")
    print(f"Saved: {pca_path}")


if __name__ == "__main__":
    main()
