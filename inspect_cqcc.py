from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def load_dataset(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    paths = data["paths"]
    metadata = json.loads(str(data["metadata"]))
    return X, y, paths, metadata


def invert_label_map(label_map: Dict[str, int]) -> Dict[int, str]:
    return {int(v): k for k, v in label_map.items()}


def build_file_level_stats(X: np.ndarray) -> np.ndarray:
    # Fixed-length representation per file: [mean(CQCC over frames), std(CQCC over frames)]
    vectors: List[np.ndarray] = []
    for feat in X:
        feat = np.asarray(feat, dtype=np.float32)
        mu = feat.mean(axis=0)
        sigma = feat.std(axis=0)
        vectors.append(np.hstack([mu, sigma]))
    return np.vstack(vectors).astype(np.float32)


def coefficient_class_means(X: np.ndarray, y: np.ndarray, class_ids: List[int]) -> Dict[int, np.ndarray]:
    means = {}
    for cls in class_ids:
        idx = np.where(y == cls)[0]
        per_file_means = np.stack([np.asarray(X[i]).mean(axis=0) for i in idx], axis=0)
        means[cls] = per_file_means
    return means


def save_boxplot(per_class_means: Dict[int, np.ndarray], id_to_name: Dict[int, str], output_path: Path) -> None:
    class_ids = sorted(per_class_means.keys())
    n_coeffs = per_class_means[class_ids[0]].shape[1]

    # Choose top coefficients by absolute difference between first two classes (if available).
    if len(class_ids) >= 2:
        diff = np.abs(per_class_means[class_ids[0]].mean(axis=0) - per_class_means[class_ids[1]].mean(axis=0))
        top_idx = np.argsort(diff)[::-1][: min(12, n_coeffs)]
    else:
        top_idx = np.arange(min(12, n_coeffs))

    fig, axes = plt.subplots(3, 4, figsize=(18, 11), constrained_layout=True)
    axes = axes.ravel()

    for ax_i, coeff_idx in enumerate(top_idx):
        ax = axes[ax_i]
        box_data = [per_class_means[c][:, coeff_idx] for c in class_ids]
        labels = [id_to_name.get(c, str(c)) for c in class_ids]
        ax.boxplot(box_data, labels=labels, showfliers=False)
        ax.set_title(f"CQCC[{coeff_idx}]", fontsize=10)
        ax.tick_params(axis="x", labelrotation=20)

    for ax in axes[len(top_idx) :]:
        ax.axis("off")

    fig.suptitle("Top CQCC Coefficients: Per-file Means by Class", fontsize=14)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def save_pca_scatter(X_fixed: np.ndarray, y: np.ndarray, id_to_name: Dict[int, str], output_path: Path) -> None:
    pca = PCA(n_components=2, random_state=42)
    z = pca.fit_transform(X_fixed)

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    class_ids = sorted(np.unique(y).tolist())
    cmap = plt.get_cmap("tab10")

    for k, cls in enumerate(class_ids):
        idx = np.where(y == cls)[0]
        ax.scatter(
            z[idx, 0],
            z[idx, 1],
            s=14,
            alpha=0.65,
            label=f"{id_to_name.get(int(cls), cls)} (n={len(idx)})",
            color=cmap(k % 10),
            edgecolors="none",
        )

    var1 = pca.explained_variance_ratio_[0] * 100.0
    var2 = pca.explained_variance_ratio_[1] * 100.0
    ax.set_xlabel(f"PC1 ({var1:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var2:.1f}% var)")
    ax.set_title("PCA of File-level CQCC Features (mean+std)")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)

    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def print_top_diffs(per_class_means: Dict[int, np.ndarray], id_to_name: Dict[int, str], top_k: int = 10) -> None:
    class_ids = sorted(per_class_means.keys())
    if len(class_ids) < 2:
        print("Need at least 2 classes for coefficient-difference ranking.")
        return

    c0, c1 = class_ids[0], class_ids[1]
    d = per_class_means[c0].mean(axis=0) - per_class_means[c1].mean(axis=0)
    order = np.argsort(np.abs(d))[::-1][:top_k]

    print(f"Top {top_k} coefficient mean differences: {id_to_name.get(c0, c0)} - {id_to_name.get(c1, c1)}")
    for i in order:
        print(f"  CQCC[{i:02d}] diff={d[i]:+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect CQCC NPZ dataset with visualizations.")
    parser.add_argument("--npz", type=Path, default=Path("cqcc_dataset.npz"), help="Path to CQCC dataset NPZ.")
    parser.add_argument("--out-dir", type=Path, default=Path("cqcc_inspection"), help="Directory for output plots.")
    args = parser.parse_args()

    X, y, _paths, metadata = load_dataset(args.npz)
    label_map = metadata.get("label_map", {"class0": 0, "class1": 1})
    id_to_name = invert_label_map(label_map)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    classes = sorted(np.unique(y).tolist())
    per_class_means = coefficient_class_means(X, y, classes)
    X_fixed = build_file_level_stats(X)

    boxplot_path = args.out_dir / "cqcc_coeff_boxplots.png"
    pca_path = args.out_dir / "cqcc_pca_scatter.png"

    save_boxplot(per_class_means, id_to_name, boxplot_path)
    save_pca_scatter(X_fixed, y, id_to_name, pca_path)
    print_top_diffs(per_class_means, id_to_name, top_k=10)

    print(f"Saved: {boxplot_path}")
    print(f"Saved: {pca_path}")


if __name__ == "__main__":
    main()