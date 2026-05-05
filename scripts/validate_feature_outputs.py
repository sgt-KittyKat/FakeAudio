from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_FILES = {
    "train_features.npz",
    "val_features.npz",
    "test_features.npz",
    "train_metadata.csv",
    "val_metadata.csv",
    "test_metadata.csv",
    "manifest.csv",
    "speaker_splits.csv",
    "split_summary.csv",
    "config.json",
    "feature_stats.json",
}
EXPECTED_METADATA_COLUMNS = [
    "row_index",
    "file",
    "speaker",
    "label",
    "split",
    "chunk_index",
    "start_sample",
    "feature_type",
    "num_audio_samples",
    "num_feature_frames",
]
COMPARISON_COLUMNS = [
    "file",
    "speaker",
    "label",
    "split",
    "chunk_index",
    "start_sample",
    "num_audio_samples",
    "num_feature_frames",
]
COMMON_CONFIG_KEYS = [
    "sample_rate",
    "chunk_seconds",
    "hop_seconds",
    "pre_emphasis",
    "include_deltas",
    "split_seed",
    "train_ratio",
    "val_ratio",
    "test_ratio",
    "target_frames",
]


def load_method_summary(base: Path) -> dict[str, object]:
    present_files = {path.name for path in base.iterdir() if path.is_file()}
    missing = sorted(EXPECTED_FILES - present_files)
    extra = sorted(present_files - EXPECTED_FILES)
    if missing:
        raise ValueError(f"{base}: missing expected artifacts: {missing}")
    if extra:
        raise ValueError(f"{base}: unexpected extra artifacts: {extra}")

    config = json.loads((base / "config.json").read_text())
    stats = json.loads((base / "feature_stats.json").read_text())
    if sorted(stats.keys()) != ["mean", "std"]:
        raise ValueError(f"{base}: feature_stats.json must contain only mean/std")

    summary: dict[str, object] = {
        "config": config,
        "speaker_splits": pd.read_csv(base / "speaker_splits.csv").sort_values(["split", "speaker"]).reset_index(drop=True),
        "split_summary": pd.read_csv(base / "split_summary.csv").sort_values(["split", "label"]).reset_index(drop=True),
        "manifest": pd.read_csv(base / "manifest.csv").sort_values(["split", "row_index"]).reset_index(drop=True),
        "split_shapes": {},
        "split_metadata": {},
    }

    manifest = summary["manifest"]
    if list(manifest.columns) != EXPECTED_METADATA_COLUMNS:
        raise ValueError(f"{base}: manifest columns mismatch: {list(manifest.columns)}")

    for split in ("train", "val", "test"):
        features = np.load(base / f"{split}_features.npz")["X"]
        metadata = pd.read_csv(base / f"{split}_metadata.csv")
        if list(metadata.columns) != EXPECTED_METADATA_COLUMNS:
            raise ValueError(f"{base}: {split} metadata columns mismatch: {list(metadata.columns)}")
        if features.ndim != 4:
            raise ValueError(f"{base}: {split} features must be rank-4, found {features.ndim}")
        if features.shape[0] != len(metadata):
            raise ValueError(f"{base}: {split} feature rows {features.shape[0]} != metadata rows {len(metadata)}")
        summary["split_shapes"][split] = tuple(features.shape)
        summary["split_metadata"][split] = metadata.sort_values(["row_index"]).reset_index(drop=True)

    return summary


def validate_common_configs(methods: dict[str, dict[str, object]]) -> None:
    reference_name = next(iter(methods))
    reference = methods[reference_name]["config"]
    for name, data in methods.items():
        config = data["config"]
        for key in COMMON_CONFIG_KEYS:
            if config.get(key) != reference.get(key):
                raise ValueError(
                    f"Config mismatch for {key}: {reference_name}={reference.get(key)} vs {name}={config.get(key)}"
                )


def validate_alignment(methods: dict[str, dict[str, object]]) -> None:
    reference_name = next(iter(methods))
    reference = methods[reference_name]
    for name, data in methods.items():
        if name == reference_name:
            continue
        if not reference["speaker_splits"].equals(data["speaker_splits"]):
            raise ValueError(f"speaker_splits.csv mismatch between {reference_name} and {name}")
        if not reference["split_summary"].equals(data["split_summary"]):
            raise ValueError(f"split_summary.csv mismatch between {reference_name} and {name}")

        ref_manifest = reference["manifest"][COMPARISON_COLUMNS]
        cmp_manifest = data["manifest"][COMPARISON_COLUMNS]
        if not ref_manifest.equals(cmp_manifest):
            raise ValueError(f"manifest.csv mismatch between {reference_name} and {name}")

        for split in ("train", "val", "test"):
            if reference["split_shapes"][split] != data["split_shapes"][split]:
                raise ValueError(
                    f"{split} tensor shape mismatch: {reference_name}={reference['split_shapes'][split]} vs "
                    f"{name}={data['split_shapes'][split]}"
                )
            ref_meta = reference["split_metadata"][split][COMPARISON_COLUMNS]
            cmp_meta = data["split_metadata"][split][COMPARISON_COLUMNS]
            if not ref_meta.equals(cmp_meta):
                raise ValueError(f"{split} metadata mismatch between {reference_name} and {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate processed feature outputs across methods.")
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--methods", nargs="+", default=["mfcc", "lfcc", "cqcc"])
    args = parser.parse_args()

    methods = {
        method: load_method_summary(args.processed_root / method)
        for method in args.methods
    }
    validate_common_configs(methods)
    validate_alignment(methods)

    print("Validation passed for methods:", ", ".join(args.methods))
    for method in args.methods:
        split_shapes = methods[method]["split_shapes"]
        print(f"{method}: train={split_shapes['train']} val={split_shapes['val']} test={split_shapes['test']}")


if __name__ == "__main__":
    main()
