# Audio Feature Pipeline

This project now includes preprocessing pipelines for the `in_the_wild` audio deepfake dataset that extract fixed-length `4s` chunks for `MFCC`, `LFCC`, and `CQCC`, and save each split as a single `.npz` array store.

## What the pipeline does

- Reads `data/meta.csv` so labels and speakers come from metadata, not only folder names.
- Creates speaker-disjoint `train`, `val`, and `test` splits.
- Loads raw `.wav` files from `data/release_in_the_wild/{fake,real}`.
- Applies optional pre-emphasis with coefficient `0.97`.
- Chunks audio into `4s` windows.
- Extracts `MFCC`, `LFCC`, or `CQCC` features with a fixed `[N, C, F, T]` output contract.
- Adds `delta` and `delta-delta` channels by default.
- Forces every chunk tensor to a fixed `[N, C, F, T]` layout with `T=400`.
- Saves one `.npz` feature store per split instead of one file per chunk.
- Writes separate metadata CSVs plus `manifest.csv`, `speaker_splits.csv`, `split_summary.csv`, `config.json`, and `feature_stats.json`.

## Output layout

- `data/processed/mfcc`
- `data/processed/lfcc`
- `data/processed/cqcc`

Each directory is independent, so rerunning one feature extraction does not touch the others.

Each feature directory contains:

- `train_features.npz`
- `val_features.npz`
- `test_features.npz`
- `train_metadata.csv`
- `val_metadata.csv`
- `test_metadata.csv`
- `manifest.csv`
- `speaker_splits.csv`
- `split_summary.csv`
- `config.json`
- `feature_stats.json`

## Commands

Run all three preprocessing methods in one pass:

```bash
venv/bin/python scripts/run_all_preprocessing.py
```

MFCC extraction:

```bash
venv/bin/python scripts/extract_mfcc.py
```

LFCC extraction:

```bash
venv/bin/python scripts/extract_lfcc.py
```

CQCC extraction:

```bash
venv/bin/python scripts/extract_cqcc.py
```

Quick smoke test on a small subset:

```bash
venv/bin/python scripts/run_all_preprocessing.py --limit-files 12 --processed-root /private/tmp/feature_pipeline_smoke
```

Validate that all feature methods produced the same artifact structure and aligned split metadata:

```bash
venv/bin/python scripts/validate_feature_outputs.py --processed-root data/processed --methods mfcc lfcc cqcc
```

Inspect any processed feature store:

```bash
venv/bin/python inspect_preprocessed_features.py --feature-type mfcc
```

## Tensor format

Each split `.npz` file contains:

- `X`: array shaped `[N, C, F, T]`

where:

- `N` is the number of chunks in that split
- `C` is `3` when deltas are enabled
- `F` is the number of coefficients, default `20`
- `T` is fixed to `400`

With deltas enabled, `channels=3`:

- channel `0`: static coefficients
- channel `1`: delta
- channel `2`: delta-delta

The metadata CSV stores one row per chunk in the same row order as `X`, including:

- `file`
- `speaker`
- `label`
- `split`
- `chunk_index`
- `start_sample`
- `feature_type`

## Loading example

```python
import numpy as np
import pandas as pd

train_x = np.load("data/processed/mfcc/train_features.npz")["X"]
train_meta = pd.read_csv("data/processed/mfcc/train_metadata.csv")
```

## Notes

- The current defaults use non-overlapping `4s` windows. You can change that with `--hop-seconds`.
- The split assignment is speaker-disjoint and deterministic from `--split-seed`.
- `feature_stats.json` contains train-split coefficient statistics for later normalization in the modeling pipeline.
- `T=400` is enforced after extraction so every split array is stackable without shape drift.
- `MFCC`, `LFCC`, and `CQCC` now share the same split logic, metadata schema, and artifact layout.

# Deepfake Audio Detection (In-the-Wild Dataset)

## Overview

This project builds a robust deepfake audio detection model using the In-the-Wild dataset.
The focus is on generalization to unseen speakers and real-world noise conditions.

## Dataset

Dataset: https://www.kaggle.com/datasets/abdallamohamed312/in-the-wild-dataset

* ~38 hours of audio
* Real and fake speech from public figures
* High variability (noise, compression, etc.)


## Metrics

* ROC-AUC
* F1-score
* EER
* Explainability XAI
