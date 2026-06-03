# Deepfake Audio Detection Using Cepstral Features and LCNN

## Overview

This project builds a reproducible audio deepfake detection pipeline using the In-the-Wild audio deepfake dataset. The system focuses on detecting real and fake speech using handcrafted cepstral features and a Light CNN classifier.

The project includes:

* Dataset organization using metadata labels
* Speaker-disjoint train, validation, and test splitting
* MFCC, LFCC, and CQCC feature extraction
* Fixed-length 4-second audio chunking
* LCNN model training
* Evaluation using ROC-AUC, EER, and F1-score
* GradCAM explainability for model interpretation

## Dataset

Dataset: In-the-Wild Audio Deepfake Dataset
Source: https://www.kaggle.com/datasets/abdallamohamed312/in-the-wild-dataset

The dataset contains real and spoofed speech samples from public figures. The metadata file `meta.csv` provides the filename, speaker, and label information used by the pipeline.

Expected project data layout:

```text
data/
├── meta.csv
└── release_in_the_wild/
    ├── real/
    └── fake/
```

Some Kaggle downloads extract all `.wav` files directly into `data/release_in_the_wild/` without separating them into `real/` and `fake/` folders. To fix this, the project includes a helper script that organizes the audio files using `meta.csv`.

## Dataset Organization Helper

If the extracted audio files are stored in a flat folder like this:

```text
data/release_in_the_wild/
├── 0.wav
├── 1.wav
├── 2.wav
└── ...
```

run the helper script before preprocessing:

```bash
python scripts/organize_audio_by_label.py --mode dry-run
```

The dry run checks what will be moved without changing the files.

If the paths look correct, run:

```bash
python scripts/organize_audio_by_label.py --mode move
```

This creates the expected structure:

```text
data/release_in_the_wild/
├── real/
└── fake/
```

The helper reads `data/meta.csv`, maps `bona-fide` samples to `real`, maps `spoof` samples to `fake`, and moves each audio file into the correct folder.

## Project Structure

```text
FakeAudio/
├── data/
│   ├── meta.csv
│   ├── release_in_the_wild/
│   └── processed/
├── models/
├── scripts/
│   ├── organize_audio_by_label.py
│   ├── run_all_preprocessing.py
│   ├── extract_mfcc.py
│   ├── extract_lfcc.py
│   ├── extract_cqcc.py
│   └── validate_feature_outputs.py
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── explain.py
├── xai_outputs/
├── README.md
└── requirements.txt
```

## Environment Setup

Create and activate a virtual environment.

Windows CMD:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install torch torchaudio librosa scipy pandas numpy tqdm scikit-learn matplotlib
```

## Preprocessing Pipeline

The preprocessing pipeline extracts fixed-length 4-second chunks from the audio files and converts them into MFCC, LFCC, and CQCC feature tensors.

Each feature type produces tensors with this shape:

```text
[N, C, F, T]
```

where:

* `N` = number of chunks
* `C` = number of channels
* `F` = number of cepstral coefficients
* `T` = number of time frames

The default output shape per chunk is:

```text
[3, 20, 400]
```

The 3 channels are:

* Static coefficients
* Delta coefficients
* Delta-delta coefficients

## Speaker-Disjoint Splitting

The pipeline uses a speaker-disjoint split to prevent speaker leakage, this means the same speaker cannot appear in more than one split. For example, if a speaker is assigned to the training set, all real and fake samples from that speaker remain in the training set only.


Default split configuration:

```text
Train: 70%
Validation: 15%
Test: 15%
Split seed: 13
```

## Run a Smoke Test

Before running the full preprocessing pipeline, run a small smoke test:

```bash
python scripts/run_all_preprocessing.py --limit-files 12 --processed-root data/processed_smoke
```

This checks that MFCC, LFCC, and CQCC extraction work correctly on a small subset.

Validate the smoke test outputs:

```bash
python scripts/validate_feature_outputs.py --processed-root data/processed_smoke --methods mfcc lfcc cqcc
```

## Run Full Preprocessing

After the smoke test passes, run the full preprocessing pipeline:

```bash
python scripts/run_all_preprocessing.py
```

This creates:

```text
data/processed/mfcc/
data/processed/lfcc/
data/processed/cqcc/
```

Each feature directory contains:

```text
train_features.npz
val_features.npz
test_features.npz
train_metadata.csv
val_metadata.csv
test_metadata.csv
manifest.csv
speaker_splits.csv
split_summary.csv
config.json
feature_stats.json
```

## Training

Train one LCNN model per feature type.

MFCC:

```bash
python src/train.py --feature-type mfcc --epochs 30
```

LFCC:

```bash
python src/train.py --feature-type lfcc --epochs 30
```

CQCC:

```bash
python src/train.py --feature-type cqcc --epochs 30
```

The best checkpoints are saved in:

```text
models/
```

Example:

```text
models/lcnn_mfcc_best.pt
models/lcnn_lfcc_best.pt
models/lcnn_cqcc_best.pt
```

## Evaluation

Evaluate each trained model on the test split.

MFCC:

```bash
python src/evaluate.py --feature-type mfcc --split test
```

LFCC:

```bash
python src/evaluate.py --feature-type lfcc --split test
```

CQCC:

```bash
python src/evaluate.py --feature-type cqcc --split test
```

The evaluation metrics are:

* ROC-AUC
* EER
* F1-score

## Explainability

GradCAM is used to visualize which regions of the feature map the LCNN focuses on.

Generate GradCAM output for MFCC:

```bash
python src/explain.py --feature-type mfcc --split test
```

Outputs are saved in:

```text
xai_outputs/
```

## Final Test Results

| Feature Type | Best Checkpoint Epoch | ROC-AUC |    EER | F1-score |
| ------------ | --------------------: | ------: | -----: | -------: |
| MFCC         |                    29 |  0.9990 | 0.0084 |   0.9917 |
| LFCC         |                    17 |  0.9970 | 0.0084 |   0.9917 |
| CQCC         |                     5 |  0.9969 | 0.0070 |   0.9931 |

## Result Interpretation

MFCC achieved the highest ROC-AUC, showing the strongest ranking ability between real and fake samples.

CQCC achieved the lowest EER and highest F1-score, making it the strongest threshold-based detector.

LFCC also performed strongly but did not clearly outperform MFCC or CQCC.

Based on threshold-based detection performance, CQCC is the recommended final model, MFCC is retained as a strong alternative because it achieved the highest ROC-AUC.


## AI Usage Declaration

Generative AI tools were used as support during this project, the team used AI assistance for:

* Designing and refining the `organize_audio_by_label.py` helper script used to organize flat Kaggle audio downloads into `real/` and `fake/` folders based on `meta.csv`.
* Researching and summarizing relevant papers on audio deepfake detection methodologies.
* Structuring and improving the project documentation.
* Preparing and refining the project presentation.

All code, outputs, preprocessing results, model training, evaluation results, and final project decisions were reviewed, tested, and validated by the team. AI-generated suggestions were not used blindly, they were checked against the project code, experiment logs, dataset structure, and final evaluation results.


## References

Abdul, Z. K., & Al-Talabani, A. K. (2022). Mel Frequency Cepstral Coefficient and its applications: A review. IEEE Access, 10, 122136–122158.

Dewi, S. P., Prasasti, A. L., & Irawan, B. (2019). The study of baby crying analysis using MFCC and LFCC in different classification methods. 2019 IEEE International Conference on Signals and Systems.

Müller, N. M., Czempin, P., Dieckmann, F., Froghyar, A., & Böttinger, K. (2022). Does audio deepfake detection generalize?

Wang, Q., Huang, H., Pang, G., Erfani, S., & Leckie, C. (2025). AUDETER: A large-scale dataset for deepfake audio detection in open worlds.
