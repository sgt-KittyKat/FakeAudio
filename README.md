# Deepfake Audio Detection (In-the-Wild Dataset)

## Overview

This project builds a robust deepfake audio detection model using the In-the-Wild dataset.
The focus is on generalization to unseen speakers and real-world noise conditions.

## Dataset

Dataset: https://www.kaggle.com/datasets/abdallamohamed312/in-the-wild-dataset

* ~38 hours of audio
* Real and fake speech from public figures
* High variability (noise, compression, etc.)

## Project Structure
```text
deepfake-audio-detection/
│── data/
│   ├── raw/
│   ├── processed/
│
│── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │
│   ├── models/
│   │   ├── model.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   ├── evaluate.py
│
│   ├── utils/
│
│── configs/
│── experiments/
│── notebooks/
│── requirements.txt
│── README.md
│── .gitignore
```

## Setup

```bash
git clone <repo_url>
cd deepfake-audio-detection

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

## Data Preparation

Download dataset manually from Kaggle and place in:

data/raw/

Run preprocessing:

```bash
python src/data/preprocessing.py
```

## Training

```bash
python src/training/train.py --config configs/base.yaml
```

## Evaluation

```bash
python src/training/evaluate.py
```

## Reproducibility

* Fixed random seeds
* Config-driven experiments
* Speaker-disjoint train/test split

## Metrics

* ROC-AUC
* F1-score
* EER
* Explainability XAI
