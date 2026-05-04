from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.functional as audio_functional
from scipy.io import wavfile
from tqdm import tqdm


LABEL_MAP = {"bona-fide": 0, "real": 0, "spoof": 1, "fake": 1}
METADATA_COLUMNS = [
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


@dataclass
class ExtractionConfig:
    feature_type: str
    sample_rate: int = 16000
    chunk_seconds: float = 4.0
    hop_seconds: float = 4.0
    frame_length_ms: float = 25.0
    frame_hop_ms: float = 10.0
    n_fft: int = 400
    n_features: int = 20
    n_filters: int = 40
    pre_emphasis: float = 0.97
    include_deltas: bool = True
    pad_mode: str = "constant"
    split_seed: int = 42
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    target_frames: int = 400

    @property
    def win_length(self) -> int:
        return int(self.sample_rate * self.frame_length_ms / 1000.0)

    @property
    def hop_length(self) -> int:
        return int(self.sample_rate * self.frame_hop_ms / 1000.0)

    @property
    def chunk_samples(self) -> int:
        return int(self.sample_rate * self.chunk_seconds)

    @property
    def chunk_hop_samples(self) -> int:
        return int(self.sample_rate * self.hop_seconds)


def build_parser(feature_type: str, default_output_dir: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Extract {feature_type.upper()} features from the in-the-wild deepfake dataset."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/release_in_the_wild"))
    parser.add_argument("--metadata-path", type=Path, default=Path("data/meta.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path(default_output_dir))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-seconds", type=float, default=4.0)
    parser.add_argument("--hop-seconds", type=float, default=4.0)
    parser.add_argument("--frame-length-ms", type=float, default=25.0)
    parser.add_argument("--frame-hop-ms", type=float, default=10.0)
    parser.add_argument("--n-fft", type=int, default=400)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--n-filters", type=int, default=40)
    parser.add_argument("--pre-emphasis", type=float, default=0.97)
    parser.add_argument("--split-seed", type=int, default=13)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--target-frames", type=int, default=400)
    parser.add_argument("--disable-deltas", action="store_true")
    parser.add_argument("--limit-files", type=int, default=None)
    return parser


def args_to_config(args: argparse.Namespace, feature_type: str) -> ExtractionConfig:
    return ExtractionConfig(
        feature_type=feature_type,
        sample_rate=args.sample_rate,
        chunk_seconds=args.chunk_seconds,
        hop_seconds=args.hop_seconds,
        frame_length_ms=args.frame_length_ms,
        frame_hop_ms=args.frame_hop_ms,
        n_fft=args.n_fft,
        n_features=args.n_features,
        n_filters=args.n_filters,
        pre_emphasis=args.pre_emphasis,
        include_deltas=not args.disable_deltas,
        split_seed=args.split_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        target_frames=args.target_frames,
    )


def load_metadata(metadata_path: Path, data_root: Path, limit_files: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(metadata_path)
    df["label_name"] = df["label"].map(LABEL_MAP)
    if df["label_name"].isna().any():
        missing = df[df["label_name"].isna()]["label"].drop_duplicates().tolist()
        raise ValueError(f"Unsupported labels found in metadata: {missing}")

    df["label"] = df["label_name"].astype(int)
    df["audio_path"] = df.apply(resolve_audio_path, axis=1, data_root=data_root)
    missing_paths = df[~df["audio_path"].map(Path.exists)]
    if not missing_paths.empty:
        preview = missing_paths[["file", "audio_path"]].head(10).to_dict("records")
        raise FileNotFoundError(f"Missing audio files detected: {preview}")

    df = df.drop(columns=["label_name"])
    if limit_files is not None:
        df = df.head(limit_files).copy()
    return df


def resolve_audio_path(row: pd.Series, data_root: Path) -> Path:
    if "speaker" in row:
        label_dir = "fake" if int(row["label"]) == 1 else "real"
    else:
        label_dir = row["label"]
    return data_root / label_dir / row["file"]


def assign_speaker_splits(df: pd.DataFrame, seed: int, ratios: tuple[float, float, float]) -> pd.DataFrame:
    train_ratio, val_ratio, test_ratio = ratios
    split_names = ("train", "val", "test")
    tracked_metrics = ("num_files", "num_real", "num_fake", "num_speakers")
    speakers = (
        df.groupby("speaker", as_index=False)
        .agg(
            num_files=("file", "count"),
            num_fake=("label", "sum"),
        )
        .reset_index(drop=True)
    )
    speakers["num_real"] = speakers["num_files"] - speakers["num_fake"]
    speakers["random_tiebreak"] = (
        pd.Series(range(len(speakers))).sample(frac=1.0, random_state=seed).to_numpy()
    )
    speakers = speakers.sort_values(
        ["num_files", "num_fake", "random_tiebreak"],
        ascending=[False, False, True],
    ).drop(columns=["random_tiebreak"]).reset_index(drop=True)

    total_real = int((df["label"] == 0).sum())
    total_fake = int((df["label"] == 1).sum())
    split_weights = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    targets = {
        split: {
            "num_files": weight * len(df),
            "num_real": weight * total_real,
            "num_fake": weight * total_fake,
            "num_speakers": weight * len(speakers),
        }
        for split, weight in split_weights.items()
    }
    counts = {
        split: {metric: 0 for metric in tracked_metrics} for split in split_names
    }
    speaker_to_split: dict[str, str] = {}
    metric_weights = {
        "num_files": 1.0,
        "num_real": 2.0,
        "num_fake": 2.0,
        "num_speakers": 0.2,
    }

    def objective(current_counts: dict[str, dict[str, int]]) -> float:
        cost = 0.0
        for split_name in split_names:
            for metric, weight in metric_weights.items():
                target = max(targets[split_name][metric], 1.0)
                current_ratio = current_counts[split_name][metric] / target
                cost += weight * (current_ratio - 1.0) ** 2
        return cost

    for row in speakers.itertuples(index=False):
        best_split = None
        best_cost = None
        for split_name in split_names:
            for metric in ("num_files", "num_real", "num_fake"):
                counts[split_name][metric] += getattr(row, metric)
            counts[split_name]["num_speakers"] += 1
            candidate_cost = objective(counts)
            for metric in ("num_files", "num_real", "num_fake"):
                counts[split_name][metric] -= getattr(row, metric)
            counts[split_name]["num_speakers"] -= 1
            if best_cost is None or candidate_cost < best_cost:
                best_cost = candidate_cost
                best_split = split_name

        split = best_split
        if split is None:
            raise RuntimeError("Failed to assign speaker to a split during greedy initialization.")
        speaker_to_split[row.speaker] = split
        for metric in ("num_files", "num_real", "num_fake"):
            counts[split][metric] += getattr(row, metric)
        counts[split]["num_speakers"] += 1

    speaker_rows = {row.speaker: row for row in speakers.itertuples(index=False)}
    current_cost = objective(counts)

    for _ in range(20):
        changed = False
        speaker_names = list(speaker_rows)

        for speaker_name in speaker_names:
            row = speaker_rows[speaker_name]
            source_split = speaker_to_split[speaker_name]
            if counts[source_split]["num_speakers"] <= 1:
                continue

            for target_split in split_names:
                if target_split == source_split:
                    continue

                for metric in ("num_files", "num_real", "num_fake"):
                    counts[source_split][metric] -= getattr(row, metric)
                    counts[target_split][metric] += getattr(row, metric)
                counts[source_split]["num_speakers"] -= 1
                counts[target_split]["num_speakers"] += 1

                candidate_cost = objective(counts)
                if candidate_cost + 1e-12 < current_cost:
                    speaker_to_split[speaker_name] = target_split
                    current_cost = candidate_cost
                    changed = True
                    source_split = target_split
                else:
                    for metric in ("num_files", "num_real", "num_fake"):
                        counts[source_split][metric] += getattr(row, metric)
                        counts[target_split][metric] -= getattr(row, metric)
                    counts[source_split]["num_speakers"] += 1
                    counts[target_split]["num_speakers"] -= 1

        swap_improved = True
        while swap_improved:
            swap_improved = False
            speaker_names = list(speaker_rows)
            for idx_a in range(len(speaker_names)):
                speaker_a = speaker_names[idx_a]
                row_a = speaker_rows[speaker_a]
                split_a = speaker_to_split[speaker_a]

                for idx_b in range(idx_a + 1, len(speaker_names)):
                    speaker_b = speaker_names[idx_b]
                    row_b = speaker_rows[speaker_b]
                    split_b = speaker_to_split[speaker_b]
                    if split_a == split_b:
                        continue

                    for metric in ("num_files", "num_real", "num_fake"):
                        counts[split_a][metric] += getattr(row_b, metric) - getattr(row_a, metric)
                        counts[split_b][metric] += getattr(row_a, metric) - getattr(row_b, metric)

                    candidate_cost = objective(counts)
                    if candidate_cost + 1e-12 < current_cost:
                        speaker_to_split[speaker_a] = split_b
                        speaker_to_split[speaker_b] = split_a
                        current_cost = candidate_cost
                        changed = True
                        swap_improved = True
                        break

                    for metric in ("num_files", "num_real", "num_fake"):
                        counts[split_a][metric] -= getattr(row_b, metric) - getattr(row_a, metric)
                        counts[split_b][metric] -= getattr(row_a, metric) - getattr(row_b, metric)
                if swap_improved:
                    break

        if not changed:
            break

    split_df = speakers[["speaker"]].copy()
    split_df["split"] = split_df["speaker"].map(speaker_to_split)

    out = df.merge(split_df, on="speaker", how="left", validate="many_to_one")
    if out["split"].isna().any():
        raise RuntimeError("Some speakers were not assigned to a split.")
    return out


def load_waveform(audio_path: Path, expected_sample_rate: int) -> torch.Tensor:
    sample_rate, waveform = wavfile.read(audio_path)
    if sample_rate != expected_sample_rate:
        raise ValueError(f"Expected {expected_sample_rate} Hz, found {sample_rate} Hz in {audio_path}")

    waveform = torch.as_tensor(waveform, dtype=torch.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=1)
    waveform = waveform.unsqueeze(0)
    return waveform


def apply_pre_emphasis(waveform: torch.Tensor, coefficient: float) -> torch.Tensor:
    if coefficient <= 0:
        return waveform
    emphasized = waveform.clone()
    emphasized[..., 1:] = waveform[..., 1:] - coefficient * waveform[..., :-1]
    emphasized[..., 0] = waveform[..., 0]
    return emphasized


def compute_chunk_starts(num_samples: int, chunk_samples: int, hop_samples: int) -> list[int]:
    if num_samples <= chunk_samples:
        return [0]

    starts = list(range(0, num_samples - chunk_samples + 1, hop_samples))
    final_start = num_samples - chunk_samples
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def chunk_waveform(
    waveform: torch.Tensor,
    chunk_samples: int,
    hop_samples: int,
    pad_mode: str,
) -> Iterable[tuple[int, torch.Tensor]]:
    num_samples = waveform.shape[-1]
    for start in compute_chunk_starts(num_samples, chunk_samples, hop_samples):
        chunk = waveform[..., start : start + chunk_samples]
        if chunk.shape[-1] < chunk_samples:
            pad_amount = chunk_samples - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad_amount), mode=pad_mode)
        yield start, chunk


def build_transform(config: ExtractionConfig) -> torch.nn.Module:
    common_kwargs = {
        "sample_rate": config.sample_rate,
        "speckwargs": {
            "n_fft": config.n_fft,
            "win_length": config.win_length,
            "hop_length": config.hop_length,
            "center": True,
        },
    }
    if config.feature_type == "mfcc":
        return torchaudio.transforms.MFCC(
            sample_rate=config.sample_rate,
            n_mfcc=config.n_features,
            melkwargs={
                "n_fft": config.n_fft,
                "win_length": config.win_length,
                "hop_length": config.hop_length,
                "n_mels": config.n_filters,
                "center": True,
            },
            log_mels=True,
        )
    if config.feature_type == "lfcc":
        return torchaudio.transforms.LFCC(
            n_lfcc=config.n_features,
            n_filter=config.n_filters,
            **common_kwargs,
        )
    raise ValueError(f"Unsupported feature type: {config.feature_type}")


def extract_feature_tensor(
    waveform: torch.Tensor,
    transform: torch.nn.Module,
    include_deltas: bool,
    target_frames: int,
) -> torch.Tensor:
    base = transform(waveform)
    streams = [base]
    if include_deltas:
        delta = audio_functional.compute_deltas(base)
        delta2 = audio_functional.compute_deltas(delta)
        streams.extend([delta, delta2])
    stacked = torch.cat(streams, dim=0).contiguous()
    return ensure_target_frames(stacked, target_frames)


def ensure_target_frames(features: torch.Tensor, target_frames: int) -> torch.Tensor:
    current_frames = features.shape[-1]
    if current_frames == target_frames:
        return features
    if current_frames > target_frames:
        return features[..., :target_frames].contiguous()
    pad_amount = target_frames - current_frames
    return torch.nn.functional.pad(features, (0, pad_amount), mode="constant", value=0.0).contiguous()


def summarize_stats(sum_tensor: torch.Tensor, sq_sum_tensor: torch.Tensor, count: int) -> dict[str, list[float]]:
    mean = sum_tensor / count
    var = (sq_sum_tensor / count) - mean.pow(2)
    std = torch.sqrt(torch.clamp(var, min=1e-12))
    return {"mean": mean.tolist(), "std": std.tolist()}


def print_header(config: ExtractionConfig, args: argparse.Namespace) -> None:
    print("=" * 72)
    print(f"Starting {config.feature_type.upper()} extraction")
    print(f"data_root      : {args.data_root}")
    print(f"metadata_path  : {args.metadata_path}")
    print(f"output_dir     : {args.output_dir}")
    print(f"sample_rate    : {config.sample_rate}")
    print(f"chunk_seconds  : {config.chunk_seconds}")
    print(f"hop_seconds    : {config.hop_seconds}")
    print(f"frame_ms       : {config.frame_length_ms}")
    print(f"frame_hop_ms   : {config.frame_hop_ms}")
    print(f"n_fft          : {config.n_fft}")
    print(f"n_features     : {config.n_features}")
    print(f"n_filters      : {config.n_filters}")
    print(f"pre_emphasis   : {config.pre_emphasis}")
    print(f"include_deltas : {config.include_deltas}")
    print(f"target_frames  : {config.target_frames}")
    print(f"limit_files    : {args.limit_files}")
    print("=" * 72)


def print_split_overview(df: pd.DataFrame) -> None:
    speaker_counts = df.groupby("split")["speaker"].nunique().rename("num_speakers")
    file_counts = df.groupby(["split", "label"]).size().rename("num_files").reset_index()
    label_mix = (
        df.groupby("split")
        .agg(
            total_files=("file", "count"),
            real_files=("label", lambda x: int((x == 0).sum())),
            fake_files=("label", lambda x: int((x == 1).sum())),
        )
        .reset_index()
    )
    label_mix["fake_share"] = (label_mix["fake_files"] / label_mix["total_files"]).round(4)
    print("Speaker-disjoint split summary")
    print(speaker_counts.to_string())
    print(file_counts.to_string(index=False))
    print(label_mix.to_string(index=False))
    print("-" * 72)


def save_split_npz(
    output_dir: Path,
    split: str,
    features: list[np.ndarray],
    metadata_rows: list[dict[str, object]],
) -> tuple[int, Path, Path]:
    feature_path = output_dir / f"{split}_features.npz"
    metadata_path = output_dir / f"{split}_metadata.csv"

    if features:
        x_array = np.stack(features, axis=0).astype(np.float32, copy=False)
    else:
        x_array = np.empty((0, 0, 0, 0), dtype=np.float32)
    np.savez(feature_path, X=x_array)

    metadata = pd.DataFrame(metadata_rows, columns=METADATA_COLUMNS)
    if not metadata.empty:
        metadata = metadata.sort_values(["row_index"]).reset_index(drop=True)
    metadata.to_csv(metadata_path, index=False)
    return x_array.shape[0], feature_path, metadata_path


def run_extraction(args: argparse.Namespace, feature_type: str) -> None:
    config = args_to_config(args, feature_type)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print_header(config, args)

    df = load_metadata(args.metadata_path, args.data_root, limit_files=args.limit_files)
    print(f"Loaded {len(df)} metadata rows")
    df = assign_speaker_splits(
        df,
        seed=config.split_seed,
        ratios=(config.train_ratio, config.val_ratio, config.test_ratio),
    )
    print_split_overview(df)

    transform = build_transform(config)
    train_sum: torch.Tensor | None = None
    train_sq_sum: torch.Tensor | None = None
    train_frame_count = 0
    chunk_counter = 0
    split_chunk_counts = {"train": 0, "val": 0, "test": 0}
    split_file_counts = df.groupby("split").size().to_dict()
    split_features: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    split_metadata: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}
    row_index_counters = {"train": 0, "val": 0, "test": 0}

    progress = tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc=f"{config.feature_type.upper()} files",
        unit="file",
    )
    for row in progress:
        waveform = load_waveform(Path(row.audio_path), config.sample_rate)
        waveform = apply_pre_emphasis(waveform, config.pre_emphasis)
        row_chunk_count = 0

        for chunk_idx, (start_sample, chunk) in enumerate(
            chunk_waveform(
                waveform,
                chunk_samples=config.chunk_samples,
                hop_samples=config.chunk_hop_samples,
                pad_mode=config.pad_mode,
            )
        ):
            features = extract_feature_tensor(
                waveform=chunk,
                transform=transform,
                include_deltas=config.include_deltas,
                target_frames=config.target_frames,
            )
            split_features[row.split].append(features.cpu().numpy().astype(np.float32, copy=False))
            split_metadata[row.split].append(
                {
                    "row_index": row_index_counters[row.split],
                    "file": row.file,
                    "speaker": row.speaker,
                    "label": int(row.label),
                    "split": row.split,
                    "chunk_index": chunk_idx,
                    "start_sample": start_sample,
                    "feature_type": config.feature_type,
                    "num_audio_samples": chunk.shape[-1],
                    "num_feature_frames": features.shape[-1],
                }
            )
            row_index_counters[row.split] += 1

            if row.split == "train":
                stream_sum = features.sum(dim=-1)
                stream_sq_sum = features.pow(2).sum(dim=-1)
                if train_sum is None or train_sq_sum is None:
                    train_sum = torch.zeros_like(stream_sum)
                    train_sq_sum = torch.zeros_like(stream_sq_sum)
                train_sum += stream_sum
                train_sq_sum += stream_sq_sum
                train_frame_count += features.shape[-1]
            row_chunk_count += 1
            chunk_counter += 1
            split_chunk_counts[row.split] += 1

        progress.set_postfix(
            split=row.split,
            chunks=row_chunk_count,
            total_chunks=chunk_counter,
            speaker=row.speaker[:18],
            refresh=False,
        )

    speaker_summary = df[["speaker", "split"]].drop_duplicates().sort_values(["split", "speaker"]).reset_index(drop=True)
    speaker_summary.to_csv(output_dir / "speaker_splits.csv", index=False)

    if train_sum is None or train_sq_sum is None or train_frame_count == 0:
        raise RuntimeError("No training chunks were created, cannot compute training statistics.")

    manifest_frames: list[pd.DataFrame] = []
    split_artifacts: dict[str, dict[str, object]] = {}
    for split_name in ("train", "val", "test"):
        saved_count, feature_path, metadata_path = save_split_npz(
            output_dir=output_dir,
            split=split_name,
            features=split_features[split_name],
            metadata_rows=split_metadata[split_name],
        )
        split_artifacts[split_name] = {
            "saved_count": saved_count,
            "feature_path": str(feature_path),
            "metadata_path": str(metadata_path),
        }
        metadata_df = pd.DataFrame(split_metadata[split_name], columns=METADATA_COLUMNS)
        if not metadata_df.empty:
            manifest_frames.append(metadata_df)

    manifest = pd.concat(manifest_frames, ignore_index=True) if manifest_frames else pd.DataFrame()
    manifest.to_csv(output_dir / "manifest.csv", index=False)

    split_summary = (
        manifest.groupby(["split", "label"]).size().reset_index(name="num_chunks")
        if not manifest.empty
        else pd.DataFrame(columns=["split", "label", "num_chunks"])
    )
    split_summary.to_csv(output_dir / "split_summary.csv", index=False)

    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    with (output_dir / "feature_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(summarize_stats(train_sum, train_sq_sum, train_frame_count), handle, indent=2)

    print("-" * 72)
    print(f"Finished {config.feature_type.upper()} extraction")
    print(f"saved_chunks   : {len(manifest)}")
    print(f"output_dir     : {output_dir}")
    print(f"manifest       : {output_dir / 'manifest.csv'}")
    print(f"speaker_splits : {output_dir / 'speaker_splits.csv'}")
    print(f"feature_stats  : {output_dir / 'feature_stats.json'}")
    print("NPZ artifacts")
    for split_name in ("train", "val", "test"):
        artifact = split_artifacts[split_name]
        print(
            f"  {split_name:5s} : {artifact['saved_count']} chunks | "
            f"{artifact['feature_path']} | {artifact['metadata_path']}"
        )
    print("Files per split")
    for split_name, count in sorted(split_file_counts.items()):
        print(f"  {split_name:5s} : {count}")
    print("Chunks per split")
    for split_name, count in sorted(split_chunk_counts.items()):
        print(f"  {split_name:5s} : {count}")
    print("Chunk label summary")
    print(split_summary.to_string(index=False))
