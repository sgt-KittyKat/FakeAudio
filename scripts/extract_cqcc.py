from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from scipy.fft import dct
from tqdm import tqdm

from audio_feature_pipeline import (
    METADATA_COLUMNS,
    apply_pre_emphasis,
    assign_speaker_splits,
    chunk_waveform,
    load_metadata,
    load_waveform,
    print_split_overview,
    save_split_npz,
    summarize_stats,
)


@dataclass
class CQCCExtractionConfig:
    feature_type: str = "cqcc"
    sample_rate: int = 16000
    chunk_seconds: float = 4.0
    hop_seconds: float = 4.0
    hop_length: int = 160
    fmin: float = 15.0
    n_bins: int = 96
    bins_per_octave: int = 12
    n_uniform_bins: int = 96
    n_features: int = 20
    pre_emphasis: float = 0.97
    include_deltas: bool = True
    pad_mode: str = "constant"
    split_seed: int = 13
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    target_frames: int = 400
    eps: float = 1e-12

    @property
    def chunk_samples(self) -> int:
        return int(self.sample_rate * self.chunk_seconds)

    @property
    def chunk_hop_samples(self) -> int:
        return int(self.sample_rate * self.hop_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract CQCC features from the in-the-wild deepfake dataset."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/release_in_the_wild"))
    parser.add_argument("--metadata-path", type=Path, default=Path("data/meta.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/cqcc"))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-seconds", type=float, default=4.0)
    parser.add_argument("--hop-seconds", type=float, default=4.0)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--fmin", type=float, default=15.0)
    parser.add_argument("--n-bins", type=int, default=96)
    parser.add_argument("--bins-per-octave", type=int, default=12)
    parser.add_argument("--n-uniform-bins", type=int, default=96)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--pre-emphasis", type=float, default=0.97)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--split-seed", type=int, default=13)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--target-frames", type=int, default=400)
    parser.add_argument("--disable-deltas", action="store_true")
    parser.add_argument("--limit-files", type=int, default=None)
    return parser


def args_to_config(args: argparse.Namespace) -> CQCCExtractionConfig:
    return CQCCExtractionConfig(
        sample_rate=args.sample_rate,
        chunk_seconds=args.chunk_seconds,
        hop_seconds=args.hop_seconds,
        hop_length=args.hop_length,
        fmin=args.fmin,
        n_bins=args.n_bins,
        bins_per_octave=args.bins_per_octave,
        n_uniform_bins=args.n_uniform_bins,
        n_features=args.n_features,
        pre_emphasis=args.pre_emphasis,
        include_deltas=not args.disable_deltas,
        split_seed=args.split_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        target_frames=args.target_frames,
        eps=args.eps,
    )


def uniform_resample_log_spectrum(
    log_power: np.ndarray,
    cqt_freqs: np.ndarray,
    n_uniform_bins: int,
) -> np.ndarray:
    uniform_freqs = np.linspace(cqt_freqs[0], cqt_freqs[-1], n_uniform_bins, dtype=np.float64)
    uniform_log = np.empty((n_uniform_bins, log_power.shape[1]), dtype=np.float32)
    for frame_idx in range(log_power.shape[1]):
        uniform_log[:, frame_idx] = np.interp(uniform_freqs, cqt_freqs, log_power[:, frame_idx])
    return uniform_log


def extract_cqcc_base(chunk: np.ndarray, config: CQCCExtractionConfig) -> np.ndarray:
    cqt = librosa.cqt(
        y=chunk,
        sr=config.sample_rate,
        hop_length=config.hop_length,
        fmin=config.fmin,
        n_bins=config.n_bins,
        bins_per_octave=config.bins_per_octave,
    )
    power = np.abs(cqt) ** 2
    log_power = np.log(np.maximum(power, config.eps)).astype(np.float32)
    cqt_freqs = librosa.cqt_frequencies(
        n_bins=config.n_bins,
        fmin=config.fmin,
        bins_per_octave=config.bins_per_octave,
    )
    uniform_log = uniform_resample_log_spectrum(
        log_power=log_power,
        cqt_freqs=cqt_freqs.astype(np.float64),
        n_uniform_bins=config.n_uniform_bins,
    )
    cqcc = dct(uniform_log, type=2, axis=0, norm="ortho").astype(np.float32)
    return cqcc[: config.n_features, :]


def ensure_target_frames(features: np.ndarray, target_frames: int) -> np.ndarray:
    current_frames = features.shape[-1]
    if current_frames == target_frames:
        return features
    if current_frames > target_frames:
        return features[..., :target_frames].copy()
    pad_width = ((0, 0), (0, 0), (0, target_frames - current_frames))
    return np.pad(features, pad_width=pad_width, mode="constant", constant_values=0.0)


def extract_feature_tensor(chunk: torch.Tensor, config: CQCCExtractionConfig) -> torch.Tensor:
    chunk_np = chunk.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
    base = extract_cqcc_base(chunk_np, config)
    streams = [base]
    if config.include_deltas:
        delta = librosa.feature.delta(base)
        delta2 = librosa.feature.delta(base, order=2)
        streams.extend([delta.astype(np.float32), delta2.astype(np.float32)])
    stacked = np.stack(streams, axis=0).astype(np.float32, copy=False)
    stacked = ensure_target_frames(stacked, config.target_frames)
    return torch.from_numpy(stacked)


def print_header(config: CQCCExtractionConfig, args: argparse.Namespace) -> None:
    print("=" * 72)
    print("Starting CQCC extraction")
    print(f"data_root         : {args.data_root}")
    print(f"metadata_path     : {args.metadata_path}")
    print(f"output_dir        : {args.output_dir}")
    print(f"sample_rate       : {config.sample_rate}")
    print(f"chunk_seconds     : {config.chunk_seconds}")
    print(f"hop_seconds       : {config.hop_seconds}")
    print(f"hop_length        : {config.hop_length}")
    print(f"fmin              : {config.fmin}")
    print(f"n_bins            : {config.n_bins}")
    print(f"bins_per_octave   : {config.bins_per_octave}")
    print(f"n_uniform_bins    : {config.n_uniform_bins}")
    print(f"n_features        : {config.n_features}")
    print(f"pre_emphasis      : {config.pre_emphasis}")
    print(f"include_deltas    : {config.include_deltas}")
    print(f"target_frames     : {config.target_frames}")
    print(f"limit_files       : {args.limit_files}")
    print("=" * 72)


def run_extraction(args: argparse.Namespace) -> None:
    config = args_to_config(args)
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
        desc="CQCC files",
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
            features = extract_feature_tensor(chunk, config)
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

    speaker_summary = (
        df[["speaker", "split"]]
        .drop_duplicates()
        .sort_values(["split", "speaker"])
        .reset_index(drop=True)
    )
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
    print("Finished CQCC extraction")
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_extraction(args)


if __name__ == "__main__":
    main()
