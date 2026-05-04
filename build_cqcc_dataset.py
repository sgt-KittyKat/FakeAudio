from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
from scipy.fft import dct

META_LABEL_MAP = {
    "bona-fide": 0,
    "bonafide": 0,
    "real": 0,
    "genuine": 0,
    "spoof": 1,
    "fake": 1,
    "0": 0,
    "1": 1,
}

WORKER_INPUT_DIR_STR = ""
WORKER_LABEL_MAP: Dict[str, int] = {}
WORKER_TARGET_SR: Optional[int] = None
WORKER_CHUNK_DURATION_SEC = 4.0
WORKER_CHUNK_STRIDE_SEC = 4.0
WORKER_INCLUDE_TAIL_CHUNK = False
WORKER_HOP_LENGTH = 160
WORKER_FMIN = 15.0
WORKER_N_BINS = 96
WORKER_BINS_PER_OCTAVE = 12
WORKER_N_UNIFORM_BINS = 96
WORKER_N_COEFFS = 20
WORKER_EPS = 1e-12
WORKER_INCLUDE_DELTAS = False
WORKER_SPLIT_DEFAULT = "train"
WORKER_METADATA_LOOKUP: Dict[str, Tuple[str, str, int]] = {}


def parse_label_map(raw: str) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid label map token '{item}'. Expected name:int.")
        name, value = item.split(":", 1)
        mapping[name.strip()] = int(value.strip())
    if not mapping:
        raise ValueError("Label map is empty.")
    return mapping


def uniform_resample_log_spectrum(log_power: np.ndarray, cqt_freqs: np.ndarray, n_uniform_bins: int) -> np.ndarray:
    uniform_freqs = np.linspace(cqt_freqs[0], cqt_freqs[-1], n_uniform_bins, dtype=np.float64)
    uniform_log = np.empty((n_uniform_bins, log_power.shape[1]), dtype=np.float32)
    for frame_idx in range(log_power.shape[1]):
        uniform_log[:, frame_idx] = np.interp(uniform_freqs, cqt_freqs, log_power[:, frame_idx])
    return uniform_log


def extract_cqcc_base(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    fmin: float,
    n_bins: int,
    bins_per_octave: int,
    n_uniform_bins: int,
    n_coeffs: int,
    eps: float,
) -> np.ndarray:
    min_len = max(1024, hop_length * 2)
    if y.shape[0] < min_len:
        y = np.pad(y, (0, min_len - y.shape[0]), mode="constant")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"n_fft=.* is too large for input signal of length=.*",
            category=UserWarning,
        )
        cqt = librosa.cqt(
            y=y,
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
        )

    power = np.abs(cqt) ** 2
    log_power = np.log(np.maximum(power, eps)).astype(np.float32)
    cqt_freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
    uniform_log = uniform_resample_log_spectrum(log_power, cqt_freqs.astype(np.float64), n_uniform_bins)

    cqcc = dct(uniform_log, type=2, axis=0, norm="ortho").astype(np.float32)
    return cqcc[:n_coeffs, :]


def force_time_length(feat: np.ndarray, target_frames: int) -> np.ndarray:
    # feat shape: (C, F, T)
    t = feat.shape[2]
    if t == target_frames:
        return feat
    if t < target_frames:
        pad = np.zeros((feat.shape[0], feat.shape[1], target_frames - t), dtype=np.float32)
        return np.concatenate([feat, pad], axis=2)
    return feat[:, :, :target_frames]


def wav_to_chunks(y: np.ndarray, chunk_samples: int, stride_samples: int, include_tail: bool) -> List[np.ndarray]:
    if y.shape[0] <= chunk_samples:
        out = np.zeros(chunk_samples, dtype=np.float32)
        out[: y.shape[0]] = y
        return [out]

    chunks: List[np.ndarray] = []
    start = 0
    while start + chunk_samples <= y.shape[0]:
        chunks.append(y[start : start + chunk_samples])
        start += stride_samples

    if include_tail and start < y.shape[0]:
        tail = y[start:]
        if tail.size > 0:
            out = np.zeros(chunk_samples, dtype=np.float32)
            out[: tail.shape[0]] = tail
            chunks.append(out)

    return chunks


def infer_split_from_path(rel_path: str) -> str:
    p = rel_path.lower()
    if "/train/" in f"/{p}/" or p.startswith("train/"):
        return "train"
    if "/val/" in f"/{p}/" or "/valid/" in f"/{p}/" or p.startswith("val/"):
        return "val"
    if "/test/" in f"/{p}/" or p.startswith("test/"):
        return "test"
    return "unspecified"


def infer_speaker_from_name(filename: str) -> str:
    stem = Path(filename).stem
    for sep in ("_", "-", "."):
        if sep in stem:
            return stem.split(sep)[0]
    return "unknown"


def norm_key(s: str) -> str:
    return s.strip().replace("\\", "/").lstrip("./").lower()


def normalize_meta_label(raw: str) -> int:
    key = (raw or "").strip().lower()
    if key not in META_LABEL_MAP:
        raise ValueError(f"Unknown metadata label '{raw}'.")
    return META_LABEL_MAP[key]


def resolve_meta_audio_path(file_value: str, label_bin: int, input_dir: Path) -> Path:
    p = Path(file_value)
    if p.suffix.lower() != ".wav":
        p = p.with_suffix(".wav")
    if p.is_absolute():
        return p

    if len(p.parts) >= 2 and p.parts[0].lower() in {"real", "fake"}:
        return input_dir / p

    cls = "real" if label_bin == 0 else "fake"
    return input_dir / cls / p.name


def build_speaker_split_map(
    speaker_stats: Dict[str, Dict[str, int]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, str]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.0")

    rng = np.random.default_rng(seed)
    speakers = list(speaker_stats.keys())
    rng.shuffle(speakers)
    speakers.sort(
        key=lambda s: (
            speaker_stats[s]["n"],
            abs(speaker_stats[s]["pos"] - speaker_stats[s]["neg"]),
        ),
        reverse=True,
    )

    total_n = sum(speaker_stats[s]["n"] for s in speakers)
    total_pos = sum(speaker_stats[s]["pos"] for s in speakers)
    total_neg = sum(speaker_stats[s]["neg"] for s in speakers)

    targets = {
        "train": {"n": train_ratio * total_n, "pos": train_ratio * total_pos, "neg": train_ratio * total_neg},
        "val": {"n": val_ratio * total_n, "pos": val_ratio * total_pos, "neg": val_ratio * total_neg},
        "test": {"n": test_ratio * total_n, "pos": test_ratio * total_pos, "neg": test_ratio * total_neg},
    }
    buckets = {
        "train": {"n": 0, "pos": 0, "neg": 0},
        "val": {"n": 0, "pos": 0, "neg": 0},
        "test": {"n": 0, "pos": 0, "neg": 0},
    }

    def rel_err(v: float, t: float) -> float:
        if t <= 0:
            return abs(v)
        return abs(v - t) / t

    def score(split: str, speaker: str) -> float:
        cand_n = buckets[split]["n"] + speaker_stats[speaker]["n"]
        cand_pos = buckets[split]["pos"] + speaker_stats[speaker]["pos"]
        cand_neg = buckets[split]["neg"] + speaker_stats[speaker]["neg"]
        return (
            1.0 * rel_err(cand_n, targets[split]["n"])
            + 1.2 * rel_err(cand_pos, targets[split]["pos"])
            + 1.2 * rel_err(cand_neg, targets[split]["neg"])
        )

    speaker_to_split: Dict[str, str] = {}
    for split_name, speaker in zip(["train", "val", "test"], speakers[: min(3, len(speakers))]):
        speaker_to_split[speaker] = split_name
        buckets[split_name]["n"] += speaker_stats[speaker]["n"]
        buckets[split_name]["pos"] += speaker_stats[speaker]["pos"]
        buckets[split_name]["neg"] += speaker_stats[speaker]["neg"]

    for speaker in speakers[min(3, len(speakers)) :]:
        best = min(["train", "val", "test"], key=lambda s: score(s, speaker))
        speaker_to_split[speaker] = best
        buckets[best]["n"] += speaker_stats[speaker]["n"]
        buckets[best]["pos"] += speaker_stats[speaker]["pos"]
        buckets[best]["neg"] += speaker_stats[speaker]["neg"]

    return speaker_to_split


def load_metadata_lookup(
    meta_csv: Path,
    input_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    split_seed: int,
) -> Dict[str, Tuple[str, str, int]]:
    """
    Build lookup: key -> (speaker, split, label)
    Split is computed speaker-disjoint from meta labels.
    """
    lookup: Dict[str, Tuple[str, str, int]] = {}
    if not meta_csv.exists():
        return lookup

    with meta_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"file", "speaker", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {meta_csv}: {sorted(missing)}")

        rows: List[Tuple[str, str, int, Path]] = []
        for row in reader:
            file_value = (row.get("file") or "").strip()
            speaker = (row.get("speaker") or "").strip() or "unknown"
            label = normalize_meta_label((row.get("label") or "").strip())
            wav_path = resolve_meta_audio_path(file_value, label, input_dir)
            rows.append((file_value, speaker, label, wav_path))

    speaker_stats: Dict[str, Dict[str, int]] = {}
    for _, speaker, label, _ in rows:
        if speaker not in speaker_stats:
            speaker_stats[speaker] = {"n": 0, "pos": 0, "neg": 0}
        speaker_stats[speaker]["n"] += 1
        if label == 1:
            speaker_stats[speaker]["pos"] += 1
        else:
            speaker_stats[speaker]["neg"] += 1

    speaker_to_split = build_speaker_split_map(
        speaker_stats=speaker_stats,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=split_seed,
    )

    for file_value, speaker, label, wav_path in rows:
        split = speaker_to_split[speaker]
        rel_key = norm_key(wav_path.name)
        try:
            rel_key = norm_key(str(wav_path.relative_to(input_dir)))
        except ValueError:
            rel_key = norm_key(wav_path.name)
        keys = {
            norm_key(file_value),
            norm_key(Path(file_value).name),
            norm_key(str(wav_path)),
            rel_key,
            norm_key(wav_path.name),
        }
        for k in keys:
            lookup[k] = (speaker, split, label)

    return lookup


def init_worker_context(
    input_dir_str: str,
    label_map: Dict[str, int],
    target_sr: Optional[int],
    chunk_duration_sec: float,
    chunk_stride_sec: float,
    include_tail_chunk: bool,
    hop_length: int,
    fmin: float,
    n_bins: int,
    bins_per_octave: int,
    n_uniform_bins: int,
    n_coeffs: int,
    eps: float,
    include_deltas: bool,
    split_default: str,
    metadata_lookup: Dict[str, Tuple[str, str, int]],
) -> None:
    global WORKER_INPUT_DIR_STR
    global WORKER_LABEL_MAP
    global WORKER_TARGET_SR
    global WORKER_CHUNK_DURATION_SEC
    global WORKER_CHUNK_STRIDE_SEC
    global WORKER_INCLUDE_TAIL_CHUNK
    global WORKER_HOP_LENGTH
    global WORKER_FMIN
    global WORKER_N_BINS
    global WORKER_BINS_PER_OCTAVE
    global WORKER_N_UNIFORM_BINS
    global WORKER_N_COEFFS
    global WORKER_EPS
    global WORKER_INCLUDE_DELTAS
    global WORKER_SPLIT_DEFAULT
    global WORKER_METADATA_LOOKUP

    WORKER_INPUT_DIR_STR = input_dir_str
    WORKER_LABEL_MAP = label_map
    WORKER_TARGET_SR = target_sr
    WORKER_CHUNK_DURATION_SEC = chunk_duration_sec
    WORKER_CHUNK_STRIDE_SEC = chunk_stride_sec
    WORKER_INCLUDE_TAIL_CHUNK = include_tail_chunk
    WORKER_HOP_LENGTH = hop_length
    WORKER_FMIN = fmin
    WORKER_N_BINS = n_bins
    WORKER_BINS_PER_OCTAVE = bins_per_octave
    WORKER_N_UNIFORM_BINS = n_uniform_bins
    WORKER_N_COEFFS = n_coeffs
    WORKER_EPS = eps
    WORKER_INCLUDE_DELTAS = include_deltas
    WORKER_SPLIT_DEFAULT = split_default
    WORKER_METADATA_LOOKUP = metadata_lookup


def process_one_file(
    idx: int,
    wav_path_str: str,
    input_dir_str: str,
    label_map: Dict[str, int],
    target_sr: Optional[int],
    chunk_duration_sec: float,
    chunk_stride_sec: float,
    include_tail_chunk: bool,
    hop_length: int,
    fmin: float,
    n_bins: int,
    bins_per_octave: int,
    n_uniform_bins: int,
    n_coeffs: int,
    eps: float,
    include_deltas: bool,
    split_default: str,
    metadata_lookup: Dict[str, Tuple[str, str, int]],
) -> Optional[Tuple[int, str, int, List[np.ndarray], List[int], List[int], List[int], List[int], str, str, str]]:
    wav_path = Path(wav_path_str)
    input_dir = Path(input_dir_str)
    class_name = wav_path.parent.name
    if class_name not in label_map:
        return None

    y, sr = librosa.load(str(wav_path), sr=target_sr, mono=True)
    chunk_samples = int(round(chunk_duration_sec * sr))
    stride_samples = int(round(chunk_stride_sec * sr))
    stride_samples = max(1, stride_samples)

    chunks = wav_to_chunks(y, chunk_samples=chunk_samples, stride_samples=stride_samples, include_tail=include_tail_chunk)

    # For 4s at 16k with hop=160 this is 400 frames.
    target_frames = int(round(chunk_samples / hop_length))

    chunk_features: List[np.ndarray] = []
    chunk_ids: List[int] = []
    start_samples: List[int] = []
    num_audio_samples: List[int] = []
    num_feature_frames: List[int] = []
    for chunk_idx, chunk_y in enumerate(chunks):
        base = extract_cqcc_base(
            y=chunk_y,
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            n_uniform_bins=n_uniform_bins,
            n_coeffs=n_coeffs,
            eps=eps,
        )

        if include_deltas:
            delta = librosa.feature.delta(base)
            delta2 = librosa.feature.delta(base, order=2)
            feat = np.stack([base, delta, delta2], axis=0).astype(np.float32)  # (C=3,F,T)
        else:
            feat = np.expand_dims(base, axis=0).astype(np.float32)  # (C=1,F,T)

        feat = force_time_length(feat, target_frames)
        chunk_features.append(feat)
        chunk_ids.append(chunk_idx)
        start_samples.append(chunk_idx * stride_samples)
        num_audio_samples.append(int(chunk_y.shape[0]))
        num_feature_frames.append(int(feat.shape[2]))

    rel_path = str(wav_path.relative_to(input_dir).as_posix())
    basename = wav_path.name
    label = label_map[class_name]
    split = infer_split_from_path(rel_path)
    if split == "unspecified":
        split = split_default
    speaker = infer_speaker_from_name(wav_path.name)
    keys = (
        basename.strip().replace("\\", "/").lower(),
        rel_path.strip().replace("\\", "/").lower(),
        f"{wav_path.parent.name}/{basename}".strip().replace("\\", "/").lower(),
    )
    for key in keys:
        if key in metadata_lookup:
            speaker, split_meta, label_meta = metadata_lookup[key]
            label = int(label_meta)
            if split_meta and split_meta != "unspecified":
                split = split_meta
            break

    return (
        idx,
        rel_path,
        label,
        chunk_features,
        chunk_ids,
        start_samples,
        num_audio_samples,
        num_feature_frames,
        speaker,
        split,
        basename,
    )


def process_one_file_from_tuple(args: Tuple) -> Optional[Tuple[int, str, int, List[np.ndarray], List[int], List[int], List[int], List[int], str, str, str]]:
    return process_one_file(*args)


def process_one_file_mp(item: Tuple[int, str]) -> Optional[Tuple[int, str, int, List[np.ndarray], List[int], List[int], List[int], List[int], str, str, str]]:
    idx, wav_path_str = item
    return process_one_file(
        idx=idx,
        wav_path_str=wav_path_str,
        input_dir_str=WORKER_INPUT_DIR_STR,
        label_map=WORKER_LABEL_MAP,
        target_sr=WORKER_TARGET_SR,
        chunk_duration_sec=WORKER_CHUNK_DURATION_SEC,
        chunk_stride_sec=WORKER_CHUNK_STRIDE_SEC,
        include_tail_chunk=WORKER_INCLUDE_TAIL_CHUNK,
        hop_length=WORKER_HOP_LENGTH,
        fmin=WORKER_FMIN,
        n_bins=WORKER_N_BINS,
        bins_per_octave=WORKER_BINS_PER_OCTAVE,
        n_uniform_bins=WORKER_N_UNIFORM_BINS,
        n_coeffs=WORKER_N_COEFFS,
        eps=WORKER_EPS,
        include_deltas=WORKER_INCLUDE_DELTAS,
        split_default=WORKER_SPLIT_DEFAULT,
        metadata_lookup=WORKER_METADATA_LOOKUP,
    )


def collect_wavs(root: Path) -> List[Path]:
    return sorted(root.rglob("*.wav"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed-size CQCC chunk dataset (N,C,F,T) from WAV files.")
    parser.add_argument("--input-dir", type=Path, default=Path("data"), help="Root folder containing WAV files.")
    parser.add_argument("--output", type=Path, default=Path("cqcc_dataset_chunks.npz"), help="Output .npz file.")
    parser.add_argument("--label-map", type=str, default="real:0,fake:1", help="Comma-separated label mapping.")
    parser.add_argument("--target-sr", type=int, default=16000, help="Resample audio to this rate. Use <=0 to keep original SR.")
    parser.add_argument("--chunk-duration", type=float, default=4.0, help="Chunk length in seconds.")
    parser.add_argument("--chunk-stride", type=float, default=4.0, help="Chunk stride in seconds.")
    parser.add_argument("--include-tail-chunk", action="store_true", help="Include final partial chunk (zero-padded).")
    parser.add_argument("--hop-length", type=int, default=160, help="Hop length in samples.")
    parser.add_argument("--fmin", type=float, default=15.0, help="Minimum CQT frequency (Hz).")
    parser.add_argument("--n-bins", type=int, default=96, help="Number of CQT bins.")
    parser.add_argument("--bins-per-octave", type=int, default=12, help="CQT bins per octave.")
    parser.add_argument("--n-uniform-bins", type=int, default=96, help="Linear-frequency bins after interpolation.")
    parser.add_argument("--n-coeffs", type=int, default=20, help="Number of CQCC coefficients to retain.")
    parser.add_argument("--eps", type=float, default=1e-12, help="Log floor epsilon.")
    parser.add_argument("--include-deltas", action="store_true", help="Use 3 channels: static, delta, delta-delta.")
    parser.add_argument("--max-files", type=int, default=0, help="Optional debug cap on number of files.")
    parser.add_argument("--log-every", type=int, default=100, help="Progress log interval in number of files.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel worker processes.")
    parser.add_argument("--split-default", type=str, default="train", help="Fallback split when not inferable from path.")
    parser.add_argument("--manifest-csv", type=Path, default=None, help="Optional manifest CSV path. Default: <output_stem>_manifest.csv")
    parser.add_argument("--meta-csv", type=Path, default=Path("data/meta.csv"), help="Metadata CSV with file,speaker,label.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio for speaker-disjoint split.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio for speaker-disjoint split.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio for speaker-disjoint split.")
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed for speaker-disjoint split assignment.")

    args = parser.parse_args()

    label_map = parse_label_map(args.label_map)
    wav_paths = collect_wavs(args.input_dir)
    if args.max_files > 0:
        wav_paths = wav_paths[: args.max_files]
    if not wav_paths:
        raise SystemExit(f"No WAV files found in {args.input_dir}")

    metadata_lookup = load_metadata_lookup(
        meta_csv=args.meta_csv,
        input_dir=args.input_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
    )
    if metadata_lookup:
        print(f"Loaded metadata lookup from {args.meta_csv} ({len(metadata_lookup)} keys)")
    else:
        print(f"No metadata lookup loaded from {args.meta_csv}; falling back to filename/path inference.")

    target_sr = args.target_sr if args.target_sr > 0 else None
    total = len(wav_paths)
    start_time = time.time()
    skipped = 0

    print(f"Starting fixed-chunk CQCC extraction for {total} WAV files...")

    tasks = [(idx, str(wav_path)) for idx, wav_path in enumerate(wav_paths, start=1)]

    results: List[Tuple[int, str, int, List[np.ndarray], List[int], List[int], List[int], List[int], str, str, str]] = []

    if args.num_workers <= 1:
        for idx, task in enumerate(tasks, start=1):
            r = process_one_file(
                idx=task[0],
                wav_path_str=task[1],
                input_dir_str=str(args.input_dir),
                label_map=label_map,
                target_sr=target_sr,
                chunk_duration_sec=args.chunk_duration,
                chunk_stride_sec=args.chunk_stride,
                include_tail_chunk=args.include_tail_chunk,
                hop_length=args.hop_length,
                fmin=args.fmin,
                n_bins=args.n_bins,
                bins_per_octave=args.bins_per_octave,
                n_uniform_bins=args.n_uniform_bins,
                n_coeffs=args.n_coeffs,
                eps=args.eps,
                include_deltas=args.include_deltas,
                split_default=args.split_default,
                metadata_lookup=metadata_lookup,
            )
            if r is None:
                skipped += 1
            else:
                results.append(r)

            if idx % args.log_every == 0 or idx == total:
                elapsed = time.time() - start_time
                rate = idx / elapsed if elapsed > 0 else 0.0
                rem = total - idx
                eta = rem / rate if rate > 0 else float("inf")
                eta_str = f"{eta:.1f}s" if np.isfinite(eta) else "unknown"
                print(f"[{idx}/{total} | {(idx/total)*100:.1f}%] elapsed={elapsed:.1f}s eta={eta_str} processed={len(results)} skipped={skipped}")
    else:
        print(f"Using {args.num_workers} worker processes")
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=init_worker_context,
            initargs=(
                str(args.input_dir),
                label_map,
                target_sr,
                args.chunk_duration,
                args.chunk_stride,
                args.include_tail_chunk,
                args.hop_length,
                args.fmin,
                args.n_bins,
                args.bins_per_octave,
                args.n_uniform_bins,
                args.n_coeffs,
                args.eps,
                args.include_deltas,
                args.split_default,
                metadata_lookup,
            ),
        ) as ex:
            for idx, r in enumerate(ex.map(process_one_file_mp, tasks), start=1):
                if r is None:
                    skipped += 1
                else:
                    results.append(r)

                if idx % args.log_every == 0 or idx == total:
                    elapsed = time.time() - start_time
                    rate = idx / elapsed if elapsed > 0 else 0.0
                    rem = total - idx
                    eta = rem / rate if rate > 0 else float("inf")
                    eta_str = f"{eta:.1f}s" if np.isfinite(eta) else "unknown"
                    print(f"[{idx}/{total} | {(idx/total)*100:.1f}%] elapsed={elapsed:.1f}s eta={eta_str} processed={len(results)} skipped={skipped}")

    if not results:
        raise SystemExit("No files were processed. Check --input-dir and --label-map.")

    results.sort(key=lambda x: x[0])

    all_chunks: List[np.ndarray] = []
    all_labels: List[int] = []
    all_file: List[str] = []
    all_speaker: List[str] = []
    all_split: List[str] = []
    all_chunk_index: List[int] = []
    all_start_sample: List[int] = []
    all_feature_type: List[str] = []
    all_num_audio_samples: List[int] = []
    all_num_feature_frames: List[int] = []
    file_num_chunks: List[int] = []

    feature_type = "cqcc"
    for _, rel_path, label, chunk_feats, chunk_ids, start_samples, chunk_audio_samples, chunk_feature_frames, speaker, split, basename in results:
        file_num_chunks.append(len(chunk_feats))
        for cf, cid, s0, nas, nff in zip(chunk_feats, chunk_ids, start_samples, chunk_audio_samples, chunk_feature_frames):
            all_chunks.append(cf)
            all_labels.append(label)
            all_file.append(basename)
            all_speaker.append(speaker)
            all_split.append(split)
            all_chunk_index.append(cid)
            all_start_sample.append(s0)
            all_feature_type.append(feature_type)
            all_num_audio_samples.append(nas)
            all_num_feature_frames.append(nff)

    X = np.stack(all_chunks, axis=0).astype(np.float32)  # (N,C,F,T)
    y = np.array(all_labels, dtype=np.int64)
    file_name = np.array(all_file, dtype="U")
    speaker = np.array(all_speaker, dtype="U")
    split = np.array(all_split, dtype="U")
    chunk_index = np.array(all_chunk_index, dtype=np.int32)
    start_sample = np.array(all_start_sample, dtype=np.int64)
    feature_type_arr = np.array(all_feature_type, dtype="U")
    num_audio_samples_arr = np.array(all_num_audio_samples, dtype=np.int32)
    num_feature_frames_arr = np.array(all_num_feature_frames, dtype=np.int32)
    num_chunks_per_file = np.array(file_num_chunks, dtype=np.int32)
    unique_splits, split_counts = np.unique(split, return_counts=True)
    split_count_map = {str(s): int(c) for s, c in zip(unique_splits.tolist(), split_counts.tolist())}

    manifest_csv = args.manifest_csv
    if manifest_csv is None:
        manifest_csv = args.output.with_name(f"{args.output.stem}_manifest.csv")
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_index", "file", "speaker", "label", "split", "chunk_index", "start_sample", "feature_type", "num_audio_samples", "num_feature_frames"])
        for i in range(X.shape[0]):
            writer.writerow(
                [
                    i,
                    file_name[i],
                    speaker[i],
                    int(y[i]),
                    split[i],
                    int(chunk_index[i]),
                    int(start_sample[i]),
                    feature_type_arr[i],
                    int(num_audio_samples_arr[i]),
                    int(num_feature_frames_arr[i]),
                ]
            )

    metadata = {
        "transform": "CQCC",
        "format": "N,C,F,T",
        "pipeline": ["signal", "chunk_4s", "CQT", "power", "log", "uniform_linear_resample", "DCT-II"],
        "input_dir": str(args.input_dir),
        "label_map": label_map,
        "target_sr": target_sr,
        "chunk_duration_sec": args.chunk_duration,
        "chunk_stride_sec": args.chunk_stride,
        "include_tail_chunk": args.include_tail_chunk,
        "hop_length": args.hop_length,
        "fmin": args.fmin,
        "n_bins": args.n_bins,
        "bins_per_octave": args.bins_per_octave,
        "n_uniform_bins": args.n_uniform_bins,
        "n_coeffs": args.n_coeffs,
        "include_deltas": args.include_deltas,
        "channels": int(X.shape[1]),
        "coeffs": int(X.shape[2]),
        "frames": int(X.shape[3]),
        "num_files": int(len(results)),
        "num_chunks": int(X.shape[0]),
        "manifest_csv": str(manifest_csv),
        "split_strategy": "speaker_disjoint_label_aware_from_meta",
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "split_seed": args.split_seed,
        "chunk_counts_by_split": split_count_map,
        "manifest_fields": [
            "file",
            "speaker",
            "label",
            "split",
            "chunk_index",
            "start_sample",
            "feature_type",
            "num_audio_samples",
            "num_feature_frames",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X=X,
        y=y,
        num_chunks_per_file=num_chunks_per_file,
        metadata=np.array(json.dumps(metadata), dtype="U"),
    )

    print(f"Saved CQCC chunk dataset to: {args.output}")
    print(f"Saved manifest CSV to: {manifest_csv}")
    print(f"X shape: {X.shape} | files: {len(results)} | chunks: {X.shape[0]}")
    print(f"Chunk counts by split: {split_count_map}")
    print(f"Skipped files (unknown labels): {skipped}")


if __name__ == "__main__":
    main()
