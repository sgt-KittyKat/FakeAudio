from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


METHOD_SCRIPTS = {
    "mfcc": "extract_mfcc.py",
    "lfcc": "extract_lfcc.py",
    "cqcc": "extract_cqcc.py",
}


def resolve_python_executable(explicit_python: str | None) -> str:
    if explicit_python:
        return explicit_python

    repo_python = Path.cwd() / "venv" / "bin" / "python"
    if repo_python.exists():
        return str(repo_python)
    return sys.executable


def run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def build_shared_args(args: argparse.Namespace, output_dir: Path) -> list[str]:
    shared = [
        "--data-root",
        str(args.data_root),
        "--metadata-path",
        str(args.metadata_path),
        "--output-dir",
        str(output_dir),
        "--sample-rate",
        str(args.sample_rate),
        "--chunk-seconds",
        str(args.chunk_seconds),
        "--hop-seconds",
        str(args.hop_seconds),
        "--pre-emphasis",
        str(args.pre_emphasis),
        "--split-seed",
        str(args.split_seed),
        "--train-ratio",
        str(args.train_ratio),
        "--val-ratio",
        str(args.val_ratio),
        "--test-ratio",
        str(args.test_ratio),
        "--target-frames",
        str(args.target_frames),
    ]
    if args.limit_files is not None:
        shared.extend(["--limit-files", str(args.limit_files)])
    if args.disable_deltas:
        shared.append("--disable-deltas")
    return shared


def build_method_args(method: str, args: argparse.Namespace) -> list[str]:
    output_dir = args.processed_root / method
    shared = build_shared_args(args, output_dir)
    if method == "mfcc":
        return shared + [
            "--frame-length-ms",
            str(args.frame_length_ms),
            "--frame-hop-ms",
            str(args.frame_hop_ms),
            "--n-fft",
            str(args.n_fft),
            "--n-features",
            str(args.n_features),
            "--n-filters",
            str(args.n_filters),
        ]
    if method == "lfcc":
        return shared + [
            "--frame-length-ms",
            str(args.frame_length_ms),
            "--frame-hop-ms",
            str(args.frame_hop_ms),
            "--n-fft",
            str(args.n_fft),
            "--n-features",
            str(args.n_features),
            "--n-filters",
            str(args.n_filters),
        ]
    if method == "cqcc":
        return shared + [
            "--hop-length",
            str(args.cqcc_hop_length),
            "--fmin",
            str(args.cqcc_fmin),
            "--n-bins",
            str(args.cqcc_n_bins),
            "--bins-per-octave",
            str(args.cqcc_bins_per_octave),
            "--n-uniform-bins",
            str(args.cqcc_n_uniform_bins),
            "--n-features",
            str(args.n_features),
            "--eps",
            str(args.cqcc_eps),
        ]
    raise ValueError(f"Unsupported method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all preprocessing pipelines with a shared contract.")
    parser.add_argument("--data-root", type=Path, default=Path("data/release_in_the_wild"))
    parser.add_argument("--metadata-path", type=Path, default=Path("data/meta.csv"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--methods", nargs="+", default=["mfcc", "lfcc", "cqcc"])
    parser.add_argument("--python", type=str, default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-seconds", type=float, default=4.0)
    parser.add_argument("--hop-seconds", type=float, default=4.0)
    parser.add_argument("--pre-emphasis", type=float, default=0.97)
    parser.add_argument("--split-seed", type=int, default=13)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--target-frames", type=int, default=400)
    parser.add_argument("--disable-deltas", action="store_true")
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--frame-length-ms", type=float, default=25.0)
    parser.add_argument("--frame-hop-ms", type=float, default=10.0)
    parser.add_argument("--n-fft", type=int, default=400)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--n-filters", type=int, default=40)
    parser.add_argument("--cqcc-hop-length", type=int, default=160)
    parser.add_argument("--cqcc-fmin", type=float, default=15.0)
    parser.add_argument("--cqcc-n-bins", type=int, default=96)
    parser.add_argument("--cqcc-bins-per-octave", type=int, default=12)
    parser.add_argument("--cqcc-n-uniform-bins", type=int, default=96)
    parser.add_argument("--cqcc-eps", type=float, default=1e-12)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    python_executable = resolve_python_executable(args.python)
    requested_methods = args.methods
    unknown = sorted(set(requested_methods) - set(METHOD_SCRIPTS))
    if unknown:
        raise SystemExit(f"Unsupported methods requested: {unknown}")

    for method in requested_methods:
        script_path = script_dir / METHOD_SCRIPTS[method]
        command = [python_executable, str(script_path), *build_method_args(method, args)]
        run_command(command)

    if not args.skip_validation:
        validation_command = [
            python_executable,
            str(script_dir / "validate_feature_outputs.py"),
            "--processed-root",
            str(args.processed_root),
            "--methods",
            *requested_methods,
        ]
        run_command(validation_command)


if __name__ == "__main__":
    main()
