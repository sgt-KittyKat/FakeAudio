from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


LABEL_TO_FOLDER = {
    "bona-fide": "real",
    "bonafide": "real",
    "real": "real",
    "genuine": "real",
    "spoof": "fake",
    "fake": "fake",
}


def normalize_label(label: str) -> str:
    key = str(label).strip().lower()
    if key not in LABEL_TO_FOLDER:
        raise ValueError(f"Unsupported label found in metadata: {label}")
    return LABEL_TO_FOLDER[key]


def organize_audio(
    data_root: Path,
    metadata_path: Path,
    mode: str,
) -> None:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    required_columns = {"file", "label"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in metadata: {missing_columns}")

    real_dir = data_root / "real"
    fake_dir = data_root / "fake"

    if mode != "dry-run":
        real_dir.mkdir(parents=True, exist_ok=True)
        fake_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    copied = 0
    already_exists = 0
    missing = []

    for row in df.itertuples(index=False):
        filename = getattr(row, "file")
        label = getattr(row, "label")

        target_folder = normalize_label(label)

        source_path = data_root / filename
        target_path = data_root / target_folder / filename

        # If already organized, skip
        if target_path.exists():
            already_exists += 1
            continue

        # If source is missing, record it
        if not source_path.exists():
            missing.append(str(source_path))
            continue

        if mode == "dry-run":
            print(f"[DRY RUN] {source_path} -> {target_path}")
        elif mode == "copy":
            shutil.copy2(source_path, target_path)
            copied += 1
        elif mode == "move":
            shutil.move(str(source_path), str(target_path))
            moved += 1
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    print("\nAudio organization summary")
    print("-" * 50)
    print(f"Mode           : {mode}")
    print(f"Data root      : {data_root}")
    print(f"Metadata       : {metadata_path}")
    print(f"Already exists : {already_exists}")
    print(f"Moved          : {moved}")
    print(f"Copied         : {copied}")
    print(f"Missing files  : {len(missing)}")

    if missing:
        print("\nFirst missing files:")
        for path in missing[:10]:
            print(f"  {path}")

        raise FileNotFoundError(
            f"{len(missing)} files listed in metadata were not found. "
            "Check your extracted dataset path."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Organize flat in-the-wild audio files into real/fake folders using meta.csv."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/release_in_the_wild"))
    parser.add_argument("--metadata-path", type=Path, default=Path("data/meta.csv"))
    parser.add_argument(
        "--mode",
        choices=["dry-run", "copy", "move"],
        default="dry-run",
        help="dry-run shows actions, copy duplicates files, move reorganizes files without duplicating storage.",
    )

    args = parser.parse_args()

    organize_audio(
        data_root=args.data_root,
        metadata_path=args.metadata_path,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()