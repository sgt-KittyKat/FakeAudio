from audio_feature_pipeline import build_parser, run_extraction


def main() -> None:
    parser = build_parser("lfcc", "data/processed/lfcc")
    args = parser.parse_args()
    run_extraction(args, "lfcc")


if __name__ == "__main__":
    main()
