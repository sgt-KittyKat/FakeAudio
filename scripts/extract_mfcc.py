from audio_feature_pipeline import build_parser, run_extraction


def main() -> None:
    parser = build_parser("mfcc", "data/processed/mfcc")
    args = parser.parse_args()
    run_extraction(args, "mfcc")


if __name__ == "__main__":
    main()
