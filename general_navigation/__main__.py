"""Entry point for general_navigation."""

from general_navigation.cli import main  # pragma: no cover

if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser("general_navigation")
    parser.add_argument(
        "--device", "-d", default="auto", choices=["auto", "cuda", "cpu"]
    )
    parser.add_argument(
        "--media",
        "-m",
        default="media/test.mp4",
        help="File path, use camera index if you want to use the webcam",
    )

    args = parser.parse_args()
    main(args)
