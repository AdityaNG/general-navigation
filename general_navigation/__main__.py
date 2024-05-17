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
    parser.add_argument(
        "--silent",
        "-s",
        action="store_true",
        help="Don't use the UI, run silently",
    )
    parser.add_argument(
        "--max_iters",
        "-i",
        default=-1,
        type=int,
        help="Number of iterations to run for, -1 to run indefinately",
    )

    args = parser.parse_args()
    main(args)
