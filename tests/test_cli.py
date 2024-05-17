import os
from general_navigation.cli import main


def test_cli_main():
    class Args:
        device: str = "auto"
        media: str = os.path.join(os.path.dirname(__file__), "assets/test.mp4")
        silent: bool = True
        max_iters: int = 3

    args = Args()

    print(os.path.abspath(os.curdir))

    main(args)
