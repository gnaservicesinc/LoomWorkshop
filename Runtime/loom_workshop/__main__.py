import argparse
import sys

from .engine import run_cli


def main():
    parser = argparse.ArgumentParser(prog="loom_workshop")
    parser.add_argument("command", choices=("plan", "train"))
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    return run_cli(args.command, args.request)


if __name__ == "__main__":
    sys.exit(main())
