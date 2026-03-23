from __future__ import annotations

import argparse
import json
import time

from engine import run_once


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live IPL predictor automation runner")
    parser.add_argument("--mode", choices=["once", "loop"], default="once")
    parser.add_argument("--interval-seconds", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "once":
        summary = run_once()
        print(json.dumps(summary, indent=2))
        return

    while True:
        summary = run_once()
        print(json.dumps(summary, indent=2))
        time.sleep(max(10, args.interval_seconds))


if __name__ == "__main__":
    main()
