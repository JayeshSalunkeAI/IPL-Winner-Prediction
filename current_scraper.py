#!/usr/bin/env python3
"""Current/live match scraper for Cricbuzz.

Scrapes one match from scorecard+squads URLs and upserts into a CSV.
Useful for running in intervals to keep latest match state in sync.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from historical_scraper import scrape_one


def upsert_csv(row: dict, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        df = pd.read_csv(output_csv)
        if "match_id" in df.columns:
            df = df[df["match_id"].astype(str) != str(row["match_id"])]
        out = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        out = pd.DataFrame([row])

    out.to_csv(output_csv, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape current match and upsert CSV")
    parser.add_argument("--match-id", required=True, help="Cricbuzz match id")
    parser.add_argument(
        "--scorecard-url",
        default="",
        help="Optional explicit scorecard URL",
    )
    parser.add_argument(
        "--squads-url",
        default="",
        help="Optional explicit squads URL",
    )
    parser.add_argument(
        "--output",
        default="data/raw/cricbuzz_current_matches.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--watch-seconds",
        type=int,
        default=0,
        help="If >0, keep scraping in a loop every N seconds",
    )
    return parser.parse_args()


def build_urls(match_id: str, scorecard_url: str, squads_url: str) -> tuple[str, str]:
    if not scorecard_url:
        scorecard_url = f"https://www.cricbuzz.com/live-cricket-scorecard/{match_id}/match"
    if not squads_url:
        squads_url = f"https://www.cricbuzz.com/cricket-match-squads/{match_id}/match"
    return scorecard_url, squads_url


def run_once(match_id: str, scorecard_url: str, squads_url: str, output: Path) -> None:
    row = scrape_one(match_id, scorecard_url, squads_url)
    upsert_csv(row, output)
    print(
        f"Upserted match {match_id} to {output} | "
        f"winner={row.get('winner')} | result={row.get('result_text')}"
    )


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    score_url, squads_url = build_urls(args.match_id, args.scorecard_url, args.squads_url)

    if args.watch_seconds <= 0:
        run_once(args.match_id, score_url, squads_url, output)
        return 0

    interval = max(args.watch_seconds, 10)
    print(f"Watching match {args.match_id} every {interval}s; writing to {output}")
    while True:
        try:
            run_once(args.match_id, score_url, squads_url, output)
        except Exception as exc:
            print(f"Scrape error: {exc}")
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
