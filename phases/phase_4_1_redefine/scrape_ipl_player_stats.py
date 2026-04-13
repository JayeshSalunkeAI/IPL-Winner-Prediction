from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests


COMP_URL = "https://ipl-stats-sports-mechanic.s3.ap-south-1.amazonaws.com/ipl/mc/competition.js"
UA = {"User-Agent": "Mozilla/5.0"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scrape IPL season player stats (batting and bowling)")
    p.add_argument("--years", nargs="+", type=int, default=[2020, 2021, 2023, 2024, 2025, 2026])
    p.add_argument("--out-dir", default="phases/phase_4_1_redefine/data")
    return p.parse_args()


def parse_jsonp_payload(text: str) -> Any:
    m = re.search(r"^[^(]+\((.*)\)\s*;?\s*$", text, flags=re.S)
    if not m:
        raise ValueError("Unexpected JSONP format")
    return json.loads(m.group(1))


def load_competitions() -> list[dict[str, Any]]:
    raw = requests.get(COMP_URL, headers=UA, timeout=30)
    raw.raise_for_status()
    payload = parse_jsonp_payload(raw.text)
    return payload.get("competition", [])


def season_comp_map(competitions: list[dict[str, Any]], years: list[int]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    targets = set(years)
    for row in competitions:
        name = str(row.get("CompetitionName", ""))
        m = re.search(r"IPL\s+(\d{4})", name)
        if not m:
            continue
        y = int(m.group(1))
        if y in targets:
            out[y] = row
    missing = sorted(targets - set(out.keys()))
    if missing:
        raise RuntimeError(f"Missing competition metadata for years: {missing}")
    return out


def fetch_stat_file(stats_feed: str, comp_id: str, file_key: str) -> list[dict[str, Any]]:
    url = f"{stats_feed}/stats/{comp_id}-{file_key}.js"
    resp = requests.get(url, headers=UA, timeout=30)
    resp.raise_for_status()
    payload = parse_jsonp_payload(resp.text)
    # payload key generally matches file key
    if isinstance(payload, dict) and file_key in payload:
        return payload[file_key]
    if isinstance(payload, dict) and len(payload) == 1:
        return next(iter(payload.values()))
    raise RuntimeError(f"Unexpected payload shape for {url}")


def main() -> None:
    args = parse_args()
    years = sorted(set(args.years))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    comps = load_competitions()
    mapping = season_comp_map(comps, years)

    batting_rows: list[dict[str, Any]] = []
    bowling_rows: list[dict[str, Any]] = []

    for y in years:
        row = mapping[y]
        comp_id = str(row.get("statsCID") or row.get("CompetitionID"))
        stats_feed = str(row.get("statsFeed") or row.get("feedsource"))

        b = fetch_stat_file(stats_feed, comp_id, "toprunsscorers")
        w = fetch_stat_file(stats_feed, comp_id, "mostwickets")

        for r in b:
            batting_rows.append({"season": y, "stats_comp_id": comp_id, **r})
        for r in w:
            bowling_rows.append({"season": y, "stats_comp_id": comp_id, **r})

        print(f"[OK] {y}: batting={len(b)} bowling={len(w)}")

    batting_df = pd.DataFrame(batting_rows)
    bowling_df = pd.DataFrame(bowling_rows)

    bat_csv = out_dir / f"ipl_toprunsscorers_{years[0]}_{years[-1]}.csv"
    bowl_csv = out_dir / f"ipl_mostwickets_{years[0]}_{years[-1]}.csv"
    batting_df.to_csv(bat_csv, index=False)
    bowling_df.to_csv(bowl_csv, index=False)

    print(f"Saved batting: {bat_csv} ({len(batting_df)})")
    print(f"Saved bowling: {bowl_csv} ({len(bowling_df)})")


if __name__ == "__main__":
    main()
