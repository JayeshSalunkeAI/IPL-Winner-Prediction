#!/usr/bin/env python3
"""Build Cricbuzz match link inventory for IPL matches.

This script crawls Cricbuzz pages and extracts match ids plus scorecard/squad links,
then saves a deduplicated CSV for downstream scrapers.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List

import requests

BASE = "https://www.cricbuzz.com"
DEFAULT_URLS = [
    "https://www.cricbuzz.com/cricket-series/9241/indian-premier-league-2026/matches",
    "https://www.cricbuzz.com/cricket-match/live-scores",
    "https://www.cricbuzz.com/cricket-match/live-scores/recent-matches",
]
UA = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

SCORECARD_RE = re.compile(r"/live-cricket-scorecard/(\d+)/(?:[a-z0-9\-]+)")
SQUADS_RE = re.compile(r"/cricket-match-squads/(\d+)/(?:[a-z0-9\-]+)")


def fetch_html(url: str, timeout: int = 25) -> str:
    resp = requests.get(url, headers=UA, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def extract_links(html: str) -> List[str]:
    links = re.findall(r'href="([^"]+)"', html)
    out: List[str] = []
    for link in links:
        if link.startswith("/"):
            out.append(BASE + link)
        elif link.startswith("http"):
            out.append(link)
    return out


def extract_match_entries(text: str) -> Dict[str, Dict[str, str]]:
    entries: Dict[str, Dict[str, str]] = {}

    for match in SCORECARD_RE.finditer(text):
        match_id = match.group(1)
        path = match.group(0)
        entries.setdefault(match_id, {})["scorecard_url"] = BASE + path

    for match in SQUADS_RE.finditer(text):
        match_id = match.group(1)
        path = match.group(0)
        entries.setdefault(match_id, {})["squads_url"] = BASE + path

    # Add canonical fallback links if one of the two was not found in source pages.
    for mid, payload in entries.items():
        payload.setdefault("scorecard_url", f"{BASE}/live-cricket-scorecard/{mid}/match")
        payload.setdefault("squads_url", f"{BASE}/cricket-match-squads/{mid}/match")

    return entries


def crawl(urls: Iterable[str], max_depth: int = 1) -> Dict[str, Dict[str, str]]:
    seen = set()
    to_visit = list(urls)
    all_entries: Dict[str, Dict[str, str]] = {}

    for _ in range(max_depth + 1):
        if not to_visit:
            break
        next_level: List[str] = []

        for url in to_visit:
            if url in seen:
                continue
            seen.add(url)
            try:
                html = fetch_html(url)
            except Exception:
                continue

            entries = extract_match_entries(html)
            for mid, payload in entries.items():
                if mid not in all_entries:
                    all_entries[mid] = payload
                else:
                    all_entries[mid].update(payload)

            for link in extract_links(html):
                if "cricbuzz.com" not in link:
                    continue
                if "/live-cricket-scorecard/" in link or "/cricket-match-squads/" in link:
                    continue
                if "indian-premier-league-2026" in link or "/live-scores" in link:
                    if link not in seen:
                        next_level.append(link)

        to_visit = next_level

    return all_entries


def save_csv(entries: Dict[str, Dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for mid in sorted(entries, key=lambda x: int(x)):
        row = {
            "match_id": mid,
            "scorecard_url": entries[mid].get("scorecard_url", ""),
            "squads_url": entries[mid].get("squads_url", ""),
        }
        rows.append(row)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["match_id", "scorecard_url", "squads_url"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build match-id CSV from Cricbuzz pages")
    parser.add_argument(
        "--urls",
        nargs="*",
        default=DEFAULT_URLS,
        help="Seed URLs to crawl",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=1,
        help="Crawl depth from seed URLs",
    )
    parser.add_argument(
        "--output",
        default="data/raw/cricbuzz_match_links.csv",
        help="Output CSV path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entries = crawl(args.urls, max_depth=max(0, args.max_depth))
    output_csv = Path(args.output)
    save_csv(entries, output_csv)
    print(f"Saved {len(entries)} match links to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
