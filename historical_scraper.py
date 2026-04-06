#!/usr/bin/env python3
"""Historical Cricbuzz scraper.

Reads match link CSV from build_match_ids.py and writes a flat CSV with
match-level metadata plus XI/substitute lists.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

UA = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def fetch_html(url: str, timeout: int = 25) -> str:
    resp = requests.get(url, headers=UA, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def parse_winner(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    m = re.search(r"([A-Za-z .&-]+?) won by ([A-Za-z0-9 .&-]+)", text, flags=re.I)
    if not m:
        return None, None, None
    result = clean_text(m.group(0))
    winner = clean_text(m.group(1))
    margin = clean_text(m.group(2))
    return winner, margin, result


def parse_scorecard(html: str) -> Dict[str, Optional[str]]:
    soup = BeautifulSoup(html, "lxml")
    text = clean_text(soup.get_text(" ", strip=True))

    out: Dict[str, Optional[str]] = {
        "match_title": None,
        "team1": None,
        "team2": None,
        "venue": None,
        "match_date": None,
        "result_text": None,
        "winner": None,
        "win_margin": None,
        "toss_text": None,
    }

    h1 = soup.find("h1")
    if h1:
        out["match_title"] = clean_text(h1.get_text(" ", strip=True))

    if out["match_title"] and " vs " in out["match_title"]:
        left, right = out["match_title"].split(" vs ", 1)
        out["team1"] = clean_text(left)
        out["team2"] = clean_text(right.split(",")[0])

    winner, margin, result = parse_winner(text)
    out["winner"] = winner
    out["win_margin"] = margin
    out["result_text"] = result

    toss_m = re.search(r"([A-Za-z .&-]+ elected to (bat|field) first)", text, flags=re.I)
    if toss_m:
        out["toss_text"] = clean_text(toss_m.group(1))

    date_m = re.search(r"([A-Za-z]+ \d{1,2}, \d{4})", text)
    if date_m:
        out["match_date"] = clean_text(date_m.group(1))

    venue_m = re.search(r"at ([A-Za-z0-9,'&.\- ]+), [A-Za-z ]+", text)
    if venue_m:
        out["venue"] = clean_text(venue_m.group(1))

    return out


def parse_squads(html: str) -> Dict[str, object]:
    soup = BeautifulSoup(html, "lxml")

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    team1_name = None
    team2_name = None
    m = re.search(r"(.+?) vs (.+?)\s", title)
    if m:
        team1_name = clean_text(m.group(1))
        team2_name = clean_text(m.group(2))

    # Cricbuzz squad pages include profiles links for players.
    anchors = soup.select("a[href*='/profiles/']")
    names: List[str] = []
    seen = set()
    for a in anchors:
        name = clean_text(a.get_text(" ", strip=True))
        if not name:
            continue
        k = name.lower()
        if k in seen:
            continue
        seen.add(k)
        names.append(name)

    team1_xi = names[:11]
    team2_xi = names[11:22]
    remaining = names[22:]
    split = len(remaining) // 2
    team1_subs = remaining[:split]
    team2_subs = remaining[split:]

    return {
        "team1_name": team1_name,
        "team2_name": team2_name,
        "team1_xi": team1_xi,
        "team2_xi": team2_xi,
        "team1_subs": team1_subs,
        "team2_subs": team2_subs,
    }


def scrape_one(match_id: str, scorecard_url: str, squads_url: str) -> Dict[str, object]:
    score_html = fetch_html(scorecard_url)
    squad_html = fetch_html(squads_url)

    score = parse_scorecard(score_html)
    squads = parse_squads(squad_html)

    row: Dict[str, object] = {
        "match_id": str(match_id),
        "scorecard_url": scorecard_url,
        "squads_url": squads_url,
        "scraped_at_utc": datetime.now(timezone.utc).isoformat(),
        **score,
        **squads,
    }

    for col in ["team1_xi", "team2_xi", "team1_subs", "team2_subs"]:
        row[col] = json.dumps(row[col], ensure_ascii=False)

    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch scrape historical Cricbuzz matches")
    parser.add_argument(
        "--input",
        default="data/raw/cricbuzz_match_links.csv",
        help="Input CSV with match_id, scorecard_url, squads_url",
    )
    parser.add_argument(
        "--output",
        default="data/raw/cricbuzz_historical_matches.csv",
        help="Output CSV path",
    )
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep between requests")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"Input not found: {in_path}")
        return 2

    df = pd.read_csv(in_path)
    required_cols = {"match_id", "scorecard_url", "squads_url"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Missing required columns: {sorted(missing)}")
        return 2

    rows: List[Dict[str, object]] = []
    errors: List[Dict[str, str]] = []

    for _, r in df.iterrows():
        mid = str(r["match_id"])
        score_url = str(r["scorecard_url"])
        squads_url = str(r["squads_url"])
        try:
            rows.append(scrape_one(mid, score_url, squads_url))
            print(f"[OK] {mid}")
        except Exception as exc:
            errors.append({"match_id": mid, "error": str(exc)})
            print(f"[ERR] {mid}: {exc}")
        time.sleep(max(args.sleep, 0.0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved {len(rows)} rows to {out_path}")

    if errors:
        err_path = out_path.with_name(out_path.stem + "_errors.csv")
        pd.DataFrame(errors).to_csv(err_path, index=False)
        print(f"Saved {len(errors)} errors to {err_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
