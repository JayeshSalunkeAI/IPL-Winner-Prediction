#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


ROOT_DIR = Path(__file__).resolve().parents[2]
UA = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


@dataclass
class MatchLinks:
    match_id: str
    scorecard_url: str
    squads_url: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Update Phase 4.1 feature store from Cricbuzz scorecards")
    p.add_argument("--links-csv", default="data/raw/cricbuzz_match_links.csv")
    p.add_argument("--history-csv", default="production_model/data/phase41_scorecard_history.csv")
    p.add_argument("--match-features-csv", default="production_model/data/phase41_match_feature_store.csv")
    p.add_argument("--team-features-csv", default="production_model/data/phase41_team_feature_store.csv")
    p.add_argument(
        "--player-ratings-csv",
        default="phases/phase_4_1_redefine/data/player_performance_ratings_2020_2026.csv",
    )
    p.add_argument("--start-id", type=int, default=149618)
    p.add_argument("--end-id", type=int, default=0, help="0 means no upper bound")
    p.add_argument("--sleep", type=float, default=0.25)
    p.add_argument("--series-key", default="Indian Premier League 2026")
    p.add_argument("--force-refresh", action="store_true")
    return p.parse_args()


def normalize_team_name(name: str) -> str:
    text = str(name or "").strip()
    mapping = {
        "Rising Pune Supergiant": "Rising Pune Supergiants",
        "Delhi Daredevils": "Delhi Capitals",
        "Kings XI Punjab": "Punjab Kings",
    }
    return mapping.get(text, text)


def normalize_key(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def fetch_html(url: str, timeout: int = 20) -> str:
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r.text


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def parse_toss(text: str, team1: str, team2: str) -> tuple[str, str]:
    m = re.search(r"([A-Za-z .&-]+?) elected to (bat|field)", text, flags=re.I)
    if m:
        candidate = normalize_team_name(m.group(1).strip())
        if team1 and team1 in candidate:
            return team1, m.group(2).lower()
        if team2 and team2 in candidate:
            return team2, m.group(2).lower()
    return team1, "field"


def parse_winner(text: str, team1: str, team2: str) -> str:
    if team1 and re.search(rf"\b{re.escape(team1)}\b\s+won by", text, flags=re.I):
        return team1
    if team2 and re.search(rf"\b{re.escape(team2)}\b\s+won by", text, flags=re.I):
        return team2

    m = re.search(r"([A-Za-z .&-]+?) won by [A-Za-z0-9 .&-]+", text, flags=re.I)
    if not m:
        return ""
    candidate = normalize_team_name(m.group(1).strip())
    if team1 and team1 in candidate:
        return team1
    if team2 and team2 in candidate:
        return team2
    return ""


def parse_venue(soup: BeautifulSoup, text: str) -> str:
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        blob = script.get_text("", strip=True)
        if not blob:
            continue
        m = re.search(r'"location"\s*:\s*\{[^{}]*"name"\s*:\s*"([^"]+)"', blob)
        if m:
            return clean_text(m.group(1))

    m = re.search(r"Venue\s+([A-Za-z0-9,'&.\- ]{5,80}),\s*([A-Za-z .\-]{2,40})", text)
    if m:
        return clean_text(f"{m.group(1)}, {m.group(2)}")
    return "Unknown"


def parse_team_totals(text: str, team1: str, team2: str) -> dict[str, Optional[float]]:
    def one(team: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
        pattern = rf"{re.escape(team)}[^0-9]{{0,40}}(\d{{2,3}})-(\d{{1,2}})(?:\s*\((\d{{1,2}}(?:\.\d)?)\s*Ovs\))?"
        matches = list(re.finditer(pattern, text, flags=re.I))
        if not matches:
            return None, None, None
        # Prefer the last mention; scoreboard summary tends to appear later in text dump.
        m = matches[-1]
        runs = float(m.group(1))
        wkts = float(m.group(2))
        overs = float(m.group(3)) if m.group(3) else None
        return runs, wkts, overs

    t1_runs, t1_wkts, t1_overs = one(team1)
    t2_runs, t2_wkts, t2_overs = one(team2)
    return {
        "team1_runs": t1_runs,
        "team1_wkts": t1_wkts,
        "team1_overs": t1_overs,
        "team2_runs": t2_runs,
        "team2_wkts": t2_wkts,
        "team2_overs": t2_overs,
    }


def parse_powerplay_runs(text: str, toss_winner: str, toss_decision: str, team1: str, team2: str) -> tuple[Optional[float], Optional[float]]:
    vals = [float(v) for v in re.findall(r"Powerplays\s+Overs\s+Runs\s+Mandatory\s+0\.1\s*-\s*6\s*(\d+)", text, flags=re.I)]
    if not vals:
        return None, None
    # Pages can duplicate blocks. Keep first two unique values in order.
    dedup: list[float] = []
    for v in vals:
        if not dedup or dedup[-1] != v:
            dedup.append(v)
    first_pp = dedup[0] if len(dedup) >= 1 else None
    second_pp = dedup[1] if len(dedup) >= 2 else None

    if toss_decision == "field":
        batting_first = team2 if toss_winner == team1 else team1
    else:
        batting_first = toss_winner
    batting_second = team1 if batting_first == team2 else team2

    pp_map: dict[str, Optional[float]] = {batting_first: first_pp, batting_second: second_pp}
    return pp_map.get(team1), pp_map.get(team2)


def parse_match_date(text: str) -> str:
    m = re.search(r"([A-Za-z]+\s+\d{1,2},\s+\d{4})", text)
    return m.group(1) if m else ""


def parse_teams_from_h1(h1_text: str) -> tuple[str, str]:
    if " vs " not in h1_text:
        return "", ""
    left, right = h1_text.split(" vs ", 1)
    team2 = right.split(",")[0]
    return normalize_team_name(clean_text(left)), normalize_team_name(clean_text(team2))


def parse_squads(squads_html: str) -> tuple[list[str], list[str]]:
    soup = BeautifulSoup(squads_html, "lxml")
    anchors = soup.select("a[href*='/profiles/']")
    names: list[str] = []
    seen = set()
    for a in anchors:
        name = clean_text(a.get_text(" ", strip=True))
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        names.append(name)

    team1_xi = names[:11]
    team2_xi = names[11:22]
    while len(team1_xi) < 11:
        team1_xi.append(f"Unknown_Player_{len(team1_xi) + 1}")
    while len(team2_xi) < 11:
        team2_xi.append(f"Unknown_Player_{len(team2_xi) + 1}")
    return team1_xi, team2_xi


def load_links(path: Path, start_id: int, end_id: int) -> list[MatchLinks]:
    df = pd.read_csv(path)
    required = {"match_id", "scorecard_url", "squads_url"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in links CSV: {sorted(missing)}")

    df["match_id_num"] = pd.to_numeric(df["match_id"], errors="coerce")
    df = df.dropna(subset=["match_id_num"]).copy()
    df = df[df["match_id_num"] >= start_id]
    if end_id > 0:
        df = df[df["match_id_num"] <= end_id]
    df = df.sort_values("match_id_num")

    out: list[MatchLinks] = []
    for _, r in df.iterrows():
        out.append(
            MatchLinks(
                match_id=str(int(r["match_id_num"])),
                scorecard_url=str(r["scorecard_url"]),
                squads_url=str(r["squads_url"]),
            )
        )
    return out


def load_player_elo_map(path: Path) -> tuple[dict[str, float], float]:
    if not path.exists():
        return {}, 1000.0
    df = pd.read_csv(path)
    out = {
        normalize_key(r.get("player_name", "")): float(r.get("player_elo_like", 1000.0))
        for _, r in df.iterrows()
        if str(r.get("player_name", "")).strip()
    }
    default = float(np.median(list(out.values()))) if out else 1000.0
    return out, default


def scrape_match(match: MatchLinks, series_key: str) -> dict[str, Any]:
    score_html = fetch_html(match.scorecard_url)
    squads_html = fetch_html(match.squads_url)
    soup = BeautifulSoup(score_html, "lxml")
    text = clean_text(soup.get_text(" ", strip=True))

    h1 = soup.find("h1")
    h1_text = clean_text(h1.get_text(" ", strip=True)) if h1 else ""
    if series_key and series_key.lower() not in h1_text.lower():
        raise ValueError(f"skip_non_target_series: {h1_text}")
    team1, team2 = parse_teams_from_h1(h1_text)
    toss_winner, toss_decision = parse_toss(text, team1, team2)
    winner = parse_winner(text, team1, team2)
    venue = parse_venue(soup, text)
    match_date = parse_match_date(text)
    totals = parse_team_totals(text, team1, team2)
    team1_pp, team2_pp = parse_powerplay_runs(text, toss_winner, toss_decision, team1, team2)
    team1_xi, team2_xi = parse_squads(squads_html)

    if toss_decision == "field":
        batting_first = team2 if toss_winner == team1 else team1
    else:
        batting_first = toss_winner

    first_runs = totals["team1_runs"] if batting_first == team1 else totals["team2_runs"]

    return {
        "match_id": match.match_id,
        "match_name": h1_text,
        "match_date": match_date,
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "winner": winner,
        "team1_runs": totals["team1_runs"],
        "team1_wkts": totals["team1_wkts"],
        "team1_overs": totals["team1_overs"],
        "team2_runs": totals["team2_runs"],
        "team2_wkts": totals["team2_wkts"],
        "team2_overs": totals["team2_overs"],
        "team1_powerplay_runs": team1_pp,
        "team2_powerplay_runs": team2_pp,
        "first_innings_runs": first_runs,
        "team1_xi": json.dumps(team1_xi, ensure_ascii=False),
        "team2_xi": json.dumps(team2_xi, ensure_ascii=False),
        "scorecard_url": match.scorecard_url,
        "squads_url": match.squads_url,
        "scraped_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def safe_mean(values: list[float], default: float) -> float:
    vals = [float(v) for v in values if not pd.isna(v)]
    if not vals:
        return default
    return float(np.mean(vals))


def prior_team_matches(df: pd.DataFrame, team: str) -> pd.DataFrame:
    return df[(df["team1"] == team) | (df["team2"] == team)]


def team_runs_for(row: pd.Series, team: str) -> float:
    if row["team1"] == team:
        return float(row.get("team1_runs", np.nan))
    return float(row.get("team2_runs", np.nan))


def team_runs_against(row: pd.Series, team: str) -> float:
    if row["team1"] == team:
        return float(row.get("team2_runs", np.nan))
    return float(row.get("team1_runs", np.nan))


def team_wkts_taken(row: pd.Series, team: str) -> float:
    if row["team1"] == team:
        return float(row.get("team2_wkts", np.nan))
    return float(row.get("team1_wkts", np.nan))


def team_powerplay_rr(row: pd.Series, team: str) -> float:
    if row["team1"] == team:
        v = row.get("team1_powerplay_runs", np.nan)
    else:
        v = row.get("team2_powerplay_runs", np.nan)
    if pd.isna(v):
        overs = row.get("team1_overs", np.nan) if row["team1"] == team else row.get("team2_overs", np.nan)
        runs = row.get("team1_runs", np.nan) if row["team1"] == team else row.get("team2_runs", np.nan)
        if pd.isna(overs) or pd.isna(runs) or float(overs) <= 0:
            return np.nan
        return float(runs) / float(overs)
    return float(v) / 6.0


def team_death_rr(row: pd.Series, team: str) -> float:
    # Cricbuzz scorecard text does not expose a clean death-over block consistently.
    # Use innings RR proxy until a richer parser is introduced.
    overs = row.get("team1_overs", np.nan) if row["team1"] == team else row.get("team2_overs", np.nan)
    runs = row.get("team1_runs", np.nan) if row["team1"] == team else row.get("team2_runs", np.nan)
    if pd.isna(overs) or pd.isna(runs) or float(overs) <= 0:
        return np.nan
    return float(runs) / float(overs)


def parse_xi(cell: Any) -> list[str]:
    if isinstance(cell, list):
        out = [str(x).strip() for x in cell if str(x).strip()]
    else:
        text = str(cell or "").strip()
        if not text:
            out = []
        else:
            try:
                parsed = json.loads(text)
                out = [str(x).strip() for x in parsed if str(x).strip()] if isinstance(parsed, list) else []
            except Exception:
                out = [x.strip() for x in text.split("|") if x.strip()]
    out = out[:11]
    while len(out) < 11:
        out.append(f"Unknown_Player_{len(out) + 1}")
    return out


def player_elo_stats(players: list[str], elo_map: dict[str, float], default_elo: float) -> tuple[float, float, float]:
    vals = [elo_map.get(normalize_key(p), default_elo) for p in players]
    arr = np.array(vals, dtype=float)
    return float(arr.mean()), float(arr.max()), float(arr.min())


def compute_feature_rows(history_df: pd.DataFrame, elo_map: dict[str, float], default_elo: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if history_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = history_df.copy()
    df["match_id_num"] = pd.to_numeric(df["match_id"], errors="coerce")
    df = df.dropna(subset=["match_id_num"]).sort_values("match_id_num").reset_index(drop=True)
    completed = df[df["winner"].astype(str).str.strip() != ""].copy()

    out_rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        mid = float(row["match_id_num"])
        prior = completed[completed["match_id_num"] < mid].copy()

        t1 = normalize_team_name(str(row["team1"]))
        t2 = normalize_team_name(str(row["team2"]))
        venue = str(row.get("venue", "Unknown"))

        def recent_form(team: str) -> float:
            tm = prior_team_matches(prior, team).tail(5)
            if tm.empty:
                return 0.5
            return float((tm["winner"] == team).mean())

        t1m = prior_team_matches(prior, t1).tail(5)
        t2m = prior_team_matches(prior, t2).tail(5)

        h2h = prior[
            ((prior["team1"] == t1) & (prior["team2"] == t2))
            | ((prior["team1"] == t2) & (prior["team2"] == t1))
        ]
        h2h_t1 = float((h2h["winner"] == t1).mean()) if not h2h.empty else 0.5

        venue_prior = prior[prior["venue"] == venue]
        venue_t1 = venue_prior[(venue_prior["team1"] == t1) | (venue_prior["team2"] == t1)]
        venue_t2 = venue_prior[(venue_prior["team1"] == t2) | (venue_prior["team2"] == t2)]

        venue_t1_wr = float((venue_t1["winner"] == t1).mean()) if not venue_t1.empty else 0.5
        venue_t2_wr = float((venue_t2["winner"] == t2).mean()) if not venue_t2.empty else 0.5
        venue_first_avg = safe_mean(venue_prior["first_innings_runs"].tolist(), 170.0)

        chase_vals: list[bool] = []
        for _, vr in venue_prior.iterrows():
            toss_winner = str(vr.get("toss_winner", "")).strip()
            toss_decision = str(vr.get("toss_decision", "")).strip().lower()
            winner = str(vr.get("winner", "")).strip()
            if not toss_winner or toss_decision not in {"bat", "field"} or not winner:
                continue
            a = str(vr.get("team1", "")).strip()
            b = str(vr.get("team2", "")).strip()
            if toss_decision == "field":
                chasing_team = toss_winner
            else:
                chasing_team = b if toss_winner == a else a
            chase_vals.append(winner == chasing_team)
        venue_chase_wr = float(np.mean(chase_vals)) if chase_vals else 0.5

        t1_runs_for = safe_mean([team_runs_for(r, t1) for _, r in t1m.iterrows()], 160.0)
        t2_runs_for = safe_mean([team_runs_for(r, t2) for _, r in t2m.iterrows()], 160.0)
        t1_runs_against = safe_mean([team_runs_against(r, t1) for _, r in t1m.iterrows()], 160.0)
        t2_runs_against = safe_mean([team_runs_against(r, t2) for _, r in t2m.iterrows()], 160.0)
        t1_wkts_taken = safe_mean([team_wkts_taken(r, t1) for _, r in t1m.iterrows()], 7.0)
        t2_wkts_taken = safe_mean([team_wkts_taken(r, t2) for _, r in t2m.iterrows()], 7.0)
        t1_pp_rr = safe_mean([team_powerplay_rr(r, t1) for _, r in t1m.iterrows()], 8.0)
        t2_pp_rr = safe_mean([team_powerplay_rr(r, t2) for _, r in t2m.iterrows()], 8.0)
        t1_death_rr = safe_mean([team_death_rr(r, t1) for _, r in t1m.iterrows()], 10.0)
        t2_death_rr = safe_mean([team_death_rr(r, t2) for _, r in t2m.iterrows()], 10.0)

        t1_xi = parse_xi(row.get("team1_xi"))
        t2_xi = parse_xi(row.get("team2_xi"))
        t1_elo_avg, t1_elo_max, t1_elo_min = player_elo_stats(t1_xi, elo_map, default_elo)
        t2_elo_avg, t2_elo_max, t2_elo_min = player_elo_stats(t2_xi, elo_map, default_elo)

        out_rows.append(
            {
                "Match_ID": int(mid),
                "Date": row.get("match_date", ""),
                "Team1": t1,
                "Team2": t2,
                "Toss_Winner": normalize_team_name(str(row.get("toss_winner", "") or t1)),
                "Toss_Decision": str(row.get("toss_decision", "field") or "field").lower(),
                "team1_form_winrate_5": recent_form(t1),
                "team2_form_winrate_5": recent_form(t2),
                "venue_chase_winrate_prior": venue_chase_wr,
                "venue_score_prior": venue_first_avg,
                "h2h_team1_winrate_prior": h2h_t1,
                "venue_team1_winrate_prior": venue_t1_wr,
                "venue_team2_winrate_prior": venue_t2_wr,
                "venue_avg_first_innings_runs_prior": venue_first_avg,
                "team1_recent_runs_for_5": t1_runs_for,
                "team2_recent_runs_for_5": t2_runs_for,
                "team1_recent_runs_against_5": t1_runs_against,
                "team2_recent_runs_against_5": t2_runs_against,
                "team1_recent_wkts_taken_5": t1_wkts_taken,
                "team2_recent_wkts_taken_5": t2_wkts_taken,
                "team1_recent_powerplay_rr_5": t1_pp_rr,
                "team2_recent_powerplay_rr_5": t2_pp_rr,
                "team1_recent_death_rr_5": t1_death_rr,
                "team2_recent_death_rr_5": t2_death_rr,
                "team1_player_elo_avg_prior": t1_elo_avg,
                "team2_player_elo_avg_prior": t2_elo_avg,
                "team1_player_elo_max_prior": t1_elo_max,
                "team2_player_elo_max_prior": t2_elo_max,
                "team1_player_elo_min_prior": t1_elo_min,
                "team2_player_elo_min_prior": t2_elo_min,
                "player_elo_gap_prior": t1_elo_avg - t2_elo_avg,
                "Match_Winner": normalize_team_name(str(row.get("winner", ""))),
                **{f"Team1_Player_{i+1}": t1_xi[i] for i in range(11)},
                **{f"Team2_Player_{i+1}": t2_xi[i] for i in range(11)},
                "venue": venue,
            }
        )

    match_features = pd.DataFrame(out_rows)

    team_rows: list[dict[str, Any]] = []
    teams = sorted(set(df["team1"].tolist() + df["team2"].tolist()))
    now = datetime.now(timezone.utc).isoformat()
    for team in teams:
        tm = prior_team_matches(completed, team).tail(5)
        if tm.empty:
            team_rows.append(
                {
                    "team": team,
                    "recent_form_winrate_5": 0.5,
                    "recent_runs_for_5": 160.0,
                    "recent_runs_against_5": 160.0,
                    "recent_wkts_taken_5": 7.0,
                    "recent_powerplay_rr_5": 8.0,
                    "recent_death_rr_5": 10.0,
                    "as_of_match_id": np.nan,
                    "updated_at_utc": now,
                }
            )
            continue

        team_rows.append(
            {
                "team": team,
                "recent_form_winrate_5": float((tm["winner"] == team).mean()),
                "recent_runs_for_5": safe_mean([team_runs_for(r, team) for _, r in tm.iterrows()], 160.0),
                "recent_runs_against_5": safe_mean([team_runs_against(r, team) for _, r in tm.iterrows()], 160.0),
                "recent_wkts_taken_5": safe_mean([team_wkts_taken(r, team) for _, r in tm.iterrows()], 7.0),
                "recent_powerplay_rr_5": safe_mean([team_powerplay_rr(r, team) for _, r in tm.iterrows()], 8.0),
                "recent_death_rr_5": safe_mean([team_death_rr(r, team) for _, r in tm.iterrows()], 10.0),
                "as_of_match_id": float(tm["match_id_num"].iloc[-1]),
                "updated_at_utc": now,
            }
        )

    team_features = pd.DataFrame(team_rows).sort_values("team").reset_index(drop=True)
    return match_features, team_features


def main() -> int:
    args = parse_args()
    links_path = ROOT_DIR / args.links_csv
    history_path = ROOT_DIR / args.history_csv
    match_features_path = ROOT_DIR / args.match_features_csv
    team_features_path = ROOT_DIR / args.team_features_csv
    ratings_path = ROOT_DIR / args.player_ratings_csv

    if not links_path.exists():
        raise FileNotFoundError(f"Links CSV not found: {links_path}")

    links = load_links(links_path, args.start_id, args.end_id)

    if history_path.exists() and history_path.stat().st_size > 0:
        history_df = pd.read_csv(history_path)
    else:
        history_df = pd.DataFrame()

    existing_ids = set(history_df["match_id"].astype(str).tolist()) if not history_df.empty else set()

    new_rows: list[dict[str, Any]] = []
    for item in links:
        if (not args.force_refresh) and item.match_id in existing_ids:
            continue
        try:
            row = scrape_match(item, args.series_key)
            new_rows.append(row)
            print(f"[OK] {item.match_id} | {row.get('team1')} vs {row.get('team2')} | winner={row.get('winner')}")
        except Exception as exc:
            if "skip_non_target_series" not in str(exc):
                print(f"[ERR] {item.match_id}: {exc}")
        time.sleep(max(args.sleep, 0.0))

    if new_rows:
        upd = pd.DataFrame(new_rows)
        if history_df.empty:
            history_df = upd.copy()
        else:
            history_df = pd.concat([history_df, upd], ignore_index=True)
        history_df = history_df.drop_duplicates(subset=["match_id"], keep="last")

    if history_df.empty:
        print("No history available; nothing to compute.")
        return 0

    if "match_name" in history_df.columns and args.series_key:
        history_df = history_df[
            history_df["match_name"].astype(str).str.contains(args.series_key, case=False, na=False)
        ].copy()

    history_df["match_id_num"] = pd.to_numeric(history_df["match_id"], errors="coerce")
    history_df = history_df.dropna(subset=["match_id_num"]).sort_values("match_id_num").drop(columns=["match_id_num"])

    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(history_path, index=False)

    elo_map, default_elo = load_player_elo_map(ratings_path)
    match_features, team_features = compute_feature_rows(history_df, elo_map, default_elo)

    match_features_path.parent.mkdir(parents=True, exist_ok=True)
    team_features_path.parent.mkdir(parents=True, exist_ok=True)
    match_features.to_csv(match_features_path, index=False)
    team_features.to_csv(team_features_path, index=False)

    print(f"Saved scorecard history: {history_path} (rows={len(history_df)})")
    print(f"Saved match feature store: {match_features_path} (rows={len(match_features)})")
    print(f"Saved team feature store: {team_features_path} (rows={len(team_features)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
