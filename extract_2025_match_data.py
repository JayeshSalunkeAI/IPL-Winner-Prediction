from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


TEAM_RENAME_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}

DEFUNCT_TEAMS = {
    "Deccan Chargers",
    "Gujarat Lions",
    "Kochi Tuskers Kerala",
    "Pune Warriors",
    "Rising Pune Supergiants",
}


def _safe_match_id(path: Path) -> int | None:
    try:
        return int(path.stem)
    except ValueError:
        return None


def _load_match_metadata(yaml_path: Path) -> dict[str, Any]:
    # Match-level fields we need are above "innings" in cricsheet YAML files.
    # Parsing only this prefix is significantly faster than loading full deliveries.
    header_lines: list[str] = []
    with yaml_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("innings:"):
                break
            header_lines.append(line)

    header_text = "".join(header_lines)
    loaded = yaml.safe_load(header_text)
    return loaded if isinstance(loaded, dict) else {}


def _extract_base_records(yaml_dir: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for yaml_path in sorted(yaml_dir.glob("*.yaml"), key=lambda p: (p.stem.isdigit(), p.stem)):
        match = _load_match_metadata(yaml_path)

        info = (match or {}).get("info", {})
        if info.get("competition") != "IPL":
            continue

        dates = info.get("dates") or []
        if not dates:
            continue

        teams = info.get("teams") or []
        if len(teams) < 2:
            continue

        team1, team2 = teams[0], teams[1]
        outcome = info.get("outcome") or {}
        winner = outcome.get("winner") or "Draw/No Result"
        toss = info.get("toss") or {}

        match_id = _safe_match_id(yaml_path)
        if match_id is None:
            continue

        records.append(
            {
                "Match_ID": match_id,
                "Date": pd.to_datetime(dates[0]),
                "Teams": f"{team1} vs {team2}",
                "Team1": team1,
                "Team2": team2,
                "Venue": info.get("venue", "Unknown"),
                "Toss_Winner": toss.get("winner"),
                "Toss_Decision": (toss.get("decision") or "").strip().lower(),
                "Match_Winner": winner,
            }
        )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df

    return df.sort_values(["Date", "Match_ID"]).reset_index(drop=True)


def _compute_form_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    long = pd.concat(
        [
            df[["Match_ID", "Date", "Team1", "Team2", "Match_Winner"]]
            .rename(columns={"Team1": "Team", "Team2": "Opp"}),
            df[["Match_ID", "Date", "Team2", "Team1", "Match_Winner"]]
            .rename(columns={"Team2": "Team", "Team1": "Opp"}),
        ],
        ignore_index=True,
    )

    long["win"] = (long["Match_Winner"] == long["Team"]).astype(int)
    long[f"form_winrate_{window}"] = (
        long.groupby("Team")["win"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )

    t1 = long.merge(df[["Match_ID", "Team1"]], on="Match_ID")
    t1 = t1[t1["Team"] == t1["Team1"]][["Match_ID", f"form_winrate_{window}"]].rename(
        columns={f"form_winrate_{window}": f"team1_form_winrate_{window}"}
    )

    t2 = long.merge(df[["Match_ID", "Team2"]], on="Match_ID")
    t2 = t2[t2["Team"] == t2["Team2"]][["Match_ID", f"form_winrate_{window}"]].rename(
        columns={f"form_winrate_{window}": f"team2_form_winrate_{window}"}
    )

    return df.merge(t1, on="Match_ID", how="left").merge(t2, on="Match_ID", how="left")


def _other(team: str, team1: str, team2: str) -> str:
    return team2 if team == team1 else team1


def _compute_venue_features(df: pd.DataFrame) -> pd.DataFrame:
    venue_games: dict[str, int] = {}
    venue_chase_wins: dict[str, int] = {}
    alpha = 1

    venue_chase_rate_prior: list[float] = []
    venue_score_prior: list[float] = []

    bat_first: list[str | float] = []
    bat_second: list[str | float] = []

    for _, row in df.iterrows():
        t1 = row["Team1"]
        t2 = row["Team2"]
        toss_winner = row["Toss_Winner"]
        toss_decision = row["Toss_Decision"]

        if isinstance(toss_winner, str) and toss_decision == "bat":
            bf = toss_winner
        elif isinstance(toss_winner, str) and toss_decision == "field":
            bf = _other(toss_winner, t1, t2)
        else:
            bf = np.nan

        if isinstance(bf, str):
            bs = _other(bf, t1, t2)
        else:
            bs = np.nan

        bat_first.append(bf)
        bat_second.append(bs)

        venue = row["Venue"]
        games = venue_games.get(venue, 0)
        chase_wins = venue_chase_wins.get(venue, 0)

        chase_rate = (chase_wins + alpha) / (games + 2 * alpha) if games > 0 else 0.5
        score = 2 * chase_rate - 1

        venue_chase_rate_prior.append(chase_rate)
        venue_score_prior.append(score)

        if isinstance(bs, str):
            venue_games[venue] = games + 1
            if row["Match_Winner"] == bs:
                venue_chase_wins[venue] = chase_wins + 1

    df = df.copy()
    df["Bat_First"] = bat_first
    df["Bat_Second"] = bat_second
    df["venue_chase_winrate_prior"] = venue_chase_rate_prior
    df["venue_score_prior"] = venue_score_prior
    return df


def _normalize_teams(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for old, new in TEAM_RENAME_MAP.items():
        df["Teams"] = df["Teams"].str.replace(old, new, regex=False)

    for col in ["Team1", "Team2", "Toss_Winner", "Match_Winner", "Bat_First", "Bat_Second"]:
        if col in df.columns:
            df[col] = df[col].replace(TEAM_RENAME_MAP)

    split_teams = df["Teams"].str.split(" vs ", n=1, expand=True)
    df["Team1"] = split_teams[0].str.strip()
    df["Team2"] = split_teams[1].str.strip()

    return df[~(df["Team1"].isin(DEFUNCT_TEAMS) | df["Team2"].isin(DEFUNCT_TEAMS))].copy()


def build_dataset(yaml_dir: Path, target_year: int) -> pd.DataFrame:
    base = _extract_base_records(yaml_dir)
    if base.empty:
        return base

    with_form = _compute_form_features(base, window=5)
    with_venue = _compute_venue_features(with_form)
    normalized = _normalize_teams(with_venue)

    final_cols = [
        "Match_ID",
        "Date",
        "Teams",
        "Team1",
        "Team2",
        "Toss_Winner",
        "Toss_Decision",
        "team1_form_winrate_5",
        "team2_form_winrate_5",
        "venue_chase_winrate_prior",
        "venue_score_prior",
        "Match_Winner",
    ]

    out = normalized.loc[normalized["Date"].dt.year == target_year, final_cols].copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    return out.sort_values(["Date", "Match_ID"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract IPL match data from YAML files and create model-ready CSV "
            "with the IPL_Winner_Model_Dataset schema."
        )
    )
    parser.add_argument(
        "--yaml-dir",
        default="ipl_male",
        help="Directory containing YAML match files (default: ipl_male)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Target year to export (default: 2025)",
    )
    parser.add_argument(
        "--output",
        default="IPL_2025_Winner_Model_Dataset.csv",
        help="Output CSV path (default: IPL_2025_Winner_Model_Dataset.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    yaml_dir = Path(args.yaml_dir)

    if not yaml_dir.exists() or not yaml_dir.is_dir():
        raise FileNotFoundError(f"YAML directory not found: {yaml_dir}")

    dataset = build_dataset(yaml_dir=yaml_dir, target_year=args.year)
    dataset.to_csv(args.output, index=False)
    print(f"Saved {len(dataset)} rows to {args.output}")


if __name__ == "__main__":
    main()
