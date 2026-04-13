from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build cleaned IPL player performance ratings")
    p.add_argument("--batting-csv", required=True)
    p.add_argument("--bowling-csv", required=True)
    p.add_argument("--out-csv", default="phases/phase_4_1_redefine/data/player_performance_ratings_2020_2026.csv")
    return p.parse_args()


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def season_weight(year: int) -> float:
    # Recency tilt for player performance carry-forward.
    mapping = {2020: 1.0, 2021: 1.2, 2023: 1.5, 2024: 1.8, 2025: 2.1, 2026: 2.5}
    return mapping.get(int(year), 1.0)


def main() -> None:
    args = parse_args()
    bat = pd.read_csv(args.batting_csv)
    bowl = pd.read_csv(args.bowling_csv)

    bat["player_name"] = bat["StrikerName"].astype(str).str.strip()
    bat["player_key"] = bat["player_name"].map(normalize_name)
    bat["season"] = to_num(bat["season"]).astype(int)
    bat["w"] = bat["season"].map(season_weight)

    bat_runs = to_num(bat["TotalRuns"])
    bat_sr = to_num(bat["StrikeRate"])
    bat_inns = to_num(bat["Innings"]).replace(0, 1)
    bat_fours = to_num(bat["Fours"])
    bat_sixes = to_num(bat["Sixes"])
    bat_50 = to_num(bat["FiftyPlusRuns"])
    bat_100 = to_num(bat["Centuries"])
    bat["batting_score_raw"] = (
        (bat_runs / bat_inns) * 2.0
        + (bat_sr / 20.0)
        + ((bat_fours + bat_sixes) / 5.0)
        + (bat_50 * 1.5)
        + (bat_100 * 4.0)
    )

    bowl["player_name"] = bowl["BowlerName"].astype(str).str.strip()
    bowl["player_key"] = bowl["player_name"].map(normalize_name)
    bowl["season"] = to_num(bowl["season"]).astype(int)
    bowl["w"] = bowl["season"].map(season_weight)

    bowl_wkts = to_num(bowl["Wickets"])
    bowl_inns = to_num(bowl["Innings"]).replace(0, 1)
    bowl_dot = to_num(bowl["DotBallPercent"])
    bowl_eco = to_num(bowl["EconomyRate"])
    bowl_maid_w = to_num(bowl["MaidenWickets"])
    bowl["bowling_score_raw"] = (
        (bowl_wkts / bowl_inns) * 25.0
        + (bowl_dot / 2.0)
        + np.maximum(0.0, (9.0 - bowl_eco)) * 8.0
        + (bowl_maid_w * 2.0)
    )

    bat_agg = (
        bat.assign(weighted_score=bat["batting_score_raw"] * bat["w"])
        .groupby(["player_key", "player_name"], as_index=False)
        .agg(weighted_score=("weighted_score", "sum"), weight_sum=("w", "sum"))
    )
    bat_agg["batting_score"] = bat_agg["weighted_score"] / bat_agg["weight_sum"].replace(0.0, 1.0)

    bowl_agg = (
        bowl.assign(weighted_score=bowl["bowling_score_raw"] * bowl["w"])
        .groupby(["player_key", "player_name"], as_index=False)
        .agg(weighted_score=("weighted_score", "sum"), weight_sum=("w", "sum"))
    )
    bowl_agg["bowling_score"] = bowl_agg["weighted_score"] / bowl_agg["weight_sum"].replace(0.0, 1.0)

    merged = pd.merge(
        bat_agg[["player_key", "player_name", "batting_score"]],
        bowl_agg[["player_key", "bowling_score"]],
        on="player_key",
        how="outer",
    )
    merged["player_name"] = merged["player_name"].fillna(merged["player_key"])
    merged["batting_score"] = merged["batting_score"].fillna(0.0)
    merged["bowling_score"] = merged["bowling_score"].fillna(0.0)

    merged["performance_score"] = 0.60 * merged["batting_score"] + 0.40 * merged["bowling_score"]

    mean = merged["performance_score"].mean()
    std = merged["performance_score"].std(ddof=0)
    std = std if std > 1e-9 else 1.0
    merged["player_elo_like"] = (1000.0 + 150.0 * ((merged["performance_score"] - mean) / std)).round(3)

    out = merged.sort_values("player_elo_like", ascending=False).reset_index(drop=True)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Saved ratings: {out_path} ({len(out)})")
    print(out[["player_name", "player_elo_like"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
