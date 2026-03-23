from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]


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


def _norm_team(team: str | None) -> str | None:
    if not isinstance(team, str):
        return team
    return TEAM_RENAME_MAP.get(team, team)


def _other(team: str, team1: str, team2: str) -> str:
    return team2 if team == team1 else team1


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _rolling_mean(values: list[float], n: int = 5, default: float = 0.0) -> float:
    if not values:
        return default
    window = values[-n:]
    return float(np.mean(window))


def _pad_players(players: list[str], slots: int) -> list[str]:
    players = [p for p in players if isinstance(p, str) and p.strip()]
    players = players[:slots]
    if len(players) < slots:
        players.extend(["Unknown_Player"] * (slots - len(players)))
    return players


def _parse_innings_stats(match: dict[str, Any]) -> tuple[dict[str, dict[str, float]], list[str]]:
    innings = (match.get("innings") or [])
    stats: dict[str, dict[str, float]] = defaultdict(lambda: {
        "runs": 0.0,
        "wkts_lost": 0.0,
        "powerplay_runs": 0.0,
        "death_runs": 0.0,
        "balls": 0.0,
    })
    batting_order: list[str] = []

    for inn in innings:
        if not isinstance(inn, dict) or not inn:
            continue
        _, inn_data = next(iter(inn.items()))
        if not isinstance(inn_data, dict):
            continue

        batting_team = _norm_team(inn_data.get("team"))
        if not isinstance(batting_team, str):
            continue
        batting_order.append(batting_team)

        deliveries = inn_data.get("deliveries") or []
        for ball in deliveries:
            if not isinstance(ball, dict) or not ball:
                continue
            ball_key, ball_data = next(iter(ball.items()))
            if not isinstance(ball_data, dict):
                continue

            over = int(_to_float(ball_key, 0.0))
            runs_total = _to_float(((ball_data.get("runs") or {}).get("total", 0)), 0.0)

            stats[batting_team]["runs"] += runs_total
            stats[batting_team]["balls"] += 1.0

            if over < 6:
                stats[batting_team]["powerplay_runs"] += runs_total
            if over >= 16:
                stats[batting_team]["death_runs"] += runs_total

            if "wicket" in ball_data:
                stats[batting_team]["wkts_lost"] += 1.0

    return dict(stats), batting_order


def build_phase4_dataset(yaml_dir: Path, output_csv: Path, player_slots: int = 11) -> pd.DataFrame:
    matches: list[dict[str, Any]] = []

    team_wins_hist: dict[str, list[float]] = defaultdict(list)
    team_runs_for_hist: dict[str, list[float]] = defaultdict(list)
    team_runs_against_hist: dict[str, list[float]] = defaultdict(list)
    team_wkts_taken_hist: dict[str, list[float]] = defaultdict(list)
    team_powerplay_rr_hist: dict[str, list[float]] = defaultdict(list)
    team_death_rr_hist: dict[str, list[float]] = defaultdict(list)

    h2h_games: dict[tuple[str, str], int] = defaultdict(int)
    h2h_team1_wins: dict[tuple[str, str], int] = defaultdict(int)

    venue_games: dict[str, int] = defaultdict(int)
    venue_chase_wins: dict[str, int] = defaultdict(int)
    venue_first_innings_runs: dict[str, list[float]] = defaultdict(list)

    venue_team_games: dict[tuple[str, str], int] = defaultdict(int)
    venue_team_wins: dict[tuple[str, str], int] = defaultdict(int)

    player_elo: dict[str, float] = {}
    player_matches: dict[str, int] = defaultdict(int)
    base_elo = 1000.0
    k_factor = 20.0

    yaml_paths = sorted(yaml_dir.glob("*.yaml"), key=lambda p: (p.stem.isdigit(), p.stem))

    for path in yaml_paths:
        try:
            with path.open("r", encoding="utf-8") as f:
                match = yaml.safe_load(f) or {}
        except (yaml.YAMLError, Exception):
            continue  # Skip corrupted YAML files

        info = match.get("info") or {}
        if info.get("competition") != "IPL":
            continue

        teams = info.get("teams") or []
        if len(teams) < 2:
            continue

        team1 = _norm_team(teams[0])
        team2 = _norm_team(teams[1])
        if not isinstance(team1, str) or not isinstance(team2, str):
            continue
        if team1 in DEFUNCT_TEAMS or team2 in DEFUNCT_TEAMS:
            continue

        dates = info.get("dates") or []
        if not dates:
            continue

        date = pd.to_datetime(dates[0], errors="coerce")
        if pd.isna(date):
            continue

        toss = info.get("toss") or {}
        toss_winner = _norm_team(toss.get("winner"))
        toss_decision = (toss.get("decision") or "").strip().lower()

        outcome = info.get("outcome") or {}
        winner = _norm_team(outcome.get("winner")) or "Draw/No Result"

        match_id = _safe_match_id(path)
        if match_id is None:
            continue

        venue = str(info.get("venue", "Unknown"))

        players_map = info.get("players") or {}
        p1 = _pad_players([str(x) for x in players_map.get(teams[0], [])], player_slots)
        p2 = _pad_players([str(x) for x in players_map.get(teams[1], [])], player_slots)

        stats, batting_order = _parse_innings_stats(match)

        team1_runs = _to_float((stats.get(team1) or {}).get("runs", 0.0), 0.0)
        team2_runs = _to_float((stats.get(team2) or {}).get("runs", 0.0), 0.0)

        team1_wkts_lost = _to_float((stats.get(team1) or {}).get("wkts_lost", 0.0), 0.0)
        team2_wkts_lost = _to_float((stats.get(team2) or {}).get("wkts_lost", 0.0), 0.0)

        team1_balls = max(_to_float((stats.get(team1) or {}).get("balls", 0.0), 0.0), 1.0)
        team2_balls = max(_to_float((stats.get(team2) or {}).get("balls", 0.0), 0.0), 1.0)

        team1_pp_rr = 6.0 * _to_float((stats.get(team1) or {}).get("powerplay_runs", 0.0), 0.0) / min(team1_balls, 36.0)
        team2_pp_rr = 6.0 * _to_float((stats.get(team2) or {}).get("powerplay_runs", 0.0), 0.0) / min(team2_balls, 36.0)

        team1_death_balls = max(team1_balls - 96.0, 1.0)
        team2_death_balls = max(team2_balls - 96.0, 1.0)
        team1_death_rr = 6.0 * _to_float((stats.get(team1) or {}).get("death_runs", 0.0), 0.0) / team1_death_balls
        team2_death_rr = 6.0 * _to_float((stats.get(team2) or {}).get("death_runs", 0.0), 0.0) / team2_death_balls

        t1_form = _rolling_mean(team_wins_hist[team1], n=5, default=0.5)
        t2_form = _rolling_mean(team_wins_hist[team2], n=5, default=0.5)

        alpha = 1.0
        vg = venue_games[venue]
        vcw = venue_chase_wins[venue]
        venue_chase_prior = (vcw + alpha) / (vg + 2 * alpha) if vg > 0 else 0.5
        venue_score_prior = 2 * venue_chase_prior - 1
        venue_first_avg_prior = float(np.mean(venue_first_innings_runs[venue])) if venue_first_innings_runs[venue] else 165.0

        hkey = (team1, team2)
        h2h_rate = (h2h_team1_wins[hkey] / h2h_games[hkey]) if h2h_games[hkey] > 0 else 0.5

        vt1_games = venue_team_games[(venue, team1)]
        vt2_games = venue_team_games[(venue, team2)]
        vt1_rate = (venue_team_wins[(venue, team1)] + 1) / (vt1_games + 2)
        vt2_rate = (venue_team_wins[(venue, team2)] + 1) / (vt2_games + 2)

        t1_runs_for_5 = _rolling_mean(team_runs_for_hist[team1], n=5, default=160.0)
        t2_runs_for_5 = _rolling_mean(team_runs_for_hist[team2], n=5, default=160.0)
        t1_runs_against_5 = _rolling_mean(team_runs_against_hist[team1], n=5, default=160.0)
        t2_runs_against_5 = _rolling_mean(team_runs_against_hist[team2], n=5, default=160.0)

        t1_wkts_taken_5 = _rolling_mean(team_wkts_taken_hist[team1], n=5, default=6.0)
        t2_wkts_taken_5 = _rolling_mean(team_wkts_taken_hist[team2], n=5, default=6.0)

        t1_pp_rr_5 = _rolling_mean(team_powerplay_rr_hist[team1], n=5, default=7.5)
        t2_pp_rr_5 = _rolling_mean(team_powerplay_rr_hist[team2], n=5, default=7.5)
        t1_death_rr_5 = _rolling_mean(team_death_rr_hist[team1], n=5, default=9.5)
        t2_death_rr_5 = _rolling_mean(team_death_rr_hist[team2], n=5, default=9.5)

        p1_known = [p for p in p1 if p != "Unknown_Player"]
        p2_known = [p for p in p2 if p != "Unknown_Player"]

        p1_elos = [player_elo.get(p, base_elo) for p in p1_known] or [base_elo]
        p2_elos = [player_elo.get(p, base_elo) for p in p2_known] or [base_elo]

        p1_avg = float(np.mean(p1_elos))
        p2_avg = float(np.mean(p2_elos))
        p1_max = float(np.max(p1_elos))
        p2_max = float(np.max(p2_elos))
        p1_min = float(np.min(p1_elos))
        p2_min = float(np.min(p2_elos))

        record: dict[str, Any] = {
            "Match_ID": match_id,
            "Date": date,
            "Teams": f"{team1} vs {team2}",
            "Team1": team1,
            "Team2": team2,
            "Toss_Winner": toss_winner,
            "Toss_Decision": toss_decision,
            "team1_form_winrate_5": t1_form,
            "team2_form_winrate_5": t2_form,
            "venue_chase_winrate_prior": venue_chase_prior,
            "venue_score_prior": venue_score_prior,
            "h2h_team1_winrate_prior": h2h_rate,
            "venue_team1_winrate_prior": vt1_rate,
            "venue_team2_winrate_prior": vt2_rate,
            "venue_avg_first_innings_runs_prior": venue_first_avg_prior,
            "team1_recent_runs_for_5": t1_runs_for_5,
            "team2_recent_runs_for_5": t2_runs_for_5,
            "team1_recent_runs_against_5": t1_runs_against_5,
            "team2_recent_runs_against_5": t2_runs_against_5,
            "team1_recent_wkts_taken_5": t1_wkts_taken_5,
            "team2_recent_wkts_taken_5": t2_wkts_taken_5,
            "team1_recent_powerplay_rr_5": t1_pp_rr_5,
            "team2_recent_powerplay_rr_5": t2_pp_rr_5,
            "team1_recent_death_rr_5": t1_death_rr_5,
            "team2_recent_death_rr_5": t2_death_rr_5,
            "team1_player_elo_avg_prior": p1_avg,
            "team2_player_elo_avg_prior": p2_avg,
            "team1_player_elo_max_prior": p1_max,
            "team2_player_elo_max_prior": p2_max,
            "team1_player_elo_min_prior": p1_min,
            "team2_player_elo_min_prior": p2_min,
            "player_elo_gap_prior": p1_avg - p2_avg,
            "Match_Winner": winner,
        }

        for i, name in enumerate(p1, start=1):
            record[f"Team1_Player_{i}"] = name
        for i, name in enumerate(p2, start=1):
            record[f"Team2_Player_{i}"] = name

        matches.append(record)

        # Update histories after writing current prior row.
        if winner == team1:
            team_wins_hist[team1].append(1.0)
            team_wins_hist[team2].append(0.0)
        elif winner == team2:
            team_wins_hist[team1].append(0.0)
            team_wins_hist[team2].append(1.0)
        else:
            team_wins_hist[team1].append(0.5)
            team_wins_hist[team2].append(0.5)

        team_runs_for_hist[team1].append(team1_runs)
        team_runs_for_hist[team2].append(team2_runs)
        team_runs_against_hist[team1].append(team2_runs)
        team_runs_against_hist[team2].append(team1_runs)

        team_wkts_taken_hist[team1].append(team2_wkts_lost)
        team_wkts_taken_hist[team2].append(team1_wkts_lost)

        team_powerplay_rr_hist[team1].append(team1_pp_rr)
        team_powerplay_rr_hist[team2].append(team2_pp_rr)
        team_death_rr_hist[team1].append(team1_death_rr)
        team_death_rr_hist[team2].append(team2_death_rr)

        h2h_games[hkey] += 1
        if winner == team1:
            h2h_team1_wins[hkey] += 1

        venue_team_games[(venue, team1)] += 1
        venue_team_games[(venue, team2)] += 1
        if winner == team1:
            venue_team_wins[(venue, team1)] += 1
        elif winner == team2:
            venue_team_wins[(venue, team2)] += 1

        # Update venue chase stats from actual chase outcome.
        if len(batting_order) >= 2:
            bat_first = batting_order[0]
            bat_second = batting_order[1]
            first_runs = _to_float((stats.get(bat_first) or {}).get("runs", 0.0), 0.0)
            venue_first_innings_runs[venue].append(first_runs)
            venue_games[venue] += 1
            if winner == bat_second:
                venue_chase_wins[venue] += 1

        # Update player ELO.
        if winner == team1:
            s1, s2 = 1.0, 0.0
        elif winner == team2:
            s1, s2 = 0.0, 1.0
        else:
            s1, s2 = 0.5, 0.5

        e1 = 1.0 / (1.0 + 10.0 ** ((p2_avg - p1_avg) / 400.0))
        e2 = 1.0 - e1
        d1 = k_factor * (s1 - e1)
        d2 = k_factor * (s2 - e2)

        for p in p1_known:
            player_elo[p] = player_elo.get(p, base_elo) + d1
            player_matches[p] += 1
        for p in p2_known:
            player_elo[p] = player_elo.get(p, base_elo) + d2
            player_matches[p] += 1

    out = pd.DataFrame(matches)
    if out.empty:
        raise ValueError("No IPL matches were parsed from YAML files")

    out = out.sort_values(["Date", "Match_ID"]).reset_index(drop=True)
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 4 dataset with advanced ball-by-ball priors.")
    parser.add_argument("--yaml-dir", default=str(ROOT_DIR / "data/raw/ipl_male"))
    parser.add_argument("--output", default=str(ROOT_DIR / "phases/phase_4/results/phase4_dataset.csv"))
    parser.add_argument("--player-slots", type=int, default=11)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = build_phase4_dataset(
        yaml_dir=Path(args.yaml_dir),
        output_csv=Path(args.output),
        player_slots=args.player_slots,
    )
    print(f"Saved Phase 4 dataset: {args.output} ({len(data)} rows)")


if __name__ == "__main__":
    main()
