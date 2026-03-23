from __future__ import annotations

import json
import sqlite3
from typing import Any

import numpy as np


BASE_ELO = 1000.0
K_FACTOR = 20.0


def _to_json(values: list[float]) -> str:
    return json.dumps(values)


def _safe_mean(values: list[float], default: float) -> float:
    if not values:
        return default
    return float(np.mean(values))


def _rolling(values: list[float], n: int, default: float) -> float:
    if not values:
        return default
    return float(np.mean(values[-n:]))


def _fetch_team_history(conn: sqlite3.Connection, team: str) -> dict[str, list[float]]:
    cur = conn.execute(
        """
        SELECT wins_json, runs_for_json, runs_against_json, wkts_taken_json, pp_rr_json, death_rr_json
        FROM team_history WHERE team_name=?
        """,
        (team,),
    )
    row = cur.fetchone()
    if row is None:
        return {
            "wins": [],
            "runs_for": [],
            "runs_against": [],
            "wkts_taken": [],
            "pp_rr": [],
            "death_rr": [],
        }
    return {
        "wins": json.loads(row["wins_json"]),
        "runs_for": json.loads(row["runs_for_json"]),
        "runs_against": json.loads(row["runs_against_json"]),
        "wkts_taken": json.loads(row["wkts_taken_json"]),
        "pp_rr": json.loads(row["pp_rr_json"]),
        "death_rr": json.loads(row["death_rr_json"]),
    }


def _upsert_team_history(conn: sqlite3.Connection, team: str, hist: dict[str, list[float]]) -> None:
    conn.execute(
        """
        INSERT INTO team_history (
            team_name, wins_json, runs_for_json, runs_against_json,
            wkts_taken_json, pp_rr_json, death_rr_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(team_name) DO UPDATE SET
            wins_json=excluded.wins_json,
            runs_for_json=excluded.runs_for_json,
            runs_against_json=excluded.runs_against_json,
            wkts_taken_json=excluded.wkts_taken_json,
            pp_rr_json=excluded.pp_rr_json,
            death_rr_json=excluded.death_rr_json,
            updated_at=CURRENT_TIMESTAMP
        """,
        (
            team,
            _to_json(hist["wins"]),
            _to_json(hist["runs_for"]),
            _to_json(hist["runs_against"]),
            _to_json(hist["wkts_taken"]),
            _to_json(hist["pp_rr"]),
            _to_json(hist["death_rr"]),
        ),
    )


def _fetch_lineup(conn: sqlite3.Connection, team: str) -> set[str]:
    cur = conn.execute("SELECT lineup_json FROM team_lineup WHERE team_name=?", (team,))
    row = cur.fetchone()
    if row is None:
        return set()
    return set(json.loads(row["lineup_json"]))


def _upsert_lineup(conn: sqlite3.Connection, team: str, players: list[str]) -> None:
    lineup = [p for p in players if p and p != "Unknown_Player"]
    conn.execute(
        """
        INSERT INTO team_lineup (team_name, lineup_json)
        VALUES (?, ?)
        ON CONFLICT(team_name) DO UPDATE SET
            lineup_json=excluded.lineup_json,
            updated_at=CURRENT_TIMESTAMP
        """,
        (team, json.dumps(lineup)),
    )


def _fetch_player_elo(conn: sqlite3.Connection, player: str) -> float:
    cur = conn.execute("SELECT elo FROM player_elo WHERE player_name=?", (player,))
    row = cur.fetchone()
    return float(row["elo"]) if row else BASE_ELO


def _upsert_player_elo(conn: sqlite3.Connection, player: str, elo: float, seen_date: str) -> None:
    conn.execute(
        """
        INSERT INTO player_elo (player_name, elo, last_seen_date)
        VALUES (?, ?, ?)
        ON CONFLICT(player_name) DO UPDATE SET
            elo=excluded.elo,
            last_seen_date=excluded.last_seen_date
        """,
        (player, float(elo), seen_date),
    )


def _fetch_venue_stats(conn: sqlite3.Connection, venue: str) -> dict[str, Any]:
    cur = conn.execute("SELECT * FROM venue_stats WHERE venue=?", (venue,))
    row = cur.fetchone()
    if row is None:
        return {
            "games": 0,
            "chase_wins": 0,
            "first_runs": [],
            "toss_bat_count": 0,
            "toss_count": 0,
            "pp_first": [],
            "death_first": [],
            "pp_second": [],
            "death_second": [],
        }
    return {
        "games": int(row["games"]),
        "chase_wins": int(row["chase_wins"]),
        "first_runs": json.loads(row["first_innings_runs_json"]),
        "toss_bat_count": int(row["toss_bat_count"]),
        "toss_count": int(row["toss_count"]),
        "pp_first": json.loads(row["pp_first_json"]),
        "death_first": json.loads(row["death_first_json"]),
        "pp_second": json.loads(row["pp_second_json"]),
        "death_second": json.loads(row["death_second_json"]),
    }


def _upsert_venue_stats(conn: sqlite3.Connection, venue: str, s: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO venue_stats (
            venue, games, chase_wins, first_innings_runs_json,
            toss_bat_count, toss_count, pp_first_json, death_first_json,
            pp_second_json, death_second_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(venue) DO UPDATE SET
            games=excluded.games,
            chase_wins=excluded.chase_wins,
            first_innings_runs_json=excluded.first_innings_runs_json,
            toss_bat_count=excluded.toss_bat_count,
            toss_count=excluded.toss_count,
            pp_first_json=excluded.pp_first_json,
            death_first_json=excluded.death_first_json,
            pp_second_json=excluded.pp_second_json,
            death_second_json=excluded.death_second_json,
            updated_at=CURRENT_TIMESTAMP
        """,
        (
            venue,
            int(s["games"]),
            int(s["chase_wins"]),
            json.dumps(s["first_runs"]),
            int(s["toss_bat_count"]),
            int(s["toss_count"]),
            json.dumps(s["pp_first"]),
            json.dumps(s["death_first"]),
            json.dumps(s["pp_second"]),
            json.dumps(s["death_second"]),
        ),
    )


def _fetch_venue_team(conn: sqlite3.Connection, venue: str, team: str) -> tuple[int, int]:
    cur = conn.execute(
        "SELECT games, wins FROM venue_team_stats WHERE venue=? AND team_name=?",
        (venue, team),
    )
    row = cur.fetchone()
    return (int(row["games"]), int(row["wins"])) if row else (0, 0)


def _upsert_venue_team(conn: sqlite3.Connection, venue: str, team: str, games: int, wins: int) -> None:
    conn.execute(
        """
        INSERT INTO venue_team_stats (venue, team_name, games, wins)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(venue, team_name) DO UPDATE SET
            games=excluded.games,
            wins=excluded.wins
        """,
        (venue, team, int(games), int(wins)),
    )


def _fetch_h2h(conn: sqlite3.Connection, team1: str, team2: str) -> tuple[int, int]:
    cur = conn.execute("SELECT games, team1_wins FROM h2h_stats WHERE team1=? AND team2=?", (team1, team2))
    row = cur.fetchone()
    return (int(row["games"]), int(row["team1_wins"])) if row else (0, 0)


def _upsert_h2h(conn: sqlite3.Connection, team1: str, team2: str, games: int, team1_wins: int) -> None:
    conn.execute(
        """
        INSERT INTO h2h_stats (team1, team2, games, team1_wins)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(team1, team2) DO UPDATE SET
            games=excluded.games,
            team1_wins=excluded.team1_wins
        """,
        (team1, team2, int(games), int(team1_wins)),
    )


def _fetch_toss_decision_stats(conn: sqlite3.Connection, venue: str, toss_team1_flag: int, toss_decision: str) -> tuple[int, int]:
    cur = conn.execute(
        """
        SELECT games, team1_wins FROM toss_venue_decision_stats
        WHERE venue=? AND toss_team1_flag=? AND toss_decision=?
        """,
        (venue, int(toss_team1_flag), toss_decision),
    )
    row = cur.fetchone()
    return (int(row["games"]), int(row["team1_wins"])) if row else (0, 0)


def _upsert_toss_decision_stats(
    conn: sqlite3.Connection, venue: str, toss_team1_flag: int, toss_decision: str, games: int, team1_wins: int
) -> None:
    conn.execute(
        """
        INSERT INTO toss_venue_decision_stats (venue, toss_team1_flag, toss_decision, games, team1_wins)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(venue, toss_team1_flag, toss_decision) DO UPDATE SET
            games=excluded.games,
            team1_wins=excluded.team1_wins
        """,
        (venue, int(toss_team1_flag), toss_decision, int(games), int(team1_wins)),
    )


def build_pre_match_features(conn: sqlite3.Connection, payload: dict[str, Any], feature_columns: list[str]) -> dict[str, Any]:
    team1 = payload["team1"]
    team2 = payload["team2"]
    venue = payload["venue"]
    toss_winner = payload.get("toss_winner", team1)
    toss_decision = (payload.get("toss_decision", "field") or "field").lower()

    team1_players = (payload.get("team1_players") or [])[:11]
    team2_players = (payload.get("team2_players") or [])[:11]
    team1_players += ["Unknown_Player"] * (11 - len(team1_players))
    team2_players += ["Unknown_Player"] * (11 - len(team2_players))

    h1 = _fetch_team_history(conn, team1)
    h2 = _fetch_team_history(conn, team2)

    vs = _fetch_venue_stats(conn, venue)
    v1_games, v1_wins = _fetch_venue_team(conn, venue, team1)
    v2_games, v2_wins = _fetch_venue_team(conn, venue, team2)

    h2h_games, h2h_team1_wins = _fetch_h2h(conn, team1, team2)

    toss_team1_flag = 1 if toss_winner == team1 else 0
    td_games, td_wins = _fetch_toss_decision_stats(conn, venue, toss_team1_flag, toss_decision)

    t1_form = _rolling(h1["wins"], 5, 0.5)
    t2_form = _rolling(h2["wins"], 5, 0.5)

    alpha = 1.0
    venue_chase_prior = (vs["chase_wins"] + alpha) / (vs["games"] + 2 * alpha) if vs["games"] > 0 else 0.5
    venue_score_prior = 2 * venue_chase_prior - 1
    venue_first_avg_prior = _safe_mean(vs["first_runs"], 165.0)

    h2h_rate = h2h_team1_wins / h2h_games if h2h_games > 0 else 0.5
    vt1_rate = (v1_wins + 1) / (v1_games + 2)
    vt2_rate = (v2_wins + 1) / (v2_games + 2)

    t1_runs_for_5 = _rolling(h1["runs_for"], 5, 160.0)
    t2_runs_for_5 = _rolling(h2["runs_for"], 5, 160.0)
    t1_runs_against_5 = _rolling(h1["runs_against"], 5, 160.0)
    t2_runs_against_5 = _rolling(h2["runs_against"], 5, 160.0)

    t1_wkts_5 = _rolling(h1["wkts_taken"], 5, 6.0)
    t2_wkts_5 = _rolling(h2["wkts_taken"], 5, 6.0)

    t1_pp_5 = _rolling(h1["pp_rr"], 5, 7.5)
    t2_pp_5 = _rolling(h2["pp_rr"], 5, 7.5)
    t1_death_5 = _rolling(h1["death_rr"], 5, 9.5)
    t2_death_5 = _rolling(h2["death_rr"], 5, 9.5)

    p1_elo = [
        _fetch_player_elo(conn, p)
        for p in team1_players
        if p and p != "Unknown_Player"
    ] or [BASE_ELO]
    p2_elo = [
        _fetch_player_elo(conn, p)
        for p in team2_players
        if p and p != "Unknown_Player"
    ] or [BASE_ELO]

    t1_elo_avg = float(np.mean(p1_elo))
    t2_elo_avg = float(np.mean(p2_elo))
    t1_elo_max = float(np.max(p1_elo))
    t2_elo_max = float(np.max(p2_elo))
    t1_elo_min = float(np.min(p1_elo))
    t2_elo_min = float(np.min(p2_elo))

    prev_t1 = _fetch_lineup(conn, team1)
    prev_t2 = _fetch_lineup(conn, team2)
    t1_set = {p for p in team1_players if p != "Unknown_Player"}
    t2_set = {p for p in team2_players if p != "Unknown_Player"}

    t1_xi_cont = len(prev_t1 & t1_set) / max(len(t1_set), 1) if prev_t1 else 0.5
    t2_xi_cont = len(prev_t2 & t2_set) / max(len(t2_set), 1) if prev_t2 else 0.5

    venue_toss_bat_rate = vs["toss_bat_count"] / vs["toss_count"] if vs["toss_count"] > 0 else 0.5
    toss_alignment = 1.0 if ((venue_toss_bat_rate >= 0.5 and toss_decision == "bat") or (venue_toss_bat_rate < 0.5 and toss_decision == "field")) else 0.0
    toss_decision_team1_winrate = (td_wins + 1) / (td_games + 2)

    toss_field_chase_boost = 0.0
    if toss_decision == "field":
        toss_field_chase_boost = venue_chase_prior if toss_team1_flag == 1 else (1.0 - venue_chase_prior)

    feature_row: dict[str, Any] = {
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
        "team1_recent_wkts_taken_5": t1_wkts_5,
        "team2_recent_wkts_taken_5": t2_wkts_5,
        "team1_recent_powerplay_rr_5": t1_pp_5,
        "team2_recent_powerplay_rr_5": t2_pp_5,
        "team1_recent_death_rr_5": t1_death_5,
        "team2_recent_death_rr_5": t2_death_5,
        "team1_player_elo_avg_prior": t1_elo_avg,
        "team2_player_elo_avg_prior": t2_elo_avg,
        "team1_player_elo_max_prior": t1_elo_max,
        "team2_player_elo_max_prior": t2_elo_max,
        "team1_player_elo_min_prior": t1_elo_min,
        "team2_player_elo_min_prior": t2_elo_min,
        "player_elo_gap_prior": t1_elo_avg - t2_elo_avg,
        "team1_xi_continuity_prior": t1_xi_cont,
        "team2_xi_continuity_prior": t2_xi_cont,
        "team1_player_availability_prior": 0.5,
        "team2_player_availability_prior": 0.5,
        "team1_injury_proxy_prior": 0.5,
        "team2_injury_proxy_prior": 0.5,
        "venue_toss_bat_rate_prior": venue_toss_bat_rate,
        "toss_decision_alignment_prior": toss_alignment,
        "toss_decision_venue_team1_winrate_prior": toss_decision_team1_winrate,
        "toss_field_chase_boost_team1": toss_field_chase_boost,
        "venue_pp_rr_first_innings_prior": _safe_mean(vs["pp_first"], 7.5),
        "venue_death_rr_first_innings_prior": _safe_mean(vs["death_first"], 9.5),
        "venue_pp_rr_second_innings_prior": _safe_mean(vs["pp_second"], 7.3),
        "venue_death_rr_second_innings_prior": _safe_mean(vs["death_second"], 9.2),
    }

    for i, p in enumerate(team1_players, start=1):
        feature_row[f"Team1_Player_{i}"] = p
    for i, p in enumerate(team2_players, start=1):
        feature_row[f"Team2_Player_{i}"] = p

    # Ensure all model features exist.
    for c in feature_columns:
        if c not in feature_row:
            feature_row[c] = 0.0

    return feature_row


def update_post_match_state(conn: sqlite3.Connection, pre_payload: dict[str, Any], post_payload: dict[str, Any]) -> None:
    team1 = pre_payload["team1"]
    team2 = pre_payload["team2"]
    venue = pre_payload["venue"]
    toss_winner = pre_payload.get("toss_winner", team1)
    toss_decision = (pre_payload.get("toss_decision", "field") or "field").lower()
    match_date = pre_payload["date"]

    winner = post_payload["actual_winner"]

    team1_players = [p for p in (pre_payload.get("team1_players") or []) if p and p != "Unknown_Player"]
    team2_players = [p for p in (pre_payload.get("team2_players") or []) if p and p != "Unknown_Player"]

    # Team history updates.
    h1 = _fetch_team_history(conn, team1)
    h2 = _fetch_team_history(conn, team2)

    if winner == team1:
        h1["wins"].append(1.0)
        h2["wins"].append(0.0)
    elif winner == team2:
        h1["wins"].append(0.0)
        h2["wins"].append(1.0)
    else:
        h1["wins"].append(0.5)
        h2["wins"].append(0.5)

    t1_runs = float(post_payload["team1_runs"])
    t2_runs = float(post_payload["team2_runs"])
    t1_wkts_lost = float(post_payload["team1_wkts_lost"])
    t2_wkts_lost = float(post_payload["team2_wkts_lost"])
    t1_balls = max(float(post_payload["team1_balls"]), 1.0)
    t2_balls = max(float(post_payload["team2_balls"]), 1.0)
    t1_pp_runs = float(post_payload["team1_powerplay_runs"])
    t2_pp_runs = float(post_payload["team2_powerplay_runs"])
    t1_death_runs = float(post_payload["team1_death_runs"])
    t2_death_runs = float(post_payload["team2_death_runs"])

    h1["runs_for"].append(t1_runs)
    h2["runs_for"].append(t2_runs)
    h1["runs_against"].append(t2_runs)
    h2["runs_against"].append(t1_runs)

    h1["wkts_taken"].append(t2_wkts_lost)
    h2["wkts_taken"].append(t1_wkts_lost)

    h1["pp_rr"].append(6.0 * t1_pp_runs / min(t1_balls, 36.0))
    h2["pp_rr"].append(6.0 * t2_pp_runs / min(t2_balls, 36.0))

    h1["death_rr"].append(6.0 * t1_death_runs / max(t1_balls - 96.0, 1.0))
    h2["death_rr"].append(6.0 * t2_death_runs / max(t2_balls - 96.0, 1.0))

    _upsert_team_history(conn, team1, h1)
    _upsert_team_history(conn, team2, h2)

    _upsert_lineup(conn, team1, team1_players)
    _upsert_lineup(conn, team2, team2_players)

    # H2H updates.
    games, t1_wins = _fetch_h2h(conn, team1, team2)
    games += 1
    if winner == team1:
        t1_wins += 1
    _upsert_h2h(conn, team1, team2, games, t1_wins)

    # Venue/team updates.
    v1_games, v1_wins = _fetch_venue_team(conn, venue, team1)
    v2_games, v2_wins = _fetch_venue_team(conn, venue, team2)
    v1_games += 1
    v2_games += 1
    if winner == team1:
        v1_wins += 1
    elif winner == team2:
        v2_wins += 1
    _upsert_venue_team(conn, venue, team1, v1_games, v1_wins)
    _upsert_venue_team(conn, venue, team2, v2_games, v2_wins)

    # Toss decision and venue priors.
    toss_team1_flag = 1 if toss_winner == team1 else 0
    td_games, td_wins = _fetch_toss_decision_stats(conn, venue, toss_team1_flag, toss_decision)
    td_games += 1
    if winner == team1:
        td_wins += 1
    _upsert_toss_decision_stats(conn, venue, toss_team1_flag, toss_decision, td_games, td_wins)

    vs = _fetch_venue_stats(conn, venue)
    vs["games"] += 1
    if toss_decision == "bat":
        vs["toss_bat_count"] += 1
    vs["toss_count"] += 1

    batting_first = post_payload["batting_first"]
    batting_second = post_payload["batting_second"]
    first_runs = t1_runs if batting_first == team1 else t2_runs

    vs["first_runs"].append(first_runs)
    if winner == batting_second:
        vs["chase_wins"] += 1

    first_pp = h1["pp_rr"][-1] if batting_first == team1 else h2["pp_rr"][-1]
    second_pp = h2["pp_rr"][-1] if batting_first == team1 else h1["pp_rr"][-1]
    first_death = h1["death_rr"][-1] if batting_first == team1 else h2["death_rr"][-1]
    second_death = h2["death_rr"][-1] if batting_first == team1 else h1["death_rr"][-1]

    vs["pp_first"].append(first_pp)
    vs["death_first"].append(first_death)
    vs["pp_second"].append(second_pp)
    vs["death_second"].append(second_death)

    _upsert_venue_stats(conn, venue, vs)

    # Player ELO updates.
    t1_elo = [_fetch_player_elo(conn, p) for p in team1_players] or [BASE_ELO]
    t2_elo = [_fetch_player_elo(conn, p) for p in team2_players] or [BASE_ELO]
    t1_avg = float(np.mean(t1_elo))
    t2_avg = float(np.mean(t2_elo))

    if winner == team1:
        s1, s2 = 1.0, 0.0
    elif winner == team2:
        s1, s2 = 0.0, 1.0
    else:
        s1, s2 = 0.5, 0.5

    e1 = 1.0 / (1.0 + 10.0 ** ((t2_avg - t1_avg) / 400.0))
    e2 = 1.0 - e1

    d1 = K_FACTOR * (s1 - e1)
    d2 = K_FACTOR * (s2 - e2)

    for p in team1_players:
        _upsert_player_elo(conn, p, _fetch_player_elo(conn, p) + d1, match_date)
    for p in team2_players:
        _upsert_player_elo(conn, p, _fetch_player_elo(conn, p) + d2, match_date)

    conn.commit()
