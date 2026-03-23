from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS matches (
            match_id TEXT PRIMARY KEY,
            match_date TEXT NOT NULL,
            venue TEXT NOT NULL,
            team1 TEXT NOT NULL,
            team2 TEXT NOT NULL,
            toss_winner TEXT,
            toss_decision TEXT,
            predicted_winner TEXT,
            confidence REAL,
            probabilities_json TEXT,
            actual_winner TEXT,
            status TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS player_elo (
            player_name TEXT PRIMARY KEY,
            elo REAL NOT NULL,
            last_seen_date TEXT
        );

        CREATE TABLE IF NOT EXISTS team_lineup (
            team_name TEXT PRIMARY KEY,
            lineup_json TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS team_history (
            team_name TEXT PRIMARY KEY,
            wins_json TEXT NOT NULL,
            runs_for_json TEXT NOT NULL,
            runs_against_json TEXT NOT NULL,
            wkts_taken_json TEXT NOT NULL,
            pp_rr_json TEXT NOT NULL,
            death_rr_json TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS h2h_stats (
            team1 TEXT NOT NULL,
            team2 TEXT NOT NULL,
            games INTEGER NOT NULL,
            team1_wins INTEGER NOT NULL,
            PRIMARY KEY (team1, team2)
        );

        CREATE TABLE IF NOT EXISTS venue_stats (
            venue TEXT PRIMARY KEY,
            games INTEGER NOT NULL,
            chase_wins INTEGER NOT NULL,
            first_innings_runs_json TEXT NOT NULL,
            toss_bat_count INTEGER NOT NULL,
            toss_count INTEGER NOT NULL,
            pp_first_json TEXT NOT NULL,
            death_first_json TEXT NOT NULL,
            pp_second_json TEXT NOT NULL,
            death_second_json TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS venue_team_stats (
            venue TEXT NOT NULL,
            team_name TEXT NOT NULL,
            games INTEGER NOT NULL,
            wins INTEGER NOT NULL,
            PRIMARY KEY (venue, team_name)
        );

        CREATE TABLE IF NOT EXISTS toss_venue_decision_stats (
            venue TEXT NOT NULL,
            toss_team1_flag INTEGER NOT NULL,
            toss_decision TEXT NOT NULL,
            games INTEGER NOT NULL,
            team1_wins INTEGER NOT NULL,
            PRIMARY KEY (venue, toss_team1_flag, toss_decision)
        );
        """
    )
    conn.commit()


def upsert_match_prediction(
    conn: sqlite3.Connection,
    match_id: str,
    match_date: str,
    venue: str,
    team1: str,
    team2: str,
    toss_winner: str,
    toss_decision: str,
    predicted_winner: str,
    confidence: float,
    probabilities: dict[str, float],
) -> None:
    conn.execute(
        """
        INSERT INTO matches (
            match_id, match_date, venue, team1, team2, toss_winner, toss_decision,
            predicted_winner, confidence, probabilities_json, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'predicted')
        ON CONFLICT(match_id) DO UPDATE SET
            match_date=excluded.match_date,
            venue=excluded.venue,
            team1=excluded.team1,
            team2=excluded.team2,
            toss_winner=excluded.toss_winner,
            toss_decision=excluded.toss_decision,
            predicted_winner=excluded.predicted_winner,
            confidence=excluded.confidence,
            probabilities_json=excluded.probabilities_json,
            status='predicted',
            updated_at=CURRENT_TIMESTAMP
        """,
        (
            match_id,
            match_date,
            venue,
            team1,
            team2,
            toss_winner,
            toss_decision,
            predicted_winner,
            confidence,
            json.dumps(probabilities),
        ),
    )
    conn.commit()


def mark_match_final(conn: sqlite3.Connection, match_id: str, actual_winner: str) -> None:
    conn.execute(
        """
        UPDATE matches
        SET actual_winner=?, status='final', updated_at=CURRENT_TIMESTAMP
        WHERE match_id=?
        """,
        (actual_winner, match_id),
    )
    conn.commit()


def get_match(conn: sqlite3.Connection, match_id: str) -> sqlite3.Row | None:
    cur = conn.execute("SELECT * FROM matches WHERE match_id=?", (match_id,))
    return cur.fetchone()


def get_latest_matches(conn: sqlite3.Connection, limit: int = 50) -> list[sqlite3.Row]:
    cur = conn.execute(
        "SELECT * FROM matches ORDER BY match_date DESC, created_at DESC LIMIT ?",
        (limit,),
    )
    return cur.fetchall()


def get_json_field(conn: sqlite3.Connection, table: str, key_col: str, key_val: str, field: str) -> list[float]:
    cur = conn.execute(f"SELECT {field} FROM {table} WHERE {key_col}=?", (key_val,))
    row = cur.fetchone()
    if row is None:
        return []
    return json.loads(row[field])


def upsert_json_history(
    conn: sqlite3.Connection,
    table: str,
    key_col: str,
    key_val: str,
    fields: dict[str, Any],
) -> None:
    cols = [key_col] + list(fields.keys())
    placeholders = ",".join(["?"] * len(cols))
    updates = ",".join([f"{c}=excluded.{c}" for c in fields.keys()])
    sql = f"""
        INSERT INTO {table} ({','.join(cols)})
        VALUES ({placeholders})
        ON CONFLICT({key_col}) DO UPDATE SET {updates}, updated_at=CURRENT_TIMESTAMP
    """
    values = [key_val] + [fields[c] for c in fields.keys()]
    conn.execute(sql, values)
    conn.commit()
