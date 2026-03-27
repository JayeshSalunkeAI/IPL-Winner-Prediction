import sqlite3
from pathlib import Path
from typing import Dict, Optional, Tuple


def default_db_path(base_dir: Path) -> Path:
    return base_dir / "data" / "ops_matches.db"


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS match_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_match_id TEXT,
                match_name TEXT,
                series_name TEXT,
                date_utc TEXT,
                team1 TEXT NOT NULL,
                team2 TEXT NOT NULL,
                venue TEXT,
                toss_winner TEXT,
                toss_decision TEXT,
                predicted_winner TEXT NOT NULL,
                confidence REAL,
                team1_win_probability REAL,
                actual_winner TEXT,
                first_innings_score INTEGER,
                second_innings_score INTEGER,
                match_started INTEGER DEFAULT 0,
                match_ended INTEGER DEFAULT 0,
                prediction_source TEXT DEFAULT 'auto',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_match_id)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def save_prediction(db_path: Path, payload: Dict) -> int:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO match_predictions (
                source_match_id, match_name, series_name, date_utc,
                team1, team2, venue, toss_winner, toss_decision,
                predicted_winner, confidence, team1_win_probability,
                match_started, match_ended, prediction_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_match_id) DO UPDATE SET
                match_name=excluded.match_name,
                series_name=excluded.series_name,
                date_utc=excluded.date_utc,
                team1=excluded.team1,
                team2=excluded.team2,
                venue=excluded.venue,
                toss_winner=excluded.toss_winner,
                toss_decision=excluded.toss_decision,
                predicted_winner=excluded.predicted_winner,
                confidence=excluded.confidence,
                team1_win_probability=excluded.team1_win_probability,
                match_started=excluded.match_started,
                match_ended=excluded.match_ended,
                prediction_source=excluded.prediction_source,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                payload.get("match_id"),
                payload.get("match_name"),
                payload.get("series_name"),
                payload.get("match_start_utc"),
                payload.get("team1"),
                payload.get("team2"),
                payload.get("venue"),
                payload.get("toss_winner"),
                payload.get("toss_decision"),
                payload.get("predicted_winner"),
                float(payload.get("confidence", 0.0) or 0.0),
                float(payload.get("team1_win_probability", 0.0) or 0.0),
                int(bool(payload.get("match_started", False))),
                int(bool(payload.get("match_ended", False))),
                payload.get("prediction_source", "auto"),
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def update_match_result(
    db_path: Path,
    source_match_id: Optional[str],
    team1: Optional[str],
    team2: Optional[str],
    actual_winner: str,
    first_innings_score: Optional[int] = None,
    second_innings_score: Optional[int] = None,
) -> int:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        if source_match_id:
            cur.execute(
                """
                UPDATE match_predictions
                SET actual_winner=?, first_innings_score=?, second_innings_score=?,
                    match_ended=1, updated_at=CURRENT_TIMESTAMP
                WHERE source_match_id=?
                """,
                (actual_winner, first_innings_score, second_innings_score, source_match_id),
            )
        else:
            cur.execute(
                """
                UPDATE match_predictions
                SET actual_winner=?, first_innings_score=?, second_innings_score=?,
                    match_ended=1, updated_at=CURRENT_TIMESTAMP
                WHERE team1=? AND team2=?
                ORDER BY date_utc DESC
                LIMIT 1
                """,
                (actual_winner, first_innings_score, second_innings_score, team1, team2),
            )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def fetch_recent_team_form(db_path: Path, team: str, n: int = 5) -> Optional[float]:
    if not db_path.exists():
        return None
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT team1, team2, actual_winner
            FROM match_predictions
            WHERE actual_winner IS NOT NULL
              AND (team1=? OR team2=?)
            ORDER BY COALESCE(date_utc, created_at) DESC
            LIMIT ?
            """,
            (team, team, n),
        )
        rows = cur.fetchall()
        if not rows:
            return None
        wins = sum(1 for _, _, winner in rows if winner == team)
        return wins / len(rows)
    finally:
        conn.close()


def fetch_recent_venue_stats(db_path: Path, venue: str) -> Optional[Tuple[float, float]]:
    if not db_path.exists():
        return None
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT toss_winner, actual_winner, first_innings_score
            FROM match_predictions
            WHERE venue=? AND actual_winner IS NOT NULL
            ORDER BY COALESCE(date_utc, created_at) DESC
            LIMIT 20
            """,
            (venue,),
        )
        rows = cur.fetchall()
        if not rows:
            return None

        valid_scores = [r[2] for r in rows if r[2] is not None]
        avg_score = float(sum(valid_scores) / len(valid_scores)) if valid_scores else 170.0

        toss_matches = [r for r in rows if r[0]]
        toss_adv = (
            sum(1 for toss_winner, actual_winner, _ in toss_matches if toss_winner == actual_winner)
            / len(toss_matches)
            if toss_matches
            else 0.5
        )
        return avg_score, float(toss_adv)
    finally:
        conn.close()
