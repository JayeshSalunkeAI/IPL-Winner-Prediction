import argparse
import sqlite3
from pathlib import Path

from ops_db import default_db_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Show prediction history and running hit-rate from ops DB")
    p.add_argument("--limit", type=int, default=10, help="Number of latest rows to display")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent.parent
    db_path = default_db_path(base_dir)

    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()

        cur.execute(
            """
            SELECT
                COALESCE(date_utc, created_at) AS ts,
                source_match_id,
                team1,
                team2,
                predicted_winner,
                confidence,
                actual_winner,
                prediction_source
            FROM match_predictions
            ORDER BY COALESCE(date_utc, created_at) DESC
            LIMIT ?
            """,
            (args.limit,),
        )
        rows = cur.fetchall()

        cur.execute(
            """
            SELECT
                COUNT(*) AS total_with_actual,
                SUM(CASE WHEN predicted_winner = actual_winner THEN 1 ELSE 0 END) AS correct
            FROM match_predictions
            WHERE actual_winner IS NOT NULL
            """
        )
        total_with_actual, correct = cur.fetchone()
        total_with_actual = int(total_with_actual or 0)
        correct = int(correct or 0)
        hit_rate = (correct / total_with_actual) if total_with_actual else 0.0

        print("\n=== Prediction Report ===")
        print(f"DB: {db_path}")
        print(f"Rows shown: {len(rows)}")
        print(f"Scored matches: {total_with_actual}")
        print(f"Correct: {correct}")
        print(f"Running hit-rate: {hit_rate:.2%}")
        print("\nLatest entries:")
        print(
            f"{'TS':<20} | {'MATCH_ID':<16} | {'FIXTURE':<33} | {'PRED':<22} | {'CONF':<7} | {'ACTUAL':<22} | {'OK'}"
        )
        print("-" * 140)

        for ts, mid, t1, t2, pred, conf, actual, src in rows:
            fixture = f"{t1} vs {t2}"
            ok = "-"
            if actual:
                ok = "Y" if pred == actual else "N"
            ts_s = (ts or "")[:19]
            mid_s = (mid or "")[:16]
            pred_s = (pred or "")[:22]
            actual_s = (actual or "-")[:22]
            conf_s = f"{float(conf):.2f}" if conf is not None else "-"
            print(
                f"{ts_s:<20} | {mid_s:<16} | {fixture[:33]:<33} | {pred_s:<22} | {conf_s:<7} | {actual_s:<22} | {ok}"
            )

        print("\nTip: store post-match results with record_match_result.py to improve this report and next-match form blend.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
