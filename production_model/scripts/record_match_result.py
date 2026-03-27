import argparse
from pathlib import Path

from ops_db import default_db_path, update_match_result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record completed match result in ops DB")
    p.add_argument("--match-id", default=None, help="API/source match ID used during prediction")
    p.add_argument("--team1", default=None)
    p.add_argument("--team2", default=None)
    p.add_argument("--winner", required=True)
    p.add_argument("--first-innings-score", type=int, default=None)
    p.add_argument("--second-innings-score", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.match_id and not (args.team1 and args.team2):
        raise SystemExit("Provide either --match-id or both --team1 and --team2")

    base_dir = Path(__file__).resolve().parent.parent
    db_path = default_db_path(base_dir)

    updated = update_match_result(
        db_path=db_path,
        source_match_id=args.match_id,
        team1=args.team1,
        team2=args.team2,
        actual_winner=args.winner,
        first_innings_score=args.first_innings_score,
        second_innings_score=args.second_innings_score,
    )

    print(f"Rows updated: {updated}")
    print(f"Database: {db_path}")


if __name__ == "__main__":
    main()
