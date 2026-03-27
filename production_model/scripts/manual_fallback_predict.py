import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from auto_predict_trigger import AutoPredictor
from ops_db import default_db_path, save_prediction


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Manual fallback predictor when live API automation fails")
    p.add_argument("--api-key", default="manual", help="Optional, not required for manual mode")
    p.add_argument("--team1", required=True)
    p.add_argument("--team2", required=True)
    p.add_argument("--venue", required=True)
    p.add_argument("--toss-winner", required=True)
    p.add_argument("--toss-decision", required=True, choices=["bat", "field", "bowl"])
    p.add_argument("--match-id", default=None)
    p.add_argument("--match-name", default=None)
    p.add_argument("--series-name", default="Indian Premier League")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent.parent

    predictor = AutoPredictor(base_dir, args.api_key)

    payload = {
        "id": args.match_id or f"manual-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        "name": args.match_name or f"{args.team1} vs {args.team2}",
        "teams": [args.team1, args.team2],
        "venue": args.venue,
        "tossWinner": args.toss_winner,
        "tossChoice": "field" if args.toss_decision == "bowl" else args.toss_decision,
        "dateTimeGMT": datetime.now(timezone.utc).isoformat(),
        "matchStarted": False,
        "matchEnded": False,
    }

    result = predictor.predict_from_match_payload(payload, squad_ok=False, mins_to_start=None)
    result["prediction_source"] = "manual"
    result["series_name"] = args.series_name
    result["match_start_utc"] = payload["dateTimeGMT"]
    result["match_started"] = False
    result["match_ended"] = False

    db_path = default_db_path(base_dir)
    save_prediction(db_path, result)

    out_dir = base_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"manual_prediction_{result['match_id']}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"Saved result: {out_file}")
    print(f"Database updated: {db_path}")


if __name__ == "__main__":
    main()
