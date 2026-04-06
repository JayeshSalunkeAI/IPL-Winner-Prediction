from datetime import datetime, timezone
from pathlib import Path

from auto_predict_trigger import AutoPredictor
from ops_db import default_db_path, save_prediction


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value if value else default


def normalize_decision(value: str) -> str:
    v = value.strip().lower()
    if v in {"bowl", "field", "f", "chase"}:
        return "field"
    if v in {"bat", "b", "batting"}:
        return "bat"
    return "field"


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent

    print("Manual IPL Prediction Input")
    print("Enter match details. Press Enter to accept defaults where shown.\n")

    team1 = ask("Team 1 (full name)")
    team2 = ask("Team 2 (full name)")
    venue = ask("Venue (full name)")
    toss_winner = ask("Toss Winner (full name)")
    toss_decision = normalize_decision(ask("Toss Decision (bat/field)", "field"))

    match_id = ask("Match ID", f"manual-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}")
    match_name = ask("Match Name", f"{team1} vs {team2}")
    series_name = ask("Series Name", "Indian Premier League")

    predictor = AutoPredictor(base_dir, api_key="manual")

    payload = {
        "id": match_id,
        "name": match_name,
        "teams": [team1, team2],
        "venue": venue,
        "tossWinner": toss_winner,
        "tossChoice": toss_decision,
        "dateTimeGMT": datetime.now(timezone.utc).isoformat(),
        "matchStarted": False,
        "matchEnded": False,
    }

    result = predictor.predict_from_match_payload(payload, squad_ok=False, mins_to_start=None)
    result["prediction_source"] = "manual"
    result["series_name"] = series_name
    result["match_start_utc"] = payload["dateTimeGMT"]
    result["match_started"] = False
    result["match_ended"] = False

    db_path = default_db_path(base_dir)
    save_prediction(db_path, result)

    out_dir = base_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"manual_prediction_{result['match_id']}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

    import json

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\nPrediction complete")
    print(f"Predicted Winner: {result['predicted_winner']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Saved JSON: {out_file}")
    print(f"DB updated: {db_path}")


if __name__ == "__main__":
    main()
