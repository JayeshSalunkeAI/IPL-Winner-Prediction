#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT_DIR / "production_model" / "scripts"
PHASE41_DIR = ROOT_DIR / "phases" / "phase_4_1"

for path in [SCRIPTS_DIR, PHASE41_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from auto_predict_trigger import HistoricalFeatureBuilder, normalize_team_name, normalize_toss_choice  # type: ignore
from ops_db import default_db_path, fetch_recent_team_form, fetch_recent_venue_stats, init_db  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare production model vs phase 4.1 on Cricbuzz-scraped data")
    parser.add_argument(
        "--input",
        default=str(ROOT_DIR / "data" / "raw" / "cricbuzz_current_matches.csv"),
        help="Scraped CSV from current_scraper.py or historical_scraper.py",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "comparison_table.csv"),
        help="Comparison table CSV to upsert",
    )
    return parser.parse_args()


def load_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    except Exception:
        pass
    return [item.strip() for item in text.split("|") if item.strip()]


def parse_actual_winner(row: pd.Series) -> Optional[str]:
    fallback_value: Optional[str] = None
    for key in ["winner", "result_text", "actual_winner"]:
        value = str(row.get(key, "") or "").strip()
        if value and value.lower() != "nan":
            match = re.search(r"News\s+([A-Za-z .&-]+?)\s+won by", value, flags=re.I)
            if match:
                return normalize_team_name(match.group(1).strip())

            match = re.search(r"([A-Za-z .&-]+?)\s+won by", value, flags=re.I)
            if match:
                candidate = match.group(1).strip()
                if "News" in candidate:
                    candidate = candidate.split("News")[-1].strip()
                return normalize_team_name(candidate)

            if fallback_value is None:
                fallback_value = value

    text = str(row.get("result_text", "") or "")
    match = re.search(r"News\s+([A-Za-z .&-]+?)\s+won by", text, flags=re.I)
    if match:
        return normalize_team_name(match.group(1).strip())
    match = re.search(r"([A-Za-z .&-]+?) won by [A-Za-z0-9 .&-]+", text, flags=re.I)
    if match:
        return normalize_team_name(match.group(1).strip())
    return normalize_team_name(fallback_value) if fallback_value else None


def parse_toss(row: pd.Series, team1: str, team2: str) -> tuple[str, str]:
    text = str(row.get("toss_text", "") or row.get("toss", "") or "").strip()
    match = re.search(r"([A-Za-z .&-]+?) elected to (bat|field)", text, flags=re.I)
    if match:
        return normalize_team_name(match.group(1).strip()), match.group(2).lower()

    toss_winner = str(row.get("toss_winner", "") or "").strip()
    toss_decision = str(row.get("toss_decision", "") or "").strip().lower()
    if toss_winner:
        return normalize_team_name(toss_winner), normalize_toss_choice(toss_decision)

    return team1, "field"


def parse_reference_dt(row: pd.Series) -> pd.Timestamp:
    for key in ["dateTimeGMT", "match_date", "scraped_at_utc"]:
        value = row.get(key)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        ts = pd.to_datetime(value, errors="coerce", utc=True)
        if not pd.isna(ts):
            return pd.Timestamp(ts)
    return pd.Timestamp(datetime.now(timezone.utc))


def parse_players(cell: Any) -> list[str]:
    players = load_text_list(cell)
    players = players[:11]
    while len(players) < 11:
        players.append(f"Unknown_Player_{len(players) + 1}")
    return players


def normalize_prediction(prediction: Any, team1: str, team2: str) -> str:
    text = str(prediction).strip()
    if text in {team1, team2}:
        return text
    if text in {"0", "0.0", "False", "false"}:
        return team2
    if text in {"1", "1.0", "True", "true"}:
        return team1
    return normalize_team_name(text)


def unwrap_phase41_artifact(artifact: Any) -> tuple[Any, Any, Any]:
    if isinstance(artifact, dict):
        if "model_pipeline" in artifact:
            return artifact["model_pipeline"], artifact.get("label_encoder"), artifact.get("feature_columns")
        if "model" in artifact:
            nested = artifact["model"]
            if isinstance(nested, dict) and "pipeline" in nested:
                return nested["pipeline"], artifact.get("label_encoder"), artifact.get("feature_columns")
            return nested, artifact.get("label_encoder"), artifact.get("feature_columns")
    return artifact, None, getattr(artifact, "feature_names_in_", None)


def build_production_features(
    base_dir: Path,
    builder: HistoricalFeatureBuilder,
    model: Any,
    row: pd.Series,
) -> tuple[pd.DataFrame, str, str, str, str]:
    team1 = builder.map_team(normalize_team_name(str(row.get("team1", "") or "")))
    team2 = builder.map_team(normalize_team_name(str(row.get("team2", "") or "")))
    stadium = builder.map_stadium(str(row.get("venue", "Unknown") or "Unknown"))
    toss_winner, toss_decision = parse_toss(row, team1, team2)
    ref_dt = parse_reference_dt(row).to_pydatetime()

    t1_form = builder.recent_form(team1, ref_dt)
    t2_form = builder.recent_form(team2, ref_dt)

    db_path = default_db_path(base_dir)
    init_db(db_path)
    t1_recent = fetch_recent_team_form(db_path, team1, n=5)
    t2_recent = fetch_recent_team_form(db_path, team2, n=5)
    if t1_recent is not None:
        t1_form = 0.7 * t1_form + 0.3 * t1_recent
    if t2_recent is not None:
        t2_form = 0.7 * t2_form + 0.3 * t2_recent

    venue_profile = builder.venue_profile(stadium)
    venue_recent = fetch_recent_venue_stats(db_path, stadium)
    if venue_recent is not None:
        recent_score, recent_toss_adv = venue_recent
        venue_profile["venue_score_prior"] = 0.8 * venue_profile["venue_score_prior"] + 0.2 * recent_score
        venue_profile["toss_advantage"] = 0.8 * venue_profile["toss_advantage"] + 0.2 * recent_toss_adv

    t1_profile = builder.team_profile(team1)
    t2_profile = builder.team_profile(team2)
    is_high_dew = 1 if (toss_decision == "field" and venue_profile["venue_chase_winrate_prior"] >= 0.52) else 0

    feature_row = {
        "team1_form_winrate_5": t1_form,
        "team2_form_winrate_5": t2_form,
        "venue_score_prior": venue_profile["venue_score_prior"],
        "venue_chase_winrate_prior": venue_profile["venue_chase_winrate_prior"],
        "toss_advantage": venue_profile["toss_advantage"],
        "is_high_dew": is_high_dew,
        "t1_bat_avg": t1_profile["bat_avg"],
        "t1_bat_sr": t1_profile["bat_sr"],
        "t1_bowl_eco": t1_profile["bowl_eco"],
        "t1_bowl_sr": t1_profile["bowl_sr"],
        "t2_bat_avg": t2_profile["bat_avg"],
        "t2_bat_sr": t2_profile["bat_sr"],
        "t2_bowl_eco": t2_profile["bowl_eco"],
        "t2_bowl_sr": t2_profile["bowl_sr"],
        "pitch_type": venue_profile["pitch_type"],
        "bounce_and_carry": venue_profile["bounce_and_carry"],
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "team1": team1,
        "team2": team2,
        "stadium": stadium,
    }

    X = pd.DataFrame([feature_row])
    X = X[model.feature_names_]
    return X, team1, team2, stadium, toss_winner


def build_phase41_features(row: pd.Series) -> tuple[pd.DataFrame, str, str, str, str]:
    team1 = normalize_team_name(str(row.get("team1", "") or ""))
    team2 = normalize_team_name(str(row.get("team2", "") or ""))
    venue = str(row.get("venue", "Unknown") or "Unknown")
    toss_winner, toss_decision = parse_toss(row, team1, team2)
    team1_xi = parse_players(row.get("team1_xi"))
    team2_xi = parse_players(row.get("team2_xi"))

    feature_row = {
        "Team1": team1,
        "Team2": team2,
        "Toss_Winner": toss_winner,
        "Toss_Decision": toss_decision,
        "team1_form_winrate_5": 0.5,
        "team2_form_winrate_5": 0.5,
        "venue_chase_winrate_prior": 0.5,
        "venue_score_prior": 170.0,
        "h2h_team1_winrate_prior": 0.5,
        "venue_team1_winrate_prior": 0.5,
        "venue_team2_winrate_prior": 0.5,
        "venue_avg_first_innings_runs_prior": 170.0,
        "team1_recent_runs_for_5": 160.0,
        "team2_recent_runs_for_5": 160.0,
        "team1_recent_runs_against_5": 160.0,
        "team2_recent_runs_against_5": 160.0,
        "team1_recent_wkts_taken_5": 7.0,
        "team2_recent_wkts_taken_5": 7.0,
        "team1_recent_powerplay_rr_5": 8.0,
        "team2_recent_powerplay_rr_5": 8.0,
        "team1_recent_death_rr_5": 10.0,
        "team2_recent_death_rr_5": 10.0,
        "team1_player_elo_avg_prior": 1000.0,
        "team2_player_elo_avg_prior": 1000.0,
        "team1_player_elo_max_prior": 1150.0,
        "team2_player_elo_max_prior": 1150.0,
        "team1_player_elo_min_prior": 850.0,
        "team2_player_elo_min_prior": 850.0,
        "player_elo_gap_prior": 0.0,
        "Team1_Player_1": team1_xi[0],
        "Team1_Player_2": team1_xi[1],
        "Team1_Player_3": team1_xi[2],
        "Team1_Player_4": team1_xi[3],
        "Team1_Player_5": team1_xi[4],
        "Team1_Player_6": team1_xi[5],
        "Team1_Player_7": team1_xi[6],
        "Team1_Player_8": team1_xi[7],
        "Team1_Player_9": team1_xi[8],
        "Team1_Player_10": team1_xi[9],
        "Team1_Player_11": team1_xi[10],
        "Team2_Player_1": team2_xi[0],
        "Team2_Player_2": team2_xi[1],
        "Team2_Player_3": team2_xi[2],
        "Team2_Player_4": team2_xi[3],
        "Team2_Player_5": team2_xi[4],
        "Team2_Player_6": team2_xi[5],
        "Team2_Player_7": team2_xi[6],
        "Team2_Player_8": team2_xi[7],
        "Team2_Player_9": team2_xi[8],
        "Team2_Player_10": team2_xi[9],
        "Team2_Player_11": team2_xi[10],
    }

    X = pd.DataFrame([feature_row])
    return X, team1, team2, venue, toss_winner


def predict_production(base_dir: Path, builder: HistoricalFeatureBuilder, model: Any, row: pd.Series) -> dict[str, Any]:
    X, team1, team2, venue, toss_winner = build_production_features(base_dir, builder, model, row)
    proba = model.predict_proba(X)[0]
    class_1_index = list(model.classes_).index(1)
    team1_prob = float(proba[class_1_index])
    predicted_winner = team1 if team1_prob >= 0.5 else team2
    confidence = team1_prob if team1_prob >= 0.5 else (1.0 - team1_prob)
    return {
        "prediction": predicted_winner,
        "confidence": round(confidence, 4),
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "toss_winner": toss_winner,
    }


def predict_phase41(model: Any, label_encoder: Any, row: pd.Series) -> dict[str, Any]:
    X, team1, team2, venue, toss_winner = build_phase41_features(row)
    raw_prediction = model.predict(X)[0]
    predicted_winner = normalize_prediction(raw_prediction, team1, team2)
    if label_encoder is not None:
        try:
            decoded = label_encoder.inverse_transform([int(raw_prediction)])[0]
            predicted_winner = normalize_team_name(str(decoded))
        except Exception:
            pass
    try:
        confidence = float(np.max(model.predict_proba(X)[0]))
    except Exception:
        confidence = 0.0
    return {
        "prediction": predicted_winner,
        "confidence": round(confidence, 4),
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "toss_winner": toss_winner,
    }


def upsert_table(table_path: Path, row: dict[str, Any]) -> None:
    table_path.parent.mkdir(parents=True, exist_ok=True)
    if table_path.exists() and table_path.stat().st_size > 0:
        current = pd.read_csv(table_path)
        if "match_id" in current.columns:
            current = current[current["match_id"].astype(str) != str(row["match_id"])]
        updated = pd.concat([current, pd.DataFrame([row])], ignore_index=True)
    else:
        updated = pd.DataFrame([row])

    sort_cols = [col for col in ["match_id", "updated_at_utc"] if col in updated.columns]
    if sort_cols:
        updated = updated.sort_values(sort_cols).reset_index(drop=True)
    updated.to_csv(table_path, index=False)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input not found: {input_path}")
        return 2

    df = pd.read_csv(input_path)
    required = {"match_id", "team1", "team2", "venue"}
    missing = required - set(df.columns)
    if missing:
        print(f"Missing required columns: {sorted(missing)}")
        return 2

    builder = HistoricalFeatureBuilder(ROOT_DIR / "production_model" / "data" / "training_data.csv")
    production_model = joblib.load(ROOT_DIR / "production_model" / "model" / "model.joblib")
    phase41_artifact = joblib.load(PHASE41_DIR / "artifacts" / "phase41_ipl_winner_best_pipeline.joblib")
    phase41_model, phase41_label_encoder, phase41_feature_columns = unwrap_phase41_artifact(phase41_artifact)

    rows: list[dict[str, Any]] = []
    for _, source_row in df.iterrows():
        production = predict_production(ROOT_DIR / "production_model", builder, production_model, source_row)
        phase41 = predict_phase41(phase41_model, phase41_label_encoder, source_row)
        actual_winner = parse_actual_winner(source_row)

        match_id = str(source_row.get("match_id", "")).strip()
        match_name = str(source_row.get("match_title", "") or source_row.get("match_name", "") or "").strip()
        if not match_name:
            match_name = f"{production['team1']} vs {production['team2']}"

        output_row = {
            "match_id": match_id,
            "match_name": match_name,
            "team1": production["team1"],
            "team2": production["team2"],
            "venue": production["venue"],
            "toss_winner": production["toss_winner"],
            "toss_decision": parse_toss(source_row, production["team1"], production["team2"])[1],
            "actual_winner": actual_winner,
            "production_prediction": production["prediction"],
            "production_confidence": production["confidence"],
            "phase41_prediction": phase41["prediction"],
            "phase41_confidence": phase41["confidence"],
            "production_correct": bool(actual_winner and production["prediction"] == actual_winner),
            "phase41_correct": bool(actual_winner and phase41["prediction"] == actual_winner),
            "winner_agreement": bool(production["prediction"] == phase41["prediction"]),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_csv": str(input_path),
        }
        rows.append(output_row)
        upsert_table(output_path, output_row)

    summary = pd.DataFrame(rows)
    print(f"Saved comparison table to {output_path}")
    print(summary[["match_id", "production_prediction", "phase41_prediction", "actual_winner"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())