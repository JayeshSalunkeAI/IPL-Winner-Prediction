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


def normalize_key(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def load_player_elo_map() -> tuple[dict[str, float], float]:
    candidates = [
        ROOT_DIR / "phases" / "phase_4_1_redefine" / "data" / "player_performance_ratings_2020_2026.csv",
        ROOT_DIR / "phases" / "phase_4_1_redefine" / "data" / "player_performance_ratings_2021_2026.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            ratings_df = pd.read_csv(path)
            rating_map = {
                normalize_key(row.get("player_name", "")): float(row.get("player_elo_like", 1000.0))
                for _, row in ratings_df.iterrows()
                if str(row.get("player_name", "")).strip()
            }
            values = list(rating_map.values())
            default_rating = float(np.median(values)) if values else 1000.0
            if rating_map:
                return rating_map, default_rating
        except Exception:
            continue
    return {}, 1000.0


def safe_winrate(mask: pd.Series) -> float:
    if len(mask) == 0:
        return 0.5
    return float(mask.mean())


def recent_team_form(history: pd.DataFrame, team: str, n: int = 5) -> float:
    team_matches = history[(history["team1"] == team) | (history["team2"] == team)]
    if team_matches.empty:
        return 0.5
    recent = team_matches.tail(n)
    return safe_winrate(recent["actual_winner"] == team)


def h2h_team1_winrate(history: pd.DataFrame, team1: str, team2: str) -> float:
    h2h = history[
        ((history["team1"] == team1) & (history["team2"] == team2))
        | ((history["team1"] == team2) & (history["team2"] == team1))
    ]
    if h2h.empty:
        return 0.5
    return safe_winrate(h2h["actual_winner"] == team1)


def venue_team_winrate(history: pd.DataFrame, venue: str, team: str) -> float:
    venue_matches = history[history["venue"] == venue]
    if venue_matches.empty:
        return 0.5
    team_venue = venue_matches[(venue_matches["team1"] == team) | (venue_matches["team2"] == team)]
    if team_venue.empty:
        return 0.5
    return safe_winrate(team_venue["actual_winner"] == team)


def venue_chase_winrate(history: pd.DataFrame, venue: str) -> float:
    venue_matches = history[history["venue"] == venue]
    if venue_matches.empty:
        return 0.5

    chasing_wins: list[bool] = []
    for _, r in venue_matches.iterrows():
        toss_winner = str(r.get("toss_winner", "") or "").strip()
        toss_decision = str(r.get("toss_decision", "") or "").strip().lower()
        winner = str(r.get("actual_winner", "") or "").strip()
        if not toss_winner or not winner or toss_decision not in {"bat", "field"}:
            continue
        team1 = str(r.get("team1", "") or "").strip()
        team2 = str(r.get("team2", "") or "").strip()
        if toss_decision == "field":
            chasing_team = toss_winner
        else:
            chasing_team = team2 if toss_winner == team1 else team1
        chasing_wins.append(winner == chasing_team)

    if not chasing_wins:
        return 0.5
    return float(np.mean(chasing_wins))


def compute_player_elo_stats(players: list[str], rating_map: dict[str, float], default_rating: float) -> tuple[float, float, float]:
    values = [rating_map.get(normalize_key(p), default_rating) for p in players]
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.max()), float(arr.min())


def load_phase41_history(table_path: Path) -> pd.DataFrame:
    if not table_path.exists() or table_path.stat().st_size == 0:
        return pd.DataFrame(columns=["match_id", "team1", "team2", "venue", "toss_winner", "toss_decision", "actual_winner"])

    df = pd.read_csv(table_path)
    required = {"match_id", "team1", "team2", "venue", "toss_winner", "toss_decision", "actual_winner"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame(columns=["match_id", "team1", "team2", "venue", "toss_winner", "toss_decision", "actual_winner"])

    out = df.copy()
    out["team1"] = out["team1"].astype(str).map(normalize_team_name)
    out["team2"] = out["team2"].astype(str).map(normalize_team_name)
    out["venue"] = out["venue"].astype(str).str.strip()
    out["toss_winner"] = out["toss_winner"].astype(str).map(normalize_team_name)
    out["toss_decision"] = out["toss_decision"].astype(str).str.strip().str.lower()
    out["actual_winner"] = out["actual_winner"].astype(str).map(normalize_team_name)
    out = out[out["actual_winner"].str.strip() != ""]

    out["match_id_num"] = pd.to_numeric(out["match_id"], errors="coerce")
    out = out.sort_values(["match_id_num", "updated_at_utc" if "updated_at_utc" in out.columns else "match_id"]).reset_index(drop=True)
    return out


def load_phase41_match_feature_store(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if "Match_ID" not in df.columns:
        return pd.DataFrame()
    df["Match_ID_num"] = pd.to_numeric(df["Match_ID"], errors="coerce")
    df = df.dropna(subset=["Match_ID_num"]).copy()
    return df


def phase41_stats_from_store(
    row: pd.Series,
    team1: str,
    team2: str,
    feature_store: pd.DataFrame,
) -> dict[str, float]:
    defaults = {
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
    }

    if feature_store.empty:
        return defaults

    mid = pd.to_numeric(str(row.get("match_id", "")).strip(), errors="coerce")
    if pd.isna(mid):
        return defaults

    subset = feature_store[feature_store["Match_ID_num"] == float(mid)].copy()
    if subset.empty:
        return defaults

    subset["Team1_norm"] = subset["Team1"].astype(str).map(normalize_team_name)
    subset["Team2_norm"] = subset["Team2"].astype(str).map(normalize_team_name)
    exact = subset[(subset["Team1_norm"] == team1) & (subset["Team2_norm"] == team2)]
    if exact.empty:
        exact = subset[(subset["Team1_norm"] == team2) & (subset["Team2_norm"] == team1)]
    if exact.empty:
        return defaults

    src = exact.iloc[-1]
    for k in list(defaults.keys()):
        if k in src and not pd.isna(src[k]):
            defaults[k] = float(src[k])
    return defaults


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


def decode_phase41_class(raw_class: Any, label_encoder: Any) -> str:
    if label_encoder is not None:
        try:
            decoded = label_encoder.inverse_transform([int(raw_class)])[0]
            return normalize_team_name(str(decoded))
        except Exception:
            pass
    return normalize_team_name(str(raw_class))


def fixture_team_probs(
    model: Any,
    X: pd.DataFrame,
    team1: str,
    team2: str,
    label_encoder: Any,
) -> tuple[Optional[float], Optional[float]]:
    try:
        proba = model.predict_proba(X)[0]
        classes = getattr(model, "classes_", None)
        if classes is None:
            return None, None
    except Exception:
        return None, None

    team_probs: dict[str, float] = {}
    for raw_class, p in zip(classes, proba):
        team_name = decode_phase41_class(raw_class, label_encoder)
        team_probs[team_name] = team_probs.get(team_name, 0.0) + float(p)

    return team_probs.get(team1), team_probs.get(team2)


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


def build_phase41_features(
    row: pd.Series,
    history: pd.DataFrame,
    player_elo_map: dict[str, float],
    default_player_elo: float,
    match_feature_store: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str, str, str]:
    team1 = normalize_team_name(str(row.get("team1", "") or ""))
    team2 = normalize_team_name(str(row.get("team2", "") or ""))
    venue = str(row.get("venue", "Unknown") or "Unknown")
    toss_winner, toss_decision = parse_toss(row, team1, team2)
    team1_xi = parse_players(row.get("team1_xi"))
    team2_xi = parse_players(row.get("team2_xi"))

    store_stats = phase41_stats_from_store(row, team1, team2, match_feature_store)

    team1_elo_avg, team1_elo_max, team1_elo_min = compute_player_elo_stats(team1_xi, player_elo_map, default_player_elo)
    team2_elo_avg, team2_elo_max, team2_elo_min = compute_player_elo_stats(team2_xi, player_elo_map, default_player_elo)

    if store_stats["team1_player_elo_avg_prior"] != 1000.0 or store_stats["team2_player_elo_avg_prior"] != 1000.0:
        team1_elo_avg = store_stats["team1_player_elo_avg_prior"]
        team2_elo_avg = store_stats["team2_player_elo_avg_prior"]
        team1_elo_max = store_stats["team1_player_elo_max_prior"]
        team2_elo_max = store_stats["team2_player_elo_max_prior"]
        team1_elo_min = store_stats["team1_player_elo_min_prior"]
        team2_elo_min = store_stats["team2_player_elo_min_prior"]

    feature_row = {
        "Team1": team1,
        "Team2": team2,
        "Toss_Winner": toss_winner,
        "Toss_Decision": toss_decision,
        "team1_form_winrate_5": store_stats["team1_form_winrate_5"] if not history.empty else 0.5,
        "team2_form_winrate_5": store_stats["team2_form_winrate_5"] if not history.empty else 0.5,
        "venue_chase_winrate_prior": store_stats["venue_chase_winrate_prior"] if not history.empty else 0.5,
        "venue_score_prior": store_stats["venue_score_prior"],
        "h2h_team1_winrate_prior": store_stats["h2h_team1_winrate_prior"] if not history.empty else 0.5,
        "venue_team1_winrate_prior": store_stats["venue_team1_winrate_prior"] if not history.empty else 0.5,
        "venue_team2_winrate_prior": store_stats["venue_team2_winrate_prior"] if not history.empty else 0.5,
        "venue_avg_first_innings_runs_prior": store_stats["venue_avg_first_innings_runs_prior"],
        "team1_recent_runs_for_5": store_stats["team1_recent_runs_for_5"],
        "team2_recent_runs_for_5": store_stats["team2_recent_runs_for_5"],
        "team1_recent_runs_against_5": store_stats["team1_recent_runs_against_5"],
        "team2_recent_runs_against_5": store_stats["team2_recent_runs_against_5"],
        "team1_recent_wkts_taken_5": store_stats["team1_recent_wkts_taken_5"],
        "team2_recent_wkts_taken_5": store_stats["team2_recent_wkts_taken_5"],
        "team1_recent_powerplay_rr_5": store_stats["team1_recent_powerplay_rr_5"],
        "team2_recent_powerplay_rr_5": store_stats["team2_recent_powerplay_rr_5"],
        "team1_recent_death_rr_5": store_stats["team1_recent_death_rr_5"],
        "team2_recent_death_rr_5": store_stats["team2_recent_death_rr_5"],
        "team1_player_elo_avg_prior": team1_elo_avg,
        "team2_player_elo_avg_prior": team2_elo_avg,
        "team1_player_elo_max_prior": team1_elo_max,
        "team2_player_elo_max_prior": team2_elo_max,
        "team1_player_elo_min_prior": team1_elo_min,
        "team2_player_elo_min_prior": team2_elo_min,
        "player_elo_gap_prior": team1_elo_avg - team2_elo_avg,
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


def predict_phase41(
    model: Any,
    label_encoder: Any,
    row: pd.Series,
    history: pd.DataFrame,
    player_elo_map: dict[str, float],
    default_player_elo: float,
    match_feature_store: pd.DataFrame,
) -> dict[str, Any]:
    X, team1, team2, venue, toss_winner = build_phase41_features(
        row,
        history,
        player_elo_map,
        default_player_elo,
        match_feature_store,
    )
    team1_prob, team2_prob = fixture_team_probs(model, X, team1, team2, label_encoder)

    if team1_prob is not None and team2_prob is not None and (team1_prob + team2_prob) > 0.0:
        total = team1_prob + team2_prob
        team1_norm = team1_prob / total
        team2_norm = team2_prob / total
        predicted_winner = team1 if team1_norm >= team2_norm else team2
        confidence = max(team1_norm, team2_norm)
    else:
        raw_prediction = model.predict(X)[0]
        decoded_prediction = decode_phase41_class(raw_prediction, label_encoder)
        if decoded_prediction in {team1, team2}:
            predicted_winner = decoded_prediction
            confidence = 0.5
        else:
            predicted_winner = normalize_prediction(raw_prediction, team1, team2)
            confidence = 0.5

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
    player_elo_map, default_player_elo = load_player_elo_map()
    match_feature_store = load_phase41_match_feature_store(ROOT_DIR / "production_model" / "data" / "phase41_match_feature_store.csv")

    history_all = load_phase41_history(output_path)

    rows: list[dict[str, Any]] = []
    for _, source_row in df.iterrows():
        current_match_id = pd.to_numeric(str(source_row.get("match_id", "")).strip(), errors="coerce")
        if pd.isna(current_match_id):
            history = history_all.copy()
        else:
            history = history_all[history_all["match_id_num"] < float(current_match_id)].copy()

        production = predict_production(ROOT_DIR / "production_model", builder, production_model, source_row)
        phase41 = predict_phase41(
            phase41_model,
            phase41_label_encoder,
            source_row,
            history,
            player_elo_map,
            default_player_elo,
            match_feature_store,
        )
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

        # Keep in-memory history aligned for subsequent rows in the same batch.
        if actual_winner:
            next_row = {
                "match_id": match_id,
                "team1": production["team1"],
                "team2": production["team2"],
                "venue": production["venue"],
                "toss_winner": production["toss_winner"],
                "toss_decision": parse_toss(source_row, production["team1"], production["team2"])[1],
                "actual_winner": actual_winner,
                "match_id_num": pd.to_numeric(match_id, errors="coerce"),
            }
            history_all = pd.concat([history_all, pd.DataFrame([next_row])], ignore_index=True)
            history_all = history_all.sort_values("match_id_num").reset_index(drop=True)

    summary = pd.DataFrame(rows)
    print(f"Saved comparison table to {output_path}")
    print(summary[["match_id", "production_prediction", "phase41_prediction", "actual_winner"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())