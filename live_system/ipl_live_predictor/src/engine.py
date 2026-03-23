from __future__ import annotations

from typing import Any

from config import settings
from data_provider import DummyAPIProvider
from model_runtime import ModelRuntime
from state_engine import build_pre_match_features, update_post_match_state
from storage import connect, get_match, init_db, mark_match_final, upsert_match_prediction


def run_once() -> dict[str, int]:
    provider = DummyAPIProvider(settings.dummy_feed_path)
    runtime = ModelRuntime(settings.model_path)

    conn = connect(settings.db_path)
    init_db(conn)

    payload = provider.fetch()
    matches = payload.get("matches", [])

    predicted_count = 0
    finalized_count = 0

    for item in matches:
        pre_match = item.get("pre_match") or {}
        post_match = item.get("post_match") or None
        status = (item.get("status") or "upcoming").lower()

        match_id = str(pre_match.get("match_id") or "")
        if not match_id:
            continue

        row = get_match(conn, match_id)

        # 1) Predict when upcoming/live and no prediction yet.
        if status in {"upcoming", "live"} and (row is None or row["status"] not in {"predicted", "final"}):
            feature_row = build_pre_match_features(conn, pre_match, runtime.feature_columns)
            pred_winner, confidence, probs = runtime.predict(feature_row)

            upsert_match_prediction(
                conn=conn,
                match_id=match_id,
                match_date=str(pre_match.get("date", "")),
                venue=str(pre_match.get("venue", "Unknown")),
                team1=str(pre_match.get("team1", "")),
                team2=str(pre_match.get("team2", "")),
                toss_winner=str(pre_match.get("toss_winner", "")),
                toss_decision=str(pre_match.get("toss_decision", "")),
                predicted_winner=pred_winner,
                confidence=confidence,
                probabilities=probs,
            )
            predicted_count += 1

        # 2) Finalize + update state when result appears.
        if status == "completed" and post_match is not None:
            existing = get_match(conn, match_id)
            if existing is None or existing["status"] != "final":
                # Ensure prediction row exists before finalization.
                if existing is None:
                    feature_row = build_pre_match_features(conn, pre_match, runtime.feature_columns)
                    pred_winner, confidence, probs = runtime.predict(feature_row)
                    upsert_match_prediction(
                        conn=conn,
                        match_id=match_id,
                        match_date=str(pre_match.get("date", "")),
                        venue=str(pre_match.get("venue", "Unknown")),
                        team1=str(pre_match.get("team1", "")),
                        team2=str(pre_match.get("team2", "")),
                        toss_winner=str(pre_match.get("toss_winner", "")),
                        toss_decision=str(pre_match.get("toss_decision", "")),
                        predicted_winner=pred_winner,
                        confidence=confidence,
                        probabilities=probs,
                    )
                    predicted_count += 1

                update_post_match_state(conn, pre_match, post_match)
                mark_match_final(conn, match_id, str(post_match.get("actual_winner", "Unknown")))
                finalized_count += 1

    conn.close()
    return {"predicted": predicted_count, "finalized": finalized_count, "total_feed_matches": len(matches)}
