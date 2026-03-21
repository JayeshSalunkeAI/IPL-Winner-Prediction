from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


ARTIFACT_PATH = Path("phase_3_artifacts/phase3_ipl_winner_best_pipeline.pkl")


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(..., description="Feature dictionary for one match")


def load_bundle(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found at {path}")
    with path.open("rb") as f:
        return pickle.load(f)


bundle = load_bundle(ARTIFACT_PATH)
model = bundle["model_pipeline"]
label_encoder = bundle["label_encoder"]
feature_columns: list[str] = bundle["feature_columns"]
metadata: dict[str, Any] = bundle.get("metadata", {})

app = FastAPI(
    title="IPL Winner Predictor - Phase 3",
    version="3.0.0",
    description="Prediction API using regularized player-aware Phase 3 model.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/meta")
def meta() -> dict[str, Any]:
    return {
        "phase": metadata.get("phase", 3),
        "best_model": metadata.get("best_model"),
        "models_evaluated": metadata.get("models_evaluated", []),
        "feature_count": len(feature_columns),
    }


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict[str, Any]:
    row = payload.features
    missing = [c for c in feature_columns if c not in row]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required features: {missing}")

    input_df = pd.DataFrame([row])[feature_columns]
    pred_encoded = int(model.predict(input_df)[0])
    probs = model.predict_proba(input_df)[0]

    winner = str(label_encoder.inverse_transform([pred_encoded])[0])
    winner_prob = float(probs[pred_encoded])

    top_idx = probs.argsort()[::-1][:5]
    ranked = [
        {
            "team": str(label_encoder.inverse_transform([int(i)])[0]),
            "probability": float(probs[int(i)]),
        }
        for i in top_idx
    ]

    return {
        "predicted_winner": winner,
        "winning_probability": winner_prob,
        "top_probabilities": ranked,
    }
