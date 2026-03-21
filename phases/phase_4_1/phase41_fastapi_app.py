from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


CURRENT_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = CURRENT_DIR / "artifacts/phase41_ipl_winner_best_pipeline.pkl"

# Ensure custom transformer modules in this folder are importable during unpickling.
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


class PredictionRequest(BaseModel):
    features: dict[str, object] = Field(..., description="Feature dictionary for one match")


def load_bundle(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found at {path}")
    with path.open("rb") as f:
        return pickle.load(f)


bundle = load_bundle(ARTIFACT_PATH)
model = bundle["model_pipeline"]
label_encoder = bundle["label_encoder"]
feature_columns: list[str] = bundle["feature_columns"]
metadata: dict[str, object] = bundle.get("metadata", {})

app = FastAPI(title="IPL Winner Predictor - Phase 4.1", version="4.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/meta")
def meta() -> dict[str, object]:
    return {
        "phase": metadata.get("phase", "4.1"),
        "best_model": metadata.get("best_model"),
        "feature_count": len(feature_columns),
    }


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict[str, object]:
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
