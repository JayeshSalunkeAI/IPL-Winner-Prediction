import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


def run_health_check(base_dir: Path) -> dict:
    model_path = base_dir / "model" / "model.joblib"
    data_path = base_dir / "data" / "training_data.csv"
    metadata_path = base_dir / "model" / "metadata.json"

    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].dt.year >= 2020].copy()

    feature_names = model.feature_names_
    X = df[feature_names].copy()
    y = df["target"].astype(int)

    cat_cols = ["pitch_type", "bounce_and_carry", "toss_winner", "toss_decision", "team1", "team2", "stadium"]
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].fillna("Unknown").astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    optimistic_accuracy = float(accuracy_score(y_test, preds))
    optimistic_auc = float(roc_auc_score(y_test, probs))

    result = {
        "samples_total": int(len(df)),
        "samples_test": int(len(X_test)),
        "recheck_accuracy_optimistic": round(optimistic_accuracy, 4),
        "recheck_roc_auc_optimistic": round(optimistic_auc, 4),
        "note": "Recheck is optimistic because model was trained on full dataset. Use metadata validation metrics as primary.",
    }

    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            md = json.load(f)
        md_acc = float(md.get("validation_accuracy", 0.0) or 0.0)
        md_auc = float(md.get("roc_auc", 0.0) or 0.0)
        result["metadata_validation_accuracy"] = md_acc
        result["metadata_roc_auc"] = md_auc
        result["status"] = "good" if md_acc >= 0.65 else "needs_attention"
    else:
        result["status"] = "unknown"

    return result


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    result = run_health_check(base_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
