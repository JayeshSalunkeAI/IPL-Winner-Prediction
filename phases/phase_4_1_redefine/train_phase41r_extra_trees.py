from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Phase 4.1 redefine (ExtraTrees focused)")
    p.add_argument("--input", default="phases/phase_4/results/phase4_dataset.csv")
    p.add_argument("--player-ratings", required=True)
    p.add_argument("--years", nargs="+", type=int, default=[2020, 2021, 2023, 2024, 2025])
    p.add_argument("--results-dir", default="phases/phase_4_1_redefine/results")
    p.add_argument("--artifacts-dir", default="phases/phase_4_1_redefine/artifacts")
    p.add_argument("--drop-no-result", action="store_true")
    return p.parse_args()


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=8, sparse_output=True)),
        ]
    )
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("cat", cat_pipe, cat_cols), ("num", num_pipe, num_cols)],
        remainder="drop",
    )
    return preprocessor, cat_cols, num_cols


def enrich_player_elo_features(df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
    player_cols_t1 = [f"Team1_Player_{i}" for i in range(1, 12)]
    player_cols_t2 = [f"Team2_Player_{i}" for i in range(1, 12)]

    rating_map = {
        normalize_name(r["player_name"]): float(r["player_elo_like"])
        for _, r in ratings_df.iterrows()
    }
    default_rating = float(np.median(list(rating_map.values()))) if rating_map else 1000.0

    def team_stats(row: pd.Series, cols: list[str]) -> tuple[float, float, float]:
        vals: list[float] = []
        for c in cols:
            key = normalize_name(row.get(c, ""))
            vals.append(rating_map.get(key, default_rating))
        arr = np.array(vals, dtype=float)
        return float(arr.mean()), float(arr.max()), float(arr.min())

    t1 = df.apply(lambda r: team_stats(r, player_cols_t1), axis=1)
    t2 = df.apply(lambda r: team_stats(r, player_cols_t2), axis=1)

    df = df.copy()
    df["team1_player_elo_avg_prior"] = [x[0] for x in t1]
    df["team1_player_elo_max_prior"] = [x[1] for x in t1]
    df["team1_player_elo_min_prior"] = [x[2] for x in t1]

    df["team2_player_elo_avg_prior"] = [x[0] for x in t2]
    df["team2_player_elo_max_prior"] = [x[1] for x in t2]
    df["team2_player_elo_min_prior"] = [x[2] for x in t2]

    df["player_elo_gap_prior"] = df["team1_player_elo_avg_prior"] - df["team2_player_elo_avg_prior"]
    return df


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    ratings_path = Path(args.player_ratings)
    results_dir = Path(args.results_dir)
    artifacts_dir = Path(args.artifacts_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    ratings_df = pd.read_csv(ratings_path)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Match_Winner"]).copy()
    df = df[df["Date"].dt.year.isin(set(args.years))].copy()

    if args.drop_no_result:
        df = df[df["Match_Winner"] != "Draw/No Result"].copy()

    df = enrich_player_elo_features(df, ratings_df)

    drop_cols = ["Match_ID", "Date", "Teams", "Match_Winner"]
    feature_columns = [c for c in df.columns if c not in drop_cols]

    X = df[feature_columns].copy()
    y_raw = df["Match_Winner"].astype(str).values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor, _, _ = build_preprocessor(X_train)
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                ExtraTreesClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    param_grid = {
        "model__n_estimators": [350, 450, 600],
        "model__max_depth": [10, 14, None],
        "model__min_samples_leaf": [2, 4, 6],
        "model__min_samples_split": [4, 8, 12],
        "model__max_features": ["sqrt", 0.35, 0.5],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_grid,
        n_iter=24,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_

    train_pred = best.predict(X_train)
    test_pred = best.predict(X_test)

    metrics = {
        "train_accuracy": float(accuracy_score(y_train, train_pred)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "train_f1_weighted": float(f1_score(y_train, train_pred, average="weighted", zero_division=0)),
        "test_f1_weighted": float(f1_score(y_test, test_pred, average="weighted", zero_division=0)),
        "best_cv_accuracy": float(search.best_score_),
        "best_params": search.best_params_,
        "training_years": args.years,
        "rows_used": int(len(df)),
    }

    report = classification_report(
        y_test,
        test_pred,
        labels=list(range(len(label_encoder.classes_))),
        target_names=label_encoder.classes_.tolist(),
        output_dict=True,
        zero_division=0,
    )

    model_payload = {
        "model_pipeline": best,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
    }

    joblib.dump(model_payload, artifacts_dir / "phase41r_extratrees_pipeline.joblib")
    with open(artifacts_dir / "phase41r_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "phase": "4.1_redefine",
                "model": "extra_trees_gen",
                "feature_count": len(feature_columns),
                "feature_columns": feature_columns,
                "training_years": args.years,
                "metrics": metrics,
            },
            f,
            indent=2,
        )

    with open(results_dir / "phase41r_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(results_dir / "phase41r_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Saved model:", artifacts_dir / "phase41r_extratrees_pipeline.joblib")
    print("Saved metadata:", artifacts_dir / "phase41r_model_metadata.json")
    print("Saved metrics:", results_dir / "phase41r_metrics.json")
    print("Test accuracy:", round(metrics["test_accuracy"], 4))


if __name__ == "__main__":
    main()
