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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Phase 4.1 redefine (ExtraTrees focused)")
    p.add_argument("--input", default="phases/phase_4/results/phase4_dataset.csv")
    p.add_argument("--player-ratings", required=True)
    p.add_argument("--train-years", nargs="+", type=int, default=[2021, 2022, 2023, 2024, 2025])
    p.add_argument("--test-year", type=int, default=2026)
    p.add_argument(
        "--eval-comparison-csv",
        default="production_model/Model Comparison/comparison_table.csv",
        help="Fallback 2026 evaluation source when --input has no rows for --test-year",
    )
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


def build_2026_eval_from_comparison(
    csv_path: Path,
    feature_columns: list[str],
    test_year: int,
) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    required = {"team1", "team2", "actual_winner"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()

    winners = df["actual_winner"].fillna("").astype(str).str.strip()
    mask_done = winners != ""
    if "match_name" in df.columns:
        mask_year = df["match_name"].fillna("").astype(str).str.contains(str(test_year), case=False)
    else:
        mask_year = pd.Series([True] * len(df), index=df.index)

    df_eval = df.loc[mask_done & mask_year].copy()
    if df_eval.empty:
        return pd.DataFrame()

    neutral_defaults: dict[str, float] = {
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

    rows: list[dict[str, Any]] = []
    for _, r in df_eval.iterrows():
        team1 = str(r.get("team1", "")).strip()
        team2 = str(r.get("team2", "")).strip()
        toss_winner = str(r.get("toss_winner", "")).strip() or team1
        toss_decision = str(r.get("toss_decision", "field")).strip().lower() or "field"
        if toss_decision not in {"bat", "field"}:
            toss_decision = "field"

        row: dict[str, Any] = {
            "Team1": team1,
            "Team2": team2,
            "Toss_Winner": toss_winner,
            "Toss_Decision": toss_decision,
            "Match_Winner": str(r.get("actual_winner", "")).strip(),
        }

        for col, val in neutral_defaults.items():
            row[col] = val

        for i in range(1, 12):
            row[f"Team1_Player_{i}"] = f"{team1}_Player_{i}" if team1 else f"Unknown_Team1_Player_{i}"
            row[f"Team2_Player_{i}"] = f"{team2}_Player_{i}" if team2 else f"Unknown_Team2_Player_{i}"

        # Keep only features expected by the trained pipeline + target column.
        compact = {c: row.get(c, np.nan) for c in feature_columns}
        compact["Match_Winner"] = row["Match_Winner"]
        rows.append(compact)

    return pd.DataFrame(rows)


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

    if args.drop_no_result:
        df = df[df["Match_Winner"] != "Draw/No Result"].copy()

    df = enrich_player_elo_features(df, ratings_df)

    train_years = sorted(set(args.train_years))
    train_df = df[df["Date"].dt.year.isin(set(train_years))].copy()
    test_df = df[df["Date"].dt.year == int(args.test_year)].copy()

    drop_cols = ["Match_ID", "Date", "Teams", "Match_Winner"]
    feature_columns = [c for c in train_df.columns if c not in drop_cols]

    if train_df.empty:
        raise ValueError(f"No training rows found for years: {train_years}")

    # If phase4 dataset has no 2026 rows, evaluate on completed 2026 matches from comparison table.
    test_source = "phase4_dataset"
    if test_df.empty:
        test_df = build_2026_eval_from_comparison(Path(args.eval_comparison_csv), feature_columns, int(args.test_year))
        test_source = "comparison_table_2026_completed"

    if test_df.empty:
        raise ValueError(
            f"No test rows found for {args.test_year} in input dataset or fallback comparison table: {args.eval_comparison_csv}"
        )

    X_train = train_df[feature_columns].copy()
    y_train_raw = train_df["Match_Winner"].astype(str)

    X_test = test_df[feature_columns].copy()
    y_test_raw = test_df["Match_Winner"].astype(str)

    label_encoder = LabelEncoder()
    label_encoder.fit(y_train_raw.values)

    y_train = label_encoder.transform(y_train_raw.values)

    known_mask = y_test_raw.isin(label_encoder.classes_)
    X_test = X_test.loc[known_mask].reset_index(drop=True)
    y_test_raw = y_test_raw.loc[known_mask].reset_index(drop=True)
    if y_test_raw.empty:
        raise ValueError("Test rows found, but none have winners present in training label space")
    y_test = label_encoder.transform(y_test_raw.values)

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
        "training_years": train_years,
        "test_year": int(args.test_year),
        "test_source": test_source,
        "rows_train": int(len(train_df)),
        "rows_test": int(len(X_test)),
        "rows_total_after_filters": int(len(df)),
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
                "training_years": train_years,
                "test_year": int(args.test_year),
                "test_source": test_source,
                "metrics": metrics,
            },
            f,
            indent=2,
        )

    test_pred_labels = label_encoder.inverse_transform(test_pred)
    test_preds_df = pd.DataFrame(
        {
            "actual_winner": y_test_raw,
            "predicted_winner": test_pred_labels,
            "correct": y_test_raw.values == test_pred_labels,
        }
    )
    test_preds_df.to_csv(results_dir / "phase41r_test_predictions.csv", index=False)

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
