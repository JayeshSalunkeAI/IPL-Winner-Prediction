from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tqdm.auto import tqdm
from xgboost import XGBClassifier

from phase4_transforms import CardinalityReducer


RANDOM_STATE = 42
ROOT_DIR = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase 4 winner model with advanced features.")
    parser.add_argument("--input", default=str(ROOT_DIR / "phases/phase_4/results/phase4_dataset.csv"))
    parser.add_argument("--results-dir", default=str(ROOT_DIR / "phases/phase_4/results"))
    parser.add_argument("--artifacts-dir", default=str(ROOT_DIR / "phases/phase_4/artifacts"))
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--drop-no-result", action="store_true", help="Drop Draw/No Result rows")
    parser.add_argument("--min-category-freq", type=int, default=8)
    parser.add_argument("--max-player-categories", type=int, default=90)
    return parser.parse_args()


def _param_space_size(grid: dict[str, list[Any]]) -> int:
    size = 1
    for values in grid.values():
        size *= max(1, len(values))
    return size


def top_k_accuracy(y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    topk = np.argsort(proba, axis=1)[:, -k:]
    hits = [int(y in row) for y, row in zip(y_true, topk)]
    return float(np.mean(hits))


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    conf = np.max(proba, axis=1)
    pred = np.argmax(proba, axis=1)
    correct = (pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        mask = (conf >= bins[i]) & (conf < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))

    return float(ece)


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=7,
                    sparse_output=True,
                ),
            ),
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


def evaluate_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    n_classes: int,
) -> dict[str, float]:
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_proba = model.predict_proba(X_train)
    test_proba = model.predict_proba(X_test)

    out = {
        "train_accuracy": float(accuracy_score(y_train, train_pred)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "train_balanced_accuracy": float(balanced_accuracy_score(y_train, train_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, test_pred)),
        "train_f1_weighted": float(f1_score(y_train, train_pred, average="weighted", zero_division=0)),
        "test_f1_weighted": float(f1_score(y_test, test_pred, average="weighted", zero_division=0)),
        "test_f1_macro": float(f1_score(y_test, test_pred, average="macro", zero_division=0)),
        "test_precision_weighted": float(precision_score(y_test, test_pred, average="weighted", zero_division=0)),
        "test_recall_weighted": float(recall_score(y_test, test_pred, average="weighted", zero_division=0)),
        "test_log_loss": float(log_loss(y_test, test_proba, labels=list(range(n_classes)))),
        "top2_accuracy": top_k_accuracy(y_test, test_proba, k=2),
        "top3_accuracy": top_k_accuracy(y_test, test_proba, k=3),
        "calibration_ece": expected_calibration_error(y_test, test_proba),
    }
    out["fit_gap_weighted_f1"] = out["train_f1_weighted"] - out["test_f1_weighted"]
    return out


def compute_drift_report(train_df: pd.DataFrame, test_df: pd.DataFrame, player_slot_cols: list[str]) -> dict[str, Any]:
    numeric_cols = [
        "team1_form_winrate_5",
        "team2_form_winrate_5",
        "venue_chase_winrate_prior",
        "venue_score_prior",
        "h2h_team1_winrate_prior",
        "team1_recent_runs_for_5",
        "team2_recent_runs_for_5",
        "team1_recent_runs_against_5",
        "team2_recent_runs_against_5",
        "team1_player_elo_avg_prior",
        "team2_player_elo_avg_prior",
    ]

    numeric_drift: dict[str, Any] = {}
    for col in numeric_cols:
        if col not in train_df.columns or col not in test_df.columns:
            continue
        tr = pd.to_numeric(train_df[col], errors="coerce").dropna()
        te = pd.to_numeric(test_df[col], errors="coerce").dropna()
        tr_mean = float(tr.mean()) if len(tr) else 0.0
        te_mean = float(te.mean()) if len(te) else 0.0
        tr_std = float(tr.std()) if len(tr) else 0.0
        shift_z = 0.0 if tr_std == 0 else (te_mean - tr_mean) / tr_std
        numeric_drift[col] = {
            "train_mean": round(tr_mean, 4),
            "test_mean": round(te_mean, 4),
            "mean_shift_z": round(float(shift_z), 4),
        }

    all_train_players = set()
    all_test_players = []
    per_col_unseen: dict[str, float] = {}

    for col in player_slot_cols:
        if col in train_df.columns:
            all_train_players.update(train_df[col].astype(str).unique().tolist())
        if col in test_df.columns:
            vals = test_df[col].astype(str)
            all_test_players.extend(vals.tolist())
            per_col_unseen[col] = round(float((~vals.isin(all_train_players)).mean()), 4)

    overall_unseen = 0.0
    if all_test_players:
        s = pd.Series(all_test_players, dtype="object")
        overall_unseen = float((~s.isin(all_train_players)).mean())

    return {
        "numeric_drift": numeric_drift,
        "lineup_drift": {
            "overall_unseen_player_rate": round(overall_unseen, 4),
            "per_player_slot_unseen_rate": per_col_unseen,
        },
    }


def save_plots(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    sns.set_theme(style="whitegrid")

    m = metrics_df.sort_values("test_accuracy", ascending=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=m, x="model", y="test_accuracy", hue="model", legend=False, palette="Set2")
    plt.title("Phase 4: Test Accuracy Comparison")
    plt.tight_layout()
    plt.savefig(out_dir / "phase4_model_comparison_accuracy.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(data=m, x="model", y="test_f1_weighted", hue="model", legend=False, palette="Blues")
    plt.title("Phase 4: Test Weighted F1 Comparison")
    plt.tight_layout()
    plt.savefig(out_dir / "phase4_model_comparison_f1.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(data=m, x="model", y="fit_gap_weighted_f1", hue="model", legend=False, palette="rocket")
    plt.title("Phase 4: Overfitting Gap")
    plt.tight_layout()
    plt.savefig(out_dir / "phase4_overfit_gap.png", dpi=160)
    plt.close()


def save_reliability_plot(y_test: np.ndarray, proba: np.ndarray, out_path: Path) -> None:
    conf = np.max(proba, axis=1)
    pred = np.argmax(proba, axis=1)
    correct = (pred == y_test).astype(float)

    bins = np.linspace(0.0, 1.0, 11)
    centers = []
    accs = []

    for i in range(10):
        mask = (conf >= bins[i]) & (conf < bins[i + 1])
        if mask.sum() == 0:
            continue
        centers.append((bins[i] + bins[i + 1]) / 2)
        accs.append(float(correct[mask].mean()))

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.plot(centers, accs, marker="o", label="Model")
    plt.xlabel("Confidence")
    plt.ylabel("Empirical accuracy")
    plt.title("Phase 4: Reliability Plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    results_dir = Path(args.results_dir)
    artifacts_dir = Path(args.artifacts_dir)

    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {input_path}. Run 'python phases/phase_4/phase4_extract_dataset.py' first."
        )

    df = pd.read_csv(input_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Match_Winner"]).sort_values(["Date", "Match_ID"]).reset_index(drop=True)

    if args.drop_no_result:
        df = df[df["Match_Winner"] != "Draw/No Result"].copy()

    test_mask = df["Date"].dt.year == args.test_year
    train_df = df.loc[~test_mask].copy()
    test_df = df.loc[test_mask].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train/Test split failed. Check test year and data coverage.")

    drop_cols = ["Match_Winner", "Match_ID", "Date", "Teams"]
    feature_columns = [c for c in train_df.columns if c not in drop_cols and c in df.columns]

    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()

    y_train_raw = train_df["Match_Winner"].astype(str).values
    y_test_raw = test_df["Match_Winner"].astype(str).values

    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate([y_train_raw, y_test_raw]))
    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    player_slot_cols = [c for c in feature_columns if c.startswith("Team1_Player_") or c.startswith("Team2_Player_")]

    preprocessor, _, _ = build_preprocessor(X_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    pipelines: dict[str, tuple[Pipeline, dict[str, list[Any]]]] = {
        "xgboost_phase4": (
            Pipeline(
                steps=[
                    (
                        "cardinality",
                        CardinalityReducer(
                            columns=player_slot_cols,
                            min_frequency=args.min_category_freq,
                            max_categories=args.max_player_categories,
                        ),
                    ),
                    ("preprocessor", preprocessor),
                    (
                        "model",
                        XGBClassifier(
                            objective="multi:softprob",
                            eval_metric="mlogloss",
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            {
                "model__n_estimators": [160, 220, 300],
                "model__max_depth": [3, 4],
                "model__learning_rate": [0.03, 0.05],
                "model__subsample": [0.65, 0.8],
                "model__colsample_bytree": [0.45, 0.6],
                "model__reg_lambda": [8.0, 12.0, 18.0],
                "model__reg_alpha": [2.5, 5.0, 8.0],
                "model__min_child_weight": [8, 12],
                "model__gamma": [1.0, 2.0],
            },
        ),
        "extra_trees_phase4": (
            Pipeline(
                steps=[
                    (
                        "cardinality",
                        CardinalityReducer(
                            columns=player_slot_cols,
                            min_frequency=args.min_category_freq,
                            max_categories=args.max_player_categories,
                        ),
                    ),
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
            ),
            {
                "model__n_estimators": [350, 500],
                "model__max_depth": [8, 12, None],
                "model__min_samples_leaf": [2, 4, 6],
                "model__min_samples_split": [6, 10],
                "model__max_features": ["sqrt", 0.45],
            },
        ),
        "logreg_phase4": (
            Pipeline(
                steps=[
                    (
                        "cardinality",
                        CardinalityReducer(
                            columns=player_slot_cols,
                            min_frequency=args.min_category_freq,
                            max_categories=args.max_player_categories,
                        ),
                    ),
                    ("preprocessor", preprocessor),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2500,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            {
                "model__C": [0.03, 0.08, 0.15],
            },
        ),
        "rf_phase4": (
            Pipeline(
                steps=[
                    (
                        "cardinality",
                        CardinalityReducer(
                            columns=player_slot_cols,
                            min_frequency=args.min_category_freq,
                            max_categories=args.max_player_categories,
                        ),
                    ),
                    ("preprocessor", preprocessor),
                    (
                        "model",
                        RandomForestClassifier(
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                            class_weight="balanced_subsample",
                        ),
                    ),
                ]
            ),
            {
                "model__n_estimators": [300, 450],
                "model__max_depth": [8, 12],
                "model__min_samples_leaf": [3, 5, 8],
                "model__min_samples_split": [8, 12],
                "model__max_features": ["sqrt", 0.4],
                "model__max_samples": [0.7, 0.85],
            },
        ),
    }

    best_models: dict[str, Any] = {}
    metrics: dict[str, dict[str, float]] = {}

    model_items = list(pipelines.items())
    pbar = tqdm(model_items, total=len(model_items), desc="Phase 4 training", unit="model")
    for name, (pipe, grid) in pbar:
        pbar.set_postfix_str(name)
        n_iter = min(12, _param_space_size(grid))
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=grid,
            n_iter=n_iter,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            refit=True,
            verbose=0,
        )
        search.fit(X_train, y_train)

        best_pipe = search.best_estimator_
        m = evaluate_model(best_pipe, X_train, y_train, X_test, y_test, n_classes=len(label_encoder.classes_))
        m["best_cv_accuracy"] = float(search.best_score_)

        best_models[name] = {
            "pipeline": best_pipe,
            "best_params": search.best_params_,
        }
        metrics[name] = m

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index").reset_index(names=["model"])
    metrics_df = metrics_df.sort_values(["test_accuracy", "test_f1_weighted"], ascending=False).reset_index(drop=True)

    best_model_name = str(metrics_df.loc[0, "model"])
    best_bundle = best_models[best_model_name]
    best_model = best_bundle["pipeline"]

    y_pred_best = best_model.predict(X_test)
    y_proba_best = best_model.predict_proba(X_test)

    labels = list(range(len(label_encoder.classes_)))
    report = classification_report(
        y_test,
        y_pred_best,
        labels=labels,
        target_names=label_encoder.classes_.tolist(),
        zero_division=0,
        output_dict=True,
    )

    save_plots(metrics_df, results_dir)

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred_best,
        labels=labels,
        display_labels=label_encoder.classes_,
        xticks_rotation=90,
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"Phase 4 Confusion Matrix - {best_model_name}")
    plt.tight_layout()
    plt.savefig(results_dir / "phase4_best_model_confusion_matrix.png", dpi=160)
    plt.close()

    save_reliability_plot(y_test, y_proba_best, results_dir / "phase4_best_model_reliability_plot.png")

    drift_report = compute_drift_report(train_df, test_df, player_slot_cols)

    monitoring = {
        "train_years": sorted(train_df["Date"].dt.year.unique().tolist()),
        "test_years": sorted(test_df["Date"].dt.year.unique().tolist()),
        "drift_report": drift_report,
    }

    metrics_df.to_csv(results_dir / "phase4_model_comparison_metrics.csv", index=False)
    with (results_dir / "phase4_best_model_classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with (results_dir / "phase4_model_monitoring_report.json").open("w", encoding="utf-8") as f:
        json.dump(monitoring, f, indent=2)

    metadata = {
        "phase": 4,
        "train_file": str(input_path),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "test_year": args.test_year,
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "player_slot_columns": player_slot_cols,
        "drop_no_result": bool(args.drop_no_result),
        "cardinality_controls": {
            "min_category_freq": args.min_category_freq,
            "max_player_categories": args.max_player_categories,
            "onehot_min_frequency": 7,
        },
        "models_evaluated": metrics_df["model"].tolist(),
        "model_metrics": metrics,
        "best_model": best_model_name,
        "best_model_params": best_bundle["best_params"],
        "best_model_test_metrics": metrics[best_model_name],
    }

    bundle = {
        "model_pipeline": best_model,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
        "metadata": metadata,
    }

    joblib.dump(bundle, artifacts_dir / "phase4_ipl_winner_best_pipeline.joblib")
    with (artifacts_dir / "phase4_ipl_winner_best_pipeline.pkl").open("wb") as f:
        pickle.dump(bundle, f)
    with (artifacts_dir / "phase4_model_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    sample_payload = X_test.iloc[0].to_dict()
    with (results_dir / "phase4_sample_prediction_payload.json").open("w", encoding="utf-8") as f:
        json.dump(sample_payload, f, indent=2)

    print(f"Saved results to: {results_dir}")
    print(f"Saved artifacts to: {artifacts_dir}")
    print(f"Best model: {best_model_name}")
    print(
        "Best model test metrics: "
        + json.dumps(
            {
                "accuracy": round(metrics[best_model_name]["test_accuracy"], 4),
                "f1_weighted": round(metrics[best_model_name]["test_f1_weighted"], 4),
                "f1_macro": round(metrics[best_model_name]["test_f1_macro"], 4),
                "top2_accuracy": round(metrics[best_model_name]["top2_accuracy"], 4),
                "top3_accuracy": round(metrics[best_model_name]["top3_accuracy"], 4),
                "ece": round(metrics[best_model_name]["calibration_ece"], 4),
            }
        )
    )


if __name__ == "__main__":
    main()
