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
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tqdm.auto import tqdm
from xgboost import XGBClassifier

from phase41_transforms import CardinalityReducer


RANDOM_STATE = 42
ROOT_DIR = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4.1 training with stronger generalization focus.")
    parser.add_argument("--input", default=str(ROOT_DIR / "phases/phase_4/results/phase4_dataset.csv"))
    parser.add_argument("--results-dir", default=str(ROOT_DIR / "phases/phase_4_1/results"))
    parser.add_argument("--artifacts-dir", default=str(ROOT_DIR / "phases/phase_4_1/artifacts"))
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--drop-no-result", action="store_true")
    return parser.parse_args()


def _param_space_size(grid: dict[str, list[Any]]) -> int:
    size = 1
    for values in grid.values():
        size *= max(1, len(values))
    return size


def top_k_accuracy(y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    topk = np.argsort(proba, axis=1)[:, -k:]
    return float(np.mean([int(y in row) for y, row in zip(y_true, topk)]))


def ece_score(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
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
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=True)),
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
) -> dict[str, float]:
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)

    out = {
        "train_accuracy": float(accuracy_score(y_train, train_pred)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "train_f1_weighted": float(f1_score(y_train, train_pred, average="weighted", zero_division=0)),
        "test_f1_weighted": float(f1_score(y_test, test_pred, average="weighted", zero_division=0)),
        "test_f1_macro": float(f1_score(y_test, test_pred, average="macro", zero_division=0)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, test_pred)),
        "top2_accuracy": top_k_accuracy(y_test, test_proba, 2),
        "top3_accuracy": top_k_accuracy(y_test, test_proba, 3),
        "calibration_ece": ece_score(y_test, test_proba),
    }
    out["fit_gap_weighted_f1"] = out["train_f1_weighted"] - out["test_f1_weighted"]
    # Composite objective to penalize overfit while retaining accuracy.
    out["generalization_score"] = out["test_accuracy"] - 0.20 * max(out["fit_gap_weighted_f1"], 0.0)
    return out


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    results_dir = Path(args.results_dir)
    artifacts_dir = Path(args.artifacts_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Match_Winner"]).sort_values(["Date", "Match_ID"]).reset_index(drop=True)

    if args.drop_no_result:
        df = df[df["Match_Winner"] != "Draw/No Result"].copy()

    test_mask = df["Date"].dt.year == args.test_year
    train_df = df.loc[~test_mask].copy()
    test_df = df.loc[test_mask].copy()

    drop_cols = ["Match_Winner", "Match_ID", "Date", "Teams"]
    feature_columns = [c for c in train_df.columns if c not in drop_cols]
    player_slot_cols = [c for c in feature_columns if c.startswith("Team1_Player_") or c.startswith("Team2_Player_")]

    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()

    y_train_raw = train_df["Match_Winner"].astype(str).values
    y_test_raw = test_df["Match_Winner"].astype(str).values
    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate([y_train_raw, y_test_raw]))
    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    preprocessor, _, _ = build_preprocessor(X_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    pipelines: dict[str, tuple[Pipeline, dict[str, list[Any]]]] = {
        "xgb_gen": (
            Pipeline(
                steps=[
                    ("card", CardinalityReducer(player_slot_cols, min_frequency=12, max_categories=70)),
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
                "model__n_estimators": [140, 180],
                "model__max_depth": [3, 4],
                "model__learning_rate": [0.03, 0.05],
                "model__subsample": [0.65, 0.75],
                "model__colsample_bytree": [0.40, 0.55],
                "model__reg_lambda": [12.0, 18.0],
                "model__reg_alpha": [5.0, 8.0],
                "model__min_child_weight": [10, 14],
                "model__gamma": [1.5, 2.5],
            },
        ),
        "rf_gen": (
            Pipeline(
                steps=[
                    ("card", CardinalityReducer(player_slot_cols, min_frequency=12, max_categories=70)),
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
                "model__n_estimators": [350, 500],
                "model__max_depth": [8, 10],
                "model__min_samples_leaf": [4, 6, 8],
                "model__min_samples_split": [10, 14],
                "model__max_features": ["sqrt", 0.4],
                "model__max_samples": [0.70, 0.80],
            },
        ),
        "extra_trees_gen": (
            Pipeline(
                steps=[
                    ("card", CardinalityReducer(player_slot_cols, min_frequency=12, max_categories=70)),
                    ("preprocessor", preprocessor),
                    (
                        "model",
                        ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"),
                    ),
                ]
            ),
            {
                "model__n_estimators": [300, 450],
                "model__max_depth": [10, 14],
                "model__min_samples_leaf": [4, 6],
                "model__min_samples_split": [10, 14],
                "model__max_features": ["sqrt", 0.35],
            },
        ),
    }

    best_models: dict[str, Any] = {}
    metrics: dict[str, dict[str, float]] = {}

    items = list(pipelines.items())
    pbar = tqdm(items, total=len(items), desc="Phase 4.1 training", unit="model")
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

        best = search.best_estimator_
        m = evaluate_model(best, X_train, y_train, X_test, y_test)
        m["best_cv_accuracy"] = float(search.best_score_)

        best_models[name] = {"pipeline": best, "best_params": search.best_params_}
        metrics[name] = m

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index").reset_index(names=["model"])
    metrics_df = metrics_df.sort_values(["generalization_score", "test_accuracy"], ascending=False).reset_index(drop=True)

    best_name = str(metrics_df.loc[0, "model"])
    best_bundle = best_models[best_name]
    best_model = best_bundle["pipeline"]

    y_pred = best_model.predict(X_test)
    labels = list(range(len(label_encoder.classes_)))
    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=label_encoder.classes_.tolist(),
        zero_division=0,
        output_dict=True,
    )

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 5))
    m = metrics_df.sort_values("test_accuracy", ascending=False)
    sns.barplot(data=m, x="model", y="test_accuracy", hue="model", legend=False, palette="Set2")
    plt.title("Phase 4.1: Test Accuracy")
    plt.tight_layout()
    plt.savefig(results_dir / "phase41_test_accuracy.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(data=m, x="model", y="fit_gap_weighted_f1", hue="model", legend=False, palette="rocket")
    plt.title("Phase 4.1: Overfit Gap")
    plt.tight_layout()
    plt.savefig(results_dir / "phase41_overfit_gap.png", dpi=160)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=labels,
        display_labels=label_encoder.classes_,
        xticks_rotation=90,
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"Phase 4.1 Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(results_dir / "phase41_best_confusion_matrix.png", dpi=160)
    plt.close()

    metrics_df.to_csv(results_dir / "phase41_model_comparison_metrics.csv", index=False)
    with (results_dir / "phase41_best_model_classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    metadata = {
        "phase": "4.1",
        "train_file": str(input_path),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "test_year": int(args.test_year),
        "drop_no_result": bool(args.drop_no_result),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "player_slot_columns": player_slot_cols,
        "models_evaluated": metrics_df["model"].tolist(),
        "model_metrics": metrics,
        "best_model": best_name,
        "best_model_params": best_bundle["best_params"],
        "best_model_test_metrics": metrics[best_name],
    }

    bundle = {
        "model_pipeline": best_model,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
        "metadata": metadata,
    }

    joblib.dump(bundle, artifacts_dir / "phase41_ipl_winner_best_pipeline.joblib")
    with (artifacts_dir / "phase41_ipl_winner_best_pipeline.pkl").open("wb") as f:
        pickle.dump(bundle, f)
    with (artifacts_dir / "phase41_model_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    sample_payload = X_test.iloc[0].to_dict()
    with (results_dir / "phase41_sample_prediction_payload.json").open("w", encoding="utf-8") as f:
        json.dump(sample_payload, f, indent=2)

    print(f"Saved results to: {results_dir}")
    print(f"Saved artifacts to: {artifacts_dir}")
    print(f"Best model: {best_name}")
    print(
        "Best model test metrics: "
        + json.dumps(
            {
                "accuracy": round(metrics[best_name]["test_accuracy"], 4),
                "f1_weighted": round(metrics[best_name]["test_f1_weighted"], 4),
                "f1_macro": round(metrics[best_name]["test_f1_macro"], 4),
                "fit_gap": round(metrics[best_name]["fit_gap_weighted_f1"], 4),
                "gen_score": round(metrics[best_name]["generalization_score"], 4),
            }
        )
    )


if __name__ == "__main__":
    main()
