from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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
from tqdm.auto import tqdm  # type: ignore[import-not-found]
from xgboost import XGBClassifier

from phase3_transforms import CardinalityReducer


RANDOM_STATE = 42
INPUT_PATH = Path("IPL_Winner_Model_Dataset_With_Players.csv")
RESULTS_DIR = Path("phase_3_result")
ARTIFACTS_DIR = Path("phase_3_artifacts")
TEST_YEAR = 2025


@dataclass
class Phase3Config:
    test_year: int = TEST_YEAR
    min_category_freq: int = 8
    max_categories_per_player_col: int = 80


def _param_space_size(grid: dict[str, list[Any]]) -> int:
    size = 1
    for values in grid.values():
        size *= max(1, len(values))
    return size


def load_and_split(data_path: Path, test_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Match_Winner"]).copy()
    df = df.sort_values(["Date", "Match_ID"]).reset_index(drop=True)

    test_mask = df["Date"].dt.year == test_year
    if test_mask.sum() > 0:
        train_df = df.loc[~test_mask].copy()
        test_df = df.loc[test_mask].copy()
    else:
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train/Test split failed")

    return train_df, test_df


def prune_features(df: pd.DataFrame) -> pd.DataFrame:
    # Remove redundant high-cardinality aggregate columns; slot columns are preserved.
    drop_if_exists = ["Teams", "Team1_Players", "Team2_Players"]
    keep_df = df.copy()
    for col in drop_if_exists:
        if col in keep_df.columns:
            keep_df = keep_df.drop(columns=[col])
    return keep_df


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=6,
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
        transformers=[
            ("cat", cat_pipe, categorical_cols),
            ("num", num_pipe, numeric_cols),
        ],
        remainder="drop",
    )

    return preprocessor, categorical_cols, numeric_cols


def top_k_accuracy(y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    topk = np.argsort(proba, axis=1)[:, -k:]
    hits = [int(y in row) for y, row in zip(y_true, topk)]
    return float(np.mean(hits))


def multiclass_brier_score(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    y_onehot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1)))


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
        acc_bin = float(correct[mask].mean())
        conf_bin = float(conf[mask].mean())
        ece += (mask.sum() / n) * abs(acc_bin - conf_bin)

    return float(ece)


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
        "test_precision_weighted": float(
            precision_score(y_test, test_pred, average="weighted", zero_division=0)
        ),
        "test_recall_weighted": float(recall_score(y_test, test_pred, average="weighted", zero_division=0)),
        "test_log_loss": float(log_loss(y_test, test_proba, labels=list(range(n_classes)))),
        "top2_accuracy": top_k_accuracy(y_test, test_proba, k=2),
        "top3_accuracy": top_k_accuracy(y_test, test_proba, k=3),
        "calibration_ece": expected_calibration_error(y_test, test_proba, n_bins=10),
        "multiclass_brier": multiclass_brier_score(y_test, test_proba, n_classes=n_classes),
    }
    out["fit_gap_weighted_f1"] = out["train_f1_weighted"] - out["test_f1_weighted"]
    return out


def fit_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    player_slot_cols: list[str],
    cfg: Phase3Config,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    preprocessor, _, _ = build_preprocessor(X_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    pipelines: dict[str, tuple[Pipeline, dict[str, list[Any]]]] = {
        "xgboost_reg": (
            Pipeline(
                steps=[
                    (
                        "cardinality",
                        CardinalityReducer(
                            columns=player_slot_cols,
                            min_frequency=cfg.min_category_freq,
                            max_categories=cfg.max_categories_per_player_col,
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
                "model__n_estimators": [140, 200],
                "model__max_depth": [3, 4],
                "model__learning_rate": [0.03, 0.05],
                "model__subsample": [0.65, 0.75],
                "model__colsample_bytree": [0.4, 0.55],
                "model__reg_lambda": [8.0, 12.0],
                "model__reg_alpha": [3.0, 6.0],
                "model__min_child_weight": [8, 12],
                "model__gamma": [1.0, 2.0],
            },
        ),
        "random_forest_reg": (
            Pipeline(
                steps=[
                    (
                        "cardinality",
                        CardinalityReducer(
                            columns=player_slot_cols,
                            min_frequency=cfg.min_category_freq,
                            max_categories=cfg.max_categories_per_player_col,
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
                "model__n_estimators": [350, 500],
                "model__max_depth": [8, 10, 12],
                "model__min_samples_leaf": [3, 5, 8],
                "model__min_samples_split": [8, 12],
                "model__max_features": ["sqrt", 0.4],
                "model__max_samples": [0.7, 0.85],
                "model__ccp_alpha": [0.0, 0.0005, 0.001],
            },
        ),
        "logistic_reg_l2": (
            Pipeline(
                steps=[
                    (
                        "cardinality",
                        CardinalityReducer(
                            columns=player_slot_cols,
                            min_frequency=cfg.min_category_freq,
                            max_categories=cfg.max_categories_per_player_col,
                        ),
                    ),
                    ("preprocessor", preprocessor),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            C=0.1,
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            {
                "model__C": [0.03, 0.1, 0.3],
            },
        ),
    }

    best_models: dict[str, Any] = {}
    metrics: dict[str, dict[str, float]] = {}

    model_items = list(pipelines.items())
    pbar = tqdm(model_items, total=len(model_items), desc="Phase 3 model training", unit="model")

    for model_name, (pipe, grid) in pbar:
        pbar.set_postfix_str(model_name)
        n_iter = min(10, _param_space_size(grid))

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=grid,
            n_iter=n_iter,
            scoring="f1_weighted",
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            refit=True,
            verbose=0,
        )
        search.fit(X_train, y_train)

        best_pipe = search.best_estimator_
        model_metrics = evaluate_model(
            best_pipe,
            X_train,
            y_train,
            X_test,
            y_test,
            n_classes=len(np.unique(y_train)),
        )
        model_metrics["best_cv_weighted_f1"] = float(search.best_score_)

        best_models[model_name] = {
            "pipeline": best_pipe,
            "best_params": search.best_params_,
        }
        metrics[model_name] = model_metrics

    return best_models, metrics


def compute_drift_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    player_slot_cols: list[str],
) -> dict[str, Any]:
    numeric_cols = [
        "team1_form_winrate_5",
        "team2_form_winrate_5",
        "venue_chase_winrate_prior",
        "venue_score_prior",
        "team1_player_elo_avg_prior",
        "team2_player_elo_avg_prior",
        "player_elo_gap_prior",
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

    category_cols = ["Team1", "Team2", "Toss_Winner", "Toss_Decision"]
    category_drift: dict[str, Any] = {}
    for col in category_cols:
        if col not in train_df.columns or col not in test_df.columns:
            continue
        tr_vals = set(train_df[col].astype(str).unique().tolist())
        te_vals = test_df[col].astype(str)
        unseen = (~te_vals.isin(tr_vals)).mean()
        category_drift[col] = {"test_unseen_rate": round(float(unseen), 4)}

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

    all_test_players_s = pd.Series(all_test_players, dtype="object") if all_test_players else pd.Series([], dtype="object")
    lineup_unseen_rate = float((~all_test_players_s.isin(all_train_players)).mean()) if len(all_test_players_s) else 0.0

    return {
        "numeric_drift": numeric_drift,
        "category_drift": category_drift,
        "lineup_drift": {
            "overall_unseen_player_rate": round(lineup_unseen_rate, 4),
            "per_player_slot_unseen_rate": per_col_unseen,
        },
    }


def save_plots(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    sns.set_theme(style="whitegrid")

    m = metrics_df.sort_values("test_f1_weighted", ascending=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=m, x="model", y="test_f1_weighted", hue="model", palette="Set2", legend=False)
    plt.title("Phase 3: Test Weighted F1 Comparison")
    plt.tight_layout()
    plt.savefig(out_dir / "phase3_model_comparison_test_f1.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(data=m, x="model", y="fit_gap_weighted_f1", hue="model", palette="rocket", legend=False)
    plt.title("Phase 3: Overfitting Gap (Train-Test Weighted F1)")
    plt.tight_layout()
    plt.savefig(out_dir / "phase3_overfit_gap.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    long_topk = m[["model", "top2_accuracy", "top3_accuracy"]].melt(
        id_vars=["model"], var_name="metric", value_name="value"
    )
    sns.barplot(data=long_topk, x="model", y="value", hue="metric")
    plt.title("Phase 3: Top-K Accuracy")
    plt.tight_layout()
    plt.savefig(out_dir / "phase3_topk_accuracy.png", dpi=160)
    plt.close()


def save_calibration_plot(y_test: np.ndarray, proba: np.ndarray, out_path: Path) -> None:
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
    plt.title("Phase 3: Reliability Plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    cfg = Phase3Config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df_raw, test_df_raw = load_and_split(INPUT_PATH, test_year=cfg.test_year)
    train_df = prune_features(train_df_raw)
    test_df = prune_features(test_df_raw)

    drop_cols = ["Match_Winner", "Match_ID", "Date"]
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

    best_models, metrics = fit_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        player_slot_cols=player_slot_cols,
        cfg=cfg,
    )

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index").reset_index(names=["model"])
    metrics_df = metrics_df.sort_values("test_f1_weighted", ascending=False).reset_index(drop=True)

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

    save_plots(metrics_df, RESULTS_DIR)

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
    ax.set_title(f"Phase 3 Confusion Matrix - {best_model_name}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "phase3_best_model_confusion_matrix.png", dpi=160)
    plt.close()

    save_calibration_plot(y_test, y_proba_best, RESULTS_DIR / "phase3_best_model_reliability_plot.png")

    drift_report = compute_drift_report(train_df, test_df, player_slot_cols)

    model_monitoring = {
        "train_years": sorted(train_df_raw["Date"].dt.year.unique().tolist()),
        "test_years": sorted(test_df_raw["Date"].dt.year.unique().tolist()),
        "drift_report": drift_report,
    }

    metrics_df.to_csv(RESULTS_DIR / "phase3_model_comparison_metrics.csv", index=False)
    with (RESULTS_DIR / "phase3_best_model_classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with (RESULTS_DIR / "phase3_model_monitoring_report.json").open("w", encoding="utf-8") as f:
        json.dump(model_monitoring, f, indent=2)

    metadata = {
        "phase": 3,
        "train_file": str(INPUT_PATH),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "test_year": cfg.test_year,
        "feature_count": int(len(feature_columns)),
        "features_after_pruning": feature_columns,
        "player_slot_columns": player_slot_cols,
        "cardinality_controls": {
            "min_category_freq": cfg.min_category_freq,
            "max_categories_per_player_col": cfg.max_categories_per_player_col,
            "onehot_min_frequency": 6,
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

    joblib.dump(bundle, ARTIFACTS_DIR / "phase3_ipl_winner_best_pipeline.joblib")
    with (ARTIFACTS_DIR / "phase3_ipl_winner_best_pipeline.pkl").open("wb") as f:
        pickle.dump(bundle, f)
    with (ARTIFACTS_DIR / "phase3_model_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    sample_payload = X_test.iloc[0].to_dict()
    with (RESULTS_DIR / "phase3_sample_prediction_payload.json").open("w", encoding="utf-8") as f:
        json.dump(sample_payload, f, indent=2)

    print(f"Saved results to: {RESULTS_DIR}")
    print(f"Saved artifacts to: {ARTIFACTS_DIR}")
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
