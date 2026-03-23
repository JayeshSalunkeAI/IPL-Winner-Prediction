from __future__ import annotations

import argparse
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
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.base import clone
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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tqdm.auto import tqdm
from xgboost import XGBClassifier

from phase51_transforms import CardinalityReducer


RANDOM_STATE = 42
ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass
class SearchResult:
    model_name: str
    estimator: Pipeline
    best_params: dict[str, Any]
    best_cv_accuracy: float
    stage: str


class SeedAveragedEnsemble:
    """Average probabilities from multiple fitted pipelines."""

    def __init__(self, models: list[Pipeline]):
        self.models = models

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = [m.predict_proba(X) for m in self.models]
        return np.mean(probs, axis=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5.1 training with season weighting and seeded stacking ensemble.")
    parser.add_argument("--input", default=str(ROOT_DIR / "phases/phase_5_1/results/phase51_dataset.csv"))
    parser.add_argument("--results-dir", default=str(ROOT_DIR / "phases/phase_5_1/results"))
    parser.add_argument("--artifacts-dir", default=str(ROOT_DIR / "phases/phase_5_1/artifacts"))
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--drop-no-result", action="store_true")
    parser.add_argument("--min-category-freq", type=int, default=10)
    parser.add_argument("--max-player-categories", type=int, default=90)
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument("--stack-seeds", type=int, default=5)
    parser.add_argument("--selection-metric", choices=["generalization", "accuracy"], default="generalization")
    parser.add_argument("--max-fit-gap", type=float, default=None)
    return parser.parse_args()


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


def build_sample_weights(train_dates: pd.Series) -> np.ndarray:
    years = pd.to_datetime(train_dates, errors="coerce").dt.year

    weights = np.ones(len(years), dtype=float)
    weights = np.where(years <= 2021, 0.60, weights)
    weights = np.where(years == 2022, 1.25, weights)
    weights = np.where(years == 2023, 1.55, weights)
    weights = np.where(years == 2024, 1.90, weights)
    return weights


def evaluate_predictions(
    y_train: np.ndarray,
    train_pred: np.ndarray,
    train_proba: np.ndarray,
    y_test: np.ndarray,
    test_pred: np.ndarray,
    test_proba: np.ndarray,
    n_classes: int,
) -> dict[str, float]:
    out = {
        "train_accuracy": float(accuracy_score(y_train, train_pred)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "train_f1_weighted": float(f1_score(y_train, train_pred, average="weighted", zero_division=0)),
        "test_f1_weighted": float(f1_score(y_test, test_pred, average="weighted", zero_division=0)),
        "test_f1_macro": float(f1_score(y_test, test_pred, average="macro", zero_division=0)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, test_pred)),
        "test_precision_weighted": float(precision_score(y_test, test_pred, average="weighted", zero_division=0)),
        "test_recall_weighted": float(recall_score(y_test, test_pred, average="weighted", zero_division=0)),
        "train_log_loss": float(log_loss(y_train, train_proba, labels=list(range(n_classes)))),
        "test_log_loss": float(log_loss(y_test, test_proba, labels=list(range(n_classes)))),
        "top2_accuracy": top_k_accuracy(y_test, test_proba, 2),
        "top3_accuracy": top_k_accuracy(y_test, test_proba, 3),
        "calibration_ece": ece_score(y_test, test_proba),
    }
    out["fit_gap_weighted_f1"] = out["train_f1_weighted"] - out["test_f1_weighted"]
    out["fit_gap_log_loss"] = out["test_log_loss"] - out["train_log_loss"]
    out["generalization_score"] = out["test_accuracy"] - 0.25 * max(out["fit_gap_weighted_f1"], 0.0)
    return out


def evaluate_estimator(
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    n_classes: int,
) -> dict[str, float]:
    train_pred = estimator.predict(X_train)
    test_pred = estimator.predict(X_test)
    train_proba = estimator.predict_proba(X_train)
    test_proba = estimator.predict_proba(X_test)
    return evaluate_predictions(y_train, train_pred, train_proba, y_test, test_pred, test_proba, n_classes)


def tune_with_random_grid(
    model_name: str,
    pipe: Pipeline,
    grid: dict[str, list[Any]],
    cv: StratifiedKFold,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    n_iter: int,
    stage: str,
) -> SearchResult:
    rng = np.random.RandomState(RANDOM_STATE + (0 if stage == "fast" else 100))
    keys = list(grid.keys())
    combos: list[dict[str, Any]] = []

    for _ in range(n_iter):
        combos.append({k: grid[k][rng.randint(0, len(grid[k]))] for k in keys})

    best_est = None
    best_score = -1.0
    best_params = {}

    for params in combos:
        est = clone(pipe)
        est.set_params(**params)

        fold_scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr = X_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_tr = y_train[train_idx]
            y_val = y_train[val_idx]
            sw_tr = sample_weight[train_idx]

            est.fit(X_tr, y_tr, model__sample_weight=sw_tr)
            pred = est.predict(X_val)
            fold_scores.append(accuracy_score(y_val, pred))

        cv_score = float(np.mean(fold_scores))
        if cv_score > best_score:
            best_score = cv_score
            best_params = params
            best_est = clone(pipe)
            best_est.set_params(**params)
            best_est.fit(X_train, y_train, model__sample_weight=sample_weight)

    if best_est is None:
        raise RuntimeError(f"No valid fit found for {model_name} ({stage})")

    return SearchResult(
        model_name=model_name,
        estimator=best_est,
        best_params=best_params,
        best_cv_accuracy=best_score,
        stage=stage,
    )


def build_xgb_pipeline(preprocessor: ColumnTransformer, player_slot_cols: list[str], min_freq: int, max_cat: int, seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("card", CardinalityReducer(player_slot_cols, min_frequency=min_freq, max_categories=max_cat)),
            ("preprocessor", preprocessor),
            (
                "model",
                XGBClassifier(
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
        ],
        memory=joblib.Memory(location=None, verbose=0),
    )


def build_cat_pipeline(preprocessor: ColumnTransformer, player_slot_cols: list[str], min_freq: int, max_cat: int, seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("card", CardinalityReducer(player_slot_cols, min_frequency=min_freq, max_categories=max_cat)),
            ("preprocessor", preprocessor),
            (
                "model",
                CatBoostClassifier(
                    loss_function="MultiClass",
                    eval_metric="MultiClass",
                    random_seed=seed,
                    verbose=0,
                    bootstrap_type="Bernoulli",
                    thread_count=1,
                ),
            ),
        ],
        memory=joblib.Memory(location=None, verbose=0),
    )


def build_stack_pipeline(
    preprocessor: ColumnTransformer,
    player_slot_cols: list[str],
    min_freq: int,
    max_cat: int,
    seed: int,
    cv: StratifiedKFold,
) -> Pipeline:
    base_estimators = [
        (
            "xgb",
            XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=seed,
                n_jobs=1,
            ),
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=350,
                max_depth=10,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=1,
            ),
        ),
        (
            "lr",
            LogisticRegression(
                max_iter=700,
                solver="saga",
                penalty="elasticnet",
                l1_ratio=0.3,
                random_state=seed,
            ),
        ),
    ]

    return Pipeline(
        steps=[
            ("card", CardinalityReducer(player_slot_cols, min_frequency=min_freq, max_categories=max_cat)),
            ("preprocessor", preprocessor),
            (
                "model",
                StackingClassifier(
                    estimators=base_estimators,
                    final_estimator=LogisticRegression(
                        max_iter=700,
                        solver="saga",
                        penalty="elasticnet",
                        l1_ratio=0.3,
                        random_state=seed,
                    ),
                    stack_method="predict_proba",
                    passthrough=True,
                    cv=cv,
                    n_jobs=-1,
                ),
            ),
        ],
        memory=joblib.Memory(location=None, verbose=0),
    )


def compute_drift_report(train_df: pd.DataFrame, test_df: pd.DataFrame, player_slot_cols: list[str]) -> dict[str, Any]:
    numeric_drift: dict[str, Any] = {}
    numeric_cols = [c for c in train_df.columns if c.endswith("_prior") or c.endswith("_5")]
    for col in numeric_cols:
        if col not in test_df.columns:
            continue
        tr = pd.to_numeric(train_df[col], errors="coerce").dropna()
        te = pd.to_numeric(test_df[col], errors="coerce").dropna()
        if len(tr) == 0 or len(te) == 0:
            continue
        tr_mean = float(tr.mean())
        te_mean = float(te.mean())
        tr_std = float(tr.std())
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
    plt.title("Phase 5.1 Reliability Plot")
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
        raise FileNotFoundError(f"Dataset not found at {input_path}. Run phase51_extract_dataset.py first.")

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

    sample_weight = build_sample_weights(train_df["Date"])

    preprocessor, _, _ = build_preprocessor(X_train)
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=RANDOM_STATE)

    fast_grids = {
        "xgb_gen": {
            "model__n_estimators": [180, 260],
            "model__max_depth": [3, 4],
            "model__learning_rate": [0.03, 0.06],
            "model__subsample": [0.7, 0.85],
            "model__colsample_bytree": [0.5, 0.7],
            "model__reg_lambda": [8.0, 14.0],
            "model__reg_alpha": [2.0, 6.0],
            "model__min_child_weight": [6, 10],
        },
        "catboost_gen": {
            "model__iterations": [240, 340],
            "model__depth": [6, 8],
            "model__learning_rate": [0.03, 0.05],
            "model__l2_leaf_reg": [3.0, 7.0],
            "model__subsample": [0.8, 1.0],
            "model__random_strength": [0.7, 1.3],
        },
        "stacking_gen": {
            "model__final_estimator__C": [0.5, 0.9, 1.3],
            "model__final_estimator__l1_ratio": [0.15, 0.35, 0.55],
        },
    }

    full_grids = {
        "xgb_gen": {
            "model__n_estimators": [180, 260, 360],
            "model__max_depth": [3, 4, 5],
            "model__learning_rate": [0.02, 0.04, 0.06],
            "model__subsample": [0.65, 0.8, 0.95],
            "model__colsample_bytree": [0.45, 0.6, 0.8],
            "model__reg_lambda": [8.0, 14.0, 20.0],
            "model__reg_alpha": [2.0, 6.0, 10.0],
            "model__min_child_weight": [5, 9, 13],
            "model__gamma": [0.8, 1.6, 2.4],
        },
        "catboost_gen": {
            "model__iterations": [240, 340, 460],
            "model__depth": [5, 6, 8],
            "model__learning_rate": [0.02, 0.04, 0.06],
            "model__l2_leaf_reg": [3.0, 6.0, 9.0],
            "model__subsample": [0.75, 0.9, 1.0],
            "model__random_strength": [0.5, 1.0, 1.5],
        },
        "stacking_gen": {
            "model__final_estimator__C": [0.35, 0.6, 0.9, 1.3, 1.8],
            "model__final_estimator__l1_ratio": [0.1, 0.25, 0.4, 0.55, 0.7],
        },
    }

    base_pipelines: dict[str, Pipeline] = {
        "xgb_gen": build_xgb_pipeline(preprocessor, player_slot_cols, args.min_category_freq, args.max_player_categories, RANDOM_STATE),
        "catboost_gen": build_cat_pipeline(preprocessor, player_slot_cols, args.min_category_freq, args.max_player_categories, RANDOM_STATE),
        "stacking_gen": build_stack_pipeline(preprocessor, player_slot_cols, args.min_category_freq, args.max_player_categories, RANDOM_STATE, cv),
    }

    results: list[SearchResult] = []
    metrics: dict[str, dict[str, float]] = {}

    # Stage 1: fast search on stacking + catboost + xgb.
    pbar = tqdm(["xgb_gen", "catboost_gen", "stacking_gen"], desc="Phase 5.1 fast search", unit="model")
    for name in pbar:
        pbar.set_postfix_str(name)
        sr = tune_with_random_grid(
            model_name=name,
            pipe=base_pipelines[name],
            grid=fast_grids[name],
            cv=cv,
            X_train=X_train,
            y_train=y_train,
            sample_weight=sample_weight,
            n_iter=8,
            stage="fast",
        )
        results.append(sr)
        m = evaluate_estimator(sr.estimator, X_train, y_train, X_test, y_test, n_classes=len(label_encoder.classes_))
        m["best_cv_accuracy"] = float(sr.best_cv_accuracy)
        m["search_stage"] = "fast"
        metrics[f"{name}_fast"] = m

    fast_rank = sorted(results, key=lambda r: r.best_cv_accuracy, reverse=True)
    top2 = [fast_rank[0].model_name, fast_rank[1].model_name]

    # Stage 2: full search only on top2 from fast stage.
    pbar = tqdm(top2, desc="Phase 5.1 full search top2", unit="model")
    for name in pbar:
        pbar.set_postfix_str(name)
        sr = tune_with_random_grid(
            model_name=name,
            pipe=base_pipelines[name],
            grid=full_grids[name],
            cv=cv,
            X_train=X_train,
            y_train=y_train,
            sample_weight=sample_weight,
            n_iter=14,
            stage="full",
        )

        if name == "stacking_gen":
            # Seed ensemble for stacking: average 3-5 differently seeded stacks.
            seed_count = int(np.clip(args.stack_seeds, 3, 5))
            seeds = [RANDOM_STATE + i * 73 for i in range(seed_count)]
            stack_models: list[Pipeline] = []

            for seed in seeds:
                stack_pipe = build_stack_pipeline(
                    preprocessor,
                    player_slot_cols,
                    args.min_category_freq,
                    args.max_player_categories,
                    seed,
                    cv,
                )
                stack_pipe.set_params(**sr.best_params)
                stack_pipe.fit(X_train, y_train, model__sample_weight=sample_weight)
                stack_models.append(stack_pipe)

            seed_ensemble = SeedAveragedEnsemble(stack_models)
            train_pred = seed_ensemble.predict(X_train)
            test_pred = seed_ensemble.predict(X_test)
            train_proba = seed_ensemble.predict_proba(X_train)
            test_proba = seed_ensemble.predict_proba(X_test)
            m = evaluate_predictions(
                y_train,
                train_pred,
                train_proba,
                y_test,
                test_pred,
                test_proba,
                n_classes=len(label_encoder.classes_),
            )
            m["best_cv_accuracy"] = float(sr.best_cv_accuracy)
            m["search_stage"] = "full_seed_ensemble"
            metrics[f"{name}_full"] = m

            # Save as custom object reference in results.
            sr = SearchResult(
                model_name=name,
                estimator=stack_models[0],
                best_params={**sr.best_params, "seed_ensemble_size": seed_count},
                best_cv_accuracy=sr.best_cv_accuracy,
                stage="full_seed_ensemble",
            )
            results.append(sr)
        else:
            m = evaluate_estimator(sr.estimator, X_train, y_train, X_test, y_test, n_classes=len(label_encoder.classes_))
            m["best_cv_accuracy"] = float(sr.best_cv_accuracy)
            m["search_stage"] = "full"
            metrics[f"{name}_full"] = m
            results.append(sr)

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index").reset_index(names=["model_run"])
    metrics_df["model"] = metrics_df["model_run"].str.replace("_fast", "", regex=False).str.replace("_full", "", regex=False)

    if args.selection_metric == "accuracy":
        sort_cols = ["test_accuracy", "generalization_score"]
    else:
        sort_cols = ["generalization_score", "test_accuracy"]

    candidate_df = metrics_df.copy()
    if args.max_fit_gap is not None:
        constrained = candidate_df[candidate_df["fit_gap_weighted_f1"] <= float(args.max_fit_gap)].copy()
        if not constrained.empty:
            candidate_df = constrained

    candidate_df = candidate_df.sort_values(sort_cols, ascending=False).reset_index(drop=True)
    best_run_name = str(candidate_df.loc[0, "model_run"])

    metrics_df = metrics_df.sort_values(sort_cols, ascending=False).reset_index(drop=True)
    metrics_df["selected_for_deploy"] = metrics_df["model_run"] == best_run_name

    best_model_name = best_run_name.replace("_full", "").replace("_fast", "")

    selected_models = [r for r in results if f"{r.model_name}_{'full' if 'full' in r.stage else 'fast'}" == best_run_name]
    if not selected_models:
        raise RuntimeError("Best model object not found in search results")
    selected = selected_models[-1]

    if best_model_name == "stacking_gen" and selected.stage == "full_seed_ensemble":
        seed_count = int(selected.best_params.get("seed_ensemble_size", 3))
        seeds = [RANDOM_STATE + i * 73 for i in range(seed_count)]
        stack_models: list[Pipeline] = []
        stack_params = {k: v for k, v in selected.best_params.items() if k != "seed_ensemble_size"}
        for seed in seeds:
            stack_pipe = build_stack_pipeline(
                preprocessor,
                player_slot_cols,
                args.min_category_freq,
                args.max_player_categories,
                seed,
                cv,
            )
            stack_pipe.set_params(**stack_params)
            stack_pipe.fit(X_train, y_train, model__sample_weight=sample_weight)
            stack_models.append(stack_pipe)
        best_estimator: Any = SeedAveragedEnsemble(stack_models)
        serializable_model: Any = {
            "kind": "seed_averaged_stacking",
            "seed_models": stack_models,
            "seed_count": seed_count,
            "params": stack_params,
        }
    else:
        best_estimator = selected.estimator
        serializable_model = {"kind": "single_pipeline", "pipeline": selected.estimator}

    y_pred = best_estimator.predict(X_test)
    y_proba = best_estimator.predict_proba(X_test)
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
    plt.figure(figsize=(10, 5))
    m_sorted = metrics_df.sort_values("test_accuracy", ascending=False)
    sns.barplot(data=m_sorted, x="model_run", y="test_accuracy", hue="model_run", legend=False, palette="Set2")
    plt.title("Phase 5.1: Test Accuracy")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(results_dir / "phase51_model_comparison_accuracy.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=m_sorted, x="model_run", y="fit_gap_weighted_f1", hue="model_run", legend=False, palette="rocket")
    plt.title("Phase 5.1: Overfit Gap")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(results_dir / "phase51_overfit_gap.png", dpi=160)
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
    ax.set_title(f"Phase 5.1 Confusion Matrix - {best_run_name}")
    plt.tight_layout()
    plt.savefig(results_dir / "phase51_best_model_confusion_matrix.png", dpi=160)
    plt.close()

    save_reliability_plot(y_test, y_proba, results_dir / "phase51_best_model_reliability_plot.png")

    monitoring = {
        "train_years": sorted(train_df["Date"].dt.year.unique().tolist()),
        "test_years": sorted(test_df["Date"].dt.year.unique().tolist()),
        "drift_report": compute_drift_report(train_df, test_df, player_slot_cols),
    }

    metrics_df.to_csv(results_dir / "phase51_model_comparison_metrics.csv", index=False)
    with (results_dir / "phase51_best_model_classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with (results_dir / "phase51_model_monitoring_report.json").open("w", encoding="utf-8") as f:
        json.dump(monitoring, f, indent=2)

    metadata = {
        "phase": "5.1",
        "train_file": str(input_path),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "test_year": int(args.test_year),
        "drop_no_result": bool(args.drop_no_result),
        "cv_folds": int(args.cv_folds),
        "stack_seeds": int(np.clip(args.stack_seeds, 3, 5)),
        "selection_metric": args.selection_metric,
        "max_fit_gap": None if args.max_fit_gap is None else float(args.max_fit_gap),
        "min_category_freq": int(args.min_category_freq),
        "max_player_categories": int(args.max_player_categories),
        "season_weighting": {
            "<=2021": 0.60,
            "2022": 1.25,
            "2023": 1.55,
            "2024": 1.90,
        },
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "player_slot_columns": player_slot_cols,
        "top2_from_fast": top2,
        "models_evaluated": metrics_df["model_run"].tolist(),
        "model_metrics": metrics,
        "best_model_run": best_run_name,
        "best_model_name": best_model_name,
        "best_model_params": selected.best_params,
        "best_model_test_metrics": metrics[best_run_name],
    }

    bundle = {
        "model": serializable_model,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
        "metadata": metadata,
    }

    joblib.dump(bundle, artifacts_dir / "phase51_ipl_winner_best_pipeline.joblib")
    with (artifacts_dir / "phase51_ipl_winner_best_pipeline.pkl").open("wb") as f:
        pickle.dump(bundle, f)
    with (artifacts_dir / "phase51_model_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    sample_payload = X_test.iloc[0].to_dict()
    with (results_dir / "phase51_sample_prediction_payload.json").open("w", encoding="utf-8") as f:
        json.dump(sample_payload, f, indent=2)

    print(f"Saved results to: {results_dir}")
    print(f"Saved artifacts to: {artifacts_dir}")
    print(f"Best model run: {best_run_name}")
    print(
        "Best model test metrics: "
        + json.dumps(
            {
                "accuracy": round(metrics[best_run_name]["test_accuracy"], 4),
                "f1_weighted": round(metrics[best_run_name]["test_f1_weighted"], 4),
                "f1_macro": round(metrics[best_run_name]["test_f1_macro"], 4),
                "fit_gap": round(metrics[best_run_name]["fit_gap_weighted_f1"], 4),
                "gen_score": round(metrics[best_run_name]["generalization_score"], 4),
            }
        )
    )


if __name__ == "__main__":
    main()
