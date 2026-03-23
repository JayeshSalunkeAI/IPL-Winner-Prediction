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
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
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
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tqdm.auto import tqdm
from xgboost import XGBClassifier

from phase5_transforms import CardinalityReducer


RANDOM_STATE = 42
ROOT_DIR = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5 training with LightGBM, CatBoost, and Stacking ensemble.")
    parser.add_argument("--input", default=str(ROOT_DIR / "phases/phase_4/results/phase4_dataset.csv"))
    parser.add_argument("--results-dir", default=str(ROOT_DIR / "phases/phase_5/results"))
    parser.add_argument("--artifacts-dir", default=str(ROOT_DIR / "phases/phase_5/artifacts"))
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--drop-no-result", action="store_true")
    parser.add_argument("--min-category-freq", type=int, default=12)
    parser.add_argument("--max-player-categories", type=int, default=70)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--search-mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--selection-metric", choices=["generalization", "accuracy"], default="generalization")
    parser.add_argument("--max-fit-gap", type=float, default=None)
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
    n_classes: int,
) -> dict[str, float]:
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_proba = model.predict_proba(X_train)
    test_proba = model.predict_proba(X_test)

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
    plt.title("Phase 5: Reliability Plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def build_model_bundle(
    preprocessor: ColumnTransformer,
    player_slot_cols: list[str],
    min_category_freq: int,
    max_player_categories: int,
    cv: StratifiedKFold,
    search_mode: str,
    cache_dir: Path,
) -> dict[str, dict[str, Any]]:
    memory = joblib.Memory(location=cache_dir, verbose=0)

    base_stack_estimators = [
        (
            "xgb",
            XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        ),
        (
            "rf",
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=1,
                class_weight="balanced_subsample",
            ),
        ),
        (
            "lr",
            LogisticRegression(
                max_iter=700,
                multi_class="multinomial",
                solver="saga",
                penalty="elasticnet",
                l1_ratio=0.3,
            ),
        ),
    ]

    def base_pipe(model: Any) -> Pipeline:
        return Pipeline(
            memory=memory,
            steps=[
                (
                    "card",
                    CardinalityReducer(
                        player_slot_cols,
                        min_frequency=min_category_freq,
                        max_categories=max_player_categories,
                    ),
                ),
                ("preprocessor", preprocessor),
                ("model", model),
            ],
        )

    if search_mode == "full":
        lgbm_grid = {
            "model__n_estimators": [160, 220, 300],
            "model__learning_rate": [0.03, 0.05, 0.07],
            "model__num_leaves": [31, 47, 63],
            "model__max_depth": [-1, 6, 8],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.6, 0.75, 0.9],
            "model__reg_alpha": [0.0, 0.3, 0.6],
            "model__reg_lambda": [1.0, 2.0, 3.0],
        }
        cat_grid = {
            "model__iterations": [180, 260, 340],
            "model__depth": [5, 6, 8],
            "model__learning_rate": [0.03, 0.05, 0.07],
            "model__l2_leaf_reg": [3.0, 5.0, 7.0, 9.0],
            "model__random_strength": [0.5, 1.0, 1.5],
            "model__subsample": [0.75, 0.9, 1.0],
        }
        xgb_grid = {
            "model__n_estimators": [140, 220, 320],
            "model__max_depth": [3, 4, 5],
            "model__learning_rate": [0.03, 0.05, 0.08],
            "model__subsample": [0.65, 0.8, 1.0],
            "model__colsample_bytree": [0.45, 0.6, 0.8],
            "model__reg_lambda": [8.0, 12.0, 18.0],
            "model__reg_alpha": [2.0, 5.0, 9.0],
            "model__min_child_weight": [6, 10, 14],
            "model__gamma": [0.8, 1.5, 2.2],
        }
        rf_grid = {
            "model__n_estimators": [300, 450, 650],
            "model__max_depth": [8, 11, 14],
            "model__min_samples_leaf": [2, 4, 6],
            "model__min_samples_split": [6, 10, 14],
            "model__max_features": ["sqrt", 0.4, 0.6],
            "model__max_samples": [0.7, 0.85, 1.0],
        }
        lr_grid = {
            "model__C": [0.25, 0.6, 1.0, 1.6],
            "model__l1_ratio": [0.1, 0.3, 0.5, 0.7],
        }
        stack_grid = {
            "model__final_estimator__C": [0.4, 0.8, 1.2, 1.8],
            "model__final_estimator__l1_ratio": [0.1, 0.3, 0.5, 0.7],
        }
    else:
        lgbm_grid = {
            "model__n_estimators": [170, 240],
            "model__learning_rate": [0.03, 0.06],
            "model__num_leaves": [31, 63],
            "model__max_depth": [-1, 7],
            "model__subsample": [0.75, 0.9],
            "model__colsample_bytree": [0.6, 0.8],
            "model__reg_alpha": [0.0, 0.4],
            "model__reg_lambda": [1.0, 2.0],
        }
        cat_grid = {
            "model__iterations": [200, 280],
            "model__depth": [6, 8],
            "model__learning_rate": [0.03, 0.06],
            "model__l2_leaf_reg": [3.0, 6.0],
            "model__random_strength": [0.7, 1.3],
            "model__subsample": [0.8, 1.0],
        }
        xgb_grid = {
            "model__n_estimators": [160, 240],
            "model__max_depth": [3, 4],
            "model__learning_rate": [0.03, 0.06],
            "model__subsample": [0.7, 0.85],
            "model__colsample_bytree": [0.45, 0.6],
            "model__reg_lambda": [10.0, 16.0],
            "model__reg_alpha": [4.0, 8.0],
            "model__min_child_weight": [8, 12],
            "model__gamma": [1.2, 2.2],
        }
        rf_grid = {
            "model__n_estimators": [350, 520],
            "model__max_depth": [9, 12],
            "model__min_samples_leaf": [3, 5],
            "model__min_samples_split": [8, 12],
            "model__max_features": ["sqrt", 0.4],
            "model__max_samples": [0.7, 0.85],
        }
        lr_grid = {
            "model__C": [0.35, 0.8, 1.4],
            "model__l1_ratio": [0.1, 0.3, 0.5],
        }
        stack_grid = {
            "model__final_estimator__C": [0.5, 0.9, 1.4],
            "model__final_estimator__l1_ratio": [0.15, 0.35, 0.55],
        }

    models: dict[str, dict[str, Any]] = {
        "lgbm_gen": {
            "pipeline": base_pipe(
                LGBMClassifier(
                    objective="multiclass",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    verbose=-1,
                )
            ),
            "grid": lgbm_grid,
            "resource": "n_samples",
            "max_resources": "auto",
            "min_resources": 200,
            "single_model": True,
        },
        "catboost_gen": {
            "pipeline": base_pipe(
                CatBoostClassifier(
                    loss_function="MultiClass",
                    eval_metric="MultiClass",
                    random_seed=RANDOM_STATE,
                    verbose=0,
                    od_type="Iter",
                    od_wait=30,
                    thread_count=1,
                    bootstrap_type="Bernoulli",
                )
            ),
            "grid": cat_grid,
            "resource": "n_samples",
            "max_resources": "auto",
            "min_resources": 200,
            "single_model": True,
        },
        "xgb_gen": {
            "pipeline": base_pipe(
                XGBClassifier(
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                )
            ),
            "grid": xgb_grid,
            "resource": "n_samples",
            "max_resources": "auto",
            "min_resources": 200,
            "single_model": True,
        },
        "rf_gen": {
            "pipeline": base_pipe(
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    class_weight="balanced_subsample",
                )
            ),
            "grid": rf_grid,
            "resource": "n_samples",
            "max_resources": "auto",
            "min_resources": 200,
            "single_model": True,
        },
        "lr_enet": {
            "pipeline": base_pipe(
                LogisticRegression(
                    max_iter=800,
                    multi_class="multinomial",
                    solver="saga",
                    penalty="elasticnet",
                    l1_ratio=0.3,
                )
            ),
            "grid": lr_grid,
            "resource": "n_samples",
            "max_resources": "auto",
            "min_resources": 200,
            "single_model": True,
        },
        "stacking_gen": {
            "pipeline": base_pipe(
                StackingClassifier(
                    estimators=base_stack_estimators,
                    final_estimator=LogisticRegression(
                        max_iter=700,
                        multi_class="multinomial",
                        solver="saga",
                        penalty="elasticnet",
                        l1_ratio=0.3,
                    ),
                    stack_method="predict_proba",
                    passthrough=True,
                    cv=cv,
                    n_jobs=-1,
                )
            ),
            "grid": stack_grid,
            "resource": "n_samples",
            "max_resources": "auto",
            "min_resources": 200,
            "single_model": False,
        },
    }

    return models


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
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=RANDOM_STATE)
    cache_dir = results_dir / ".cache"
    model_specs = build_model_bundle(
        preprocessor=preprocessor,
        player_slot_cols=player_slot_cols,
        min_category_freq=args.min_category_freq,
        max_player_categories=args.max_player_categories,
        cv=cv,
        search_mode=args.search_mode,
        cache_dir=cache_dir,
    )

    best_models: dict[str, Any] = {}
    metrics: dict[str, dict[str, float]] = {}

    single_items = [(k, v) for k, v in model_specs.items() if v["single_model"]]
    pbar = tqdm(single_items, total=len(single_items), desc="Phase 5 single-model tuning", unit="model")
    for name, spec in pbar:
        pbar.set_postfix_str(name)
        search = HalvingRandomSearchCV(
            estimator=spec["pipeline"],
            param_distributions=spec["grid"],
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            factor=2,
            resource=spec["resource"],
            min_resources=spec["min_resources"],
            max_resources=spec["max_resources"],
            refit=True,
            verbose=0,
        )
        search.fit(X_train, y_train)

        best = search.best_estimator_
        m = evaluate_model(best, X_train, y_train, X_test, y_test, n_classes=len(label_encoder.classes_))
        m["best_cv_accuracy"] = float(search.best_score_)

        best_models[name] = {"pipeline": best, "best_params": search.best_params_}
        metrics[name] = m

    # Build stacking from already tuned base models to avoid costly base-model CV inside search.
    stack_spec = model_specs["stacking_gen"]
    tuned_estimators = [
        ("xgb", best_models["xgb_gen"]["pipeline"].named_steps["model"]),
        ("rf", best_models["rf_gen"]["pipeline"].named_steps["model"]),
        ("lr", best_models["lr_enet"]["pipeline"].named_steps["model"]),
    ]
    stack_pipe = Pipeline(
        memory=joblib.Memory(location=cache_dir, verbose=0),
        steps=[
            (
                "card",
                CardinalityReducer(
                    player_slot_cols,
                    min_frequency=args.min_category_freq,
                    max_categories=args.max_player_categories,
                ),
            ),
            ("preprocessor", preprocessor),
            (
                "model",
                StackingClassifier(
                    estimators=tuned_estimators,
                    final_estimator=LogisticRegression(
                        max_iter=700,
                        multi_class="multinomial",
                        solver="saga",
                        penalty="elasticnet",
                        l1_ratio=0.3,
                    ),
                    stack_method="predict_proba",
                    passthrough=True,
                    cv=cv,
                    n_jobs=-1,
                ),
            ),
        ],
    )

    stack_search = HalvingRandomSearchCV(
        estimator=stack_pipe,
        param_distributions=stack_spec["grid"],
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        factor=2,
        resource=stack_spec["resource"],
        min_resources=stack_spec["min_resources"],
        max_resources=stack_spec["max_resources"],
        refit=True,
        verbose=0,
    )
    stack_search.fit(X_train, y_train)

    stack_best = stack_search.best_estimator_
    stack_metrics = evaluate_model(stack_best, X_train, y_train, X_test, y_test, n_classes=len(label_encoder.classes_))
    stack_metrics["best_cv_accuracy"] = float(stack_search.best_score_)

    best_models["stacking_gen"] = {
        "pipeline": stack_best,
        "best_params": stack_search.best_params_,
    }
    metrics["stacking_gen"] = stack_metrics

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index").reset_index(names=["model"])
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
    best_name = str(candidate_df.loc[0, "model"])

    metrics_df = metrics_df.sort_values(sort_cols, ascending=False).reset_index(drop=True)
    metrics_df["selected_for_deploy"] = metrics_df["model"] == best_name
    best_bundle = best_models[best_name]
    best_model = best_bundle["pipeline"]

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)
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
    m_sorted = metrics_df.sort_values("test_accuracy", ascending=False)
    sns.barplot(data=m_sorted, x="model", y="test_accuracy", hue="model", legend=False, palette="Set2")
    plt.title("Phase 5: Test Accuracy")
    plt.tight_layout()
    plt.savefig(results_dir / "phase5_model_comparison_accuracy.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(data=m_sorted, x="model", y="test_f1_weighted", hue="model", legend=False, palette="Blues")
    plt.title("Phase 5: Test Weighted F1")
    plt.tight_layout()
    plt.savefig(results_dir / "phase5_model_comparison_f1.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(data=m_sorted, x="model", y="fit_gap_weighted_f1", hue="model", legend=False, palette="rocket")
    plt.title("Phase 5: Overfit Gap")
    plt.tight_layout()
    plt.savefig(results_dir / "phase5_overfit_gap.png", dpi=160)
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
    ax.set_title(f"Phase 5 Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(results_dir / "phase5_best_model_confusion_matrix.png", dpi=160)
    plt.close()

    save_reliability_plot(y_test, y_proba, results_dir / "phase5_best_model_reliability_plot.png")

    monitoring = {
        "train_years": sorted(train_df["Date"].dt.year.unique().tolist()),
        "test_years": sorted(test_df["Date"].dt.year.unique().tolist()),
        "drift_report": compute_drift_report(train_df, test_df, player_slot_cols),
    }

    metrics_df.to_csv(results_dir / "phase5_model_comparison_metrics.csv", index=False)
    with (results_dir / "phase5_best_model_classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with (results_dir / "phase5_model_monitoring_report.json").open("w", encoding="utf-8") as f:
        json.dump(monitoring, f, indent=2)

    metadata = {
        "phase": "5",
        "train_file": str(input_path),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "test_year": int(args.test_year),
        "drop_no_result": bool(args.drop_no_result),
        "search_mode": args.search_mode,
        "cv_folds": int(args.cv_folds),
        "selection_metric": args.selection_metric,
        "max_fit_gap": None if args.max_fit_gap is None else float(args.max_fit_gap),
        "min_category_freq": int(args.min_category_freq),
        "max_player_categories": int(args.max_player_categories),
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

    joblib.dump(bundle, artifacts_dir / "phase5_ipl_winner_best_pipeline.joblib")
    with (artifacts_dir / "phase5_ipl_winner_best_pipeline.pkl").open("wb") as f:
        pickle.dump(bundle, f)
    with (artifacts_dir / "phase5_model_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    sample_payload = X_test.iloc[0].to_dict()
    with (results_dir / "phase5_sample_prediction_payload.json").open("w", encoding="utf-8") as f:
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
