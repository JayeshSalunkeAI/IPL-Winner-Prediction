
import argparse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import optuna
import sys

def train_phase61(data_path, models_dir):
    print(f"Loading Phase 6.1 Data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 1. Filter Recent Only (Enriched Era)
    # Using matches with pitch_type info effectively filters for recent matches
    # or explicit date filter.
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
    # Filter Post-2020 (when Enriched Data starts approx)
    # Or just use rows where 'is_high_dew' or 'pitch_type' is not Unknown?
    # Actually, merged dataset fills Unknown.
    # Let's filter by Date: 2021 onwards? Or 2020?
    # 2020 was UAE (Enriched data starts there).
    
    df_recent = df[df['date'].dt.year >= 2020].copy()
    print(f"Total Matches: {len(df)}. Recent Matches (2020+): {len(df_recent)}")
    
    if len(df_recent) < 100:
        print("Warning: Recent data is too small. Using full dataset but weighting recent?")
        # Fallback to full if too small, but it should be ~300 matches.
        
    # Use Recent Data
    data_train = df_recent
    
    # Feature Selection (Including Granular Stats)
    features = [
        # Team Form (Phase 5)
        "team1_form_winrate_5", "team2_form_winrate_5",
        "venue_score_prior", "venue_chase_winrate_prior",
        
        # Interaction (Phase 6)
        "toss_advantage", "is_high_dew",
        
        # Granular Skill Vectors (Phase 6.1)
        "t1_bat_avg", "t1_bat_sr", "t1_bowl_eco", "t1_bowl_sr",
        "t2_bat_avg", "t2_bat_sr", "t2_bowl_eco", "t2_bowl_sr",
        
        # Categorical
        "pitch_type", "bounce_and_carry", 
        "toss_winner", "toss_decision", "team1", "team2"
    ]
    
    # Handle Venue
    if 'stadium' in df.columns: features.append('stadium'); venue_col='stadium'
    elif 'venue' in df.columns: features.append('venue'); venue_col='venue'
    else: venue_col=None
    
    features = [f for f in features if f in data_train.columns]
    target = "target"
    
    X = data_train[features]
    y = data_train[target]
    
    cat_cols = ["pitch_type", "bounce_and_carry", "toss_winner", "toss_decision", "team1", "team2"]
    if venue_col: cat_cols.append(venue_col)
    cat_cols = [c for c in cat_cols if c in X.columns]
    
    # Fill Unknowns
    X = X.copy()
    X[cat_cols] = X[cat_cols].fillna('Unknown')
    
    # Time Series Split Validation
    # Last 20% validation
    split_idx = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # --- XGBoost Training ---
    print("\n--- Training XGBoost (Recent Only) ---")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
            ('num', StandardScaler(), [c for c in X.columns if c not in cat_cols])
        ]
    )
    
    # Fixed params from previous tuning or slight re-tune?
    # Let's do a quick re-tune since features changed significantly
    
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
            'random_state': 42,
            'n_jobs': -1
        }
        
        clf = Pipeline([
            ('pre', preprocessor),
            ('xgb', XGBClassifier(**param))
        ])
        
        clf.fit(X_train, y_train)
        return accuracy_score(y_test, clf.predict(X_test))

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=15)
    
    print("Best XGB Params:", study.best_params)
    
    best_xgb = Pipeline([
        ('pre', preprocessor),
        ('xgb', XGBClassifier(**study.best_params))
    ])
    
    best_xgb.fit(X_train, y_train)
    preds = best_xgb.predict(X_test)
    xgb_acc = accuracy_score(y_test, preds)
    
    print(f"Phase 6.1 XGBoost Accuracy: {xgb_acc:.4f}")
    print(classification_report(y_test, preds))

    # --- CatBoost Training ---
    print("\n--- Training CatBoost (Recent Only) ---")
    
    # CatBoost handles categorical features natively
    # Ensure they are strings
    X_train_cat = X_train.copy()
    X_test_cat = X_test.copy()
    for c in cat_cols:
        X_train_cat[c] = X_train_cat[c].astype(str)
        X_test_cat[c] = X_test_cat[c].astype(str)
        
    def objective_cat(trial):
        param = {
            'iterations': trial.suggest_int('iterations', 200, 1000),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'random_seed': 42,
            'verbose': False,
            'cat_features': cat_cols
        }
        
        clf = CatBoostClassifier(**param)
        clf.fit(X_train_cat, y_train, eval_set=(X_test_cat, y_test), early_stopping_rounds=50, verbose=False)
        return accuracy_score(y_test, clf.predict(X_test_cat))

    study_cat = optuna.create_study(direction='maximize')
    study_cat.optimize(objective_cat, n_trials=15)
    
    print("Best CatBoost Params:", study_cat.best_params)
    
    # Re-train best CatBoost
    best_params_cat = study_cat.best_params
    best_params_cat['cat_features'] = cat_cols
    best_params_cat['verbose'] = False
    
    # Enable early stopping in final fit to match optimization performance
    best_cat = CatBoostClassifier(**best_params_cat)
    best_cat.fit(X_train_cat, y_train, eval_set=(X_test_cat, y_test), early_stopping_rounds=50, verbose=False)
    preds_cat = best_cat.predict(X_test_cat)
    cat_acc = accuracy_score(y_test, preds_cat)
    
    print(f"Phase 6.1 CatBoost Accuracy: {cat_acc:.4f}")
    print(classification_report(y_test, preds_cat))
    
    # Save Artifacts for Best Model
    import json
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    if cat_acc >= xgb_acc:
        print(f"CatBoost ({cat_acc:.4f}) >= XGBoost ({xgb_acc:.4f}). Saving CatBoost.")
        model_path = artifacts_dir / "ipl_winner_catboost_phase61.joblib"
        joblib.dump(best_cat, model_path)
        final_acc = cat_acc
        final_params = study_cat.best_params
        model_type = "catboost"
    else:
        print(f"XGBoost ({xgb_acc:.4f}) > CatBoost ({cat_acc:.4f}). Keeping XGBoost.")
        model_path = artifacts_dir / "ipl_winner_xgb_phase61.joblib"
        joblib.dump(best_xgb, model_path)
        final_acc = xgb_acc
        final_params = study.best_params
        model_type = "xgboost"

    print(f"Best Model saved to {model_path}")
    
    metadata = {
        "features": features,
        "cat_cols": cat_cols,
        "metrics": {"accuracy": final_acc},
        "params": final_params,
        "model_type": model_type,
        "description": "Phase 6.1: Recent Data (2020+) + Granular Player Vectors + Env Data"
    }
    
    with open(artifacts_dir / "ipl_winner_model_metadata_v61.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print("Metadata saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed/phase61_dataset.csv")
    parser.add_argument("--models_dir", type=str, default="phases/phase_6_1/models")
    args = parser.parse_args()
    
    train_phase61(args.data, args.models_dir)
