import argparse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import optuna

def train_phase6(data_path, models_dir):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Sort by date for time-series split
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
    
    # Feature Mapping (Phase 5 -> Phase 6)
    # The merged dataset uses snake_case phase 5 names
    # Note: Column names in CSV are lowercase because of phase6_merge.py
    features = [
        "team1_form_winrate_5", "team2_form_winrate_5",
        "venue_score_prior", "venue_chase_winrate_prior",
        "toss_advantage", "is_high_dew",
        "pitch_type", "bounce_and_carry", 
        "toss_winner", "toss_decision", "team1", "team2"
    ]
    
    # Handle Venue Name
    if 'stadium' in df.columns:
        features.append('stadium')
        venue_col = 'stadium'
    elif 'venue' in df.columns:
        features.append('venue')
        venue_col = 'venue'
    else:
        venue_col = None
        print("Warning: No venue column found.")

    target = "target"
    
    # Filter columns that exist
    features = [f for f in features if f in df.columns]
    
    X = df[features]
    y = df[target]
    
    # Identify Categorical Columns
    cat_cols = ["pitch_type", "bounce_and_carry", "toss_winner", "toss_decision", "team1", "team2"]
    if venue_col:
        cat_cols.append(venue_col)
        
    cat_cols = [c for c in cat_cols if c in X.columns]
    
    # Handle Missing Values in Categorical Features
    X = X.copy()
    X[cat_cols] = X[cat_cols].fillna('Unknown')
    
    # Train/Test Split (Last 15%)
    split_idx = int(len(df) * 0.85)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Features: {X.columns.tolist()}")

    # --- CATBOOST ---
    print("\n--- Training CatBoost ---")
    model_cb = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.02, # Lower LR for better generalization
        depth=6,
        loss_function='Logloss',
        cat_features=cat_cols,
        verbose=100,
        early_stopping_rounds=100,
        random_seed=42,
        allow_writing_files=False
    )
    
    model_cb.fit(X_train, y_train, eval_set=(X_test, y_test))
    
    preds_cb = model_cb.predict(X_test)
    acc_cb = accuracy_score(y_test, preds_cb)
    print(f"CatBoost Accuracy: {acc_cb:.4f}")
    print(classification_report(y_test, preds_cb))
    
    # Save CatBoost
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    model_cb.save_model(str(Path(models_dir) / "catboost_model.cbm"))
    
    # --- XGBOOST (Optuna) ---
    print("\n--- Tuning XGBoost ---")
    
    # Preprocessing for XGB (Encoding)
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ],
        remainder='passthrough'
    )
    
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'n_jobs': -1,
            'random_state': 42
        }
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(**param))
        ])
        
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        return accuracy_score(y_test, preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20) 
    
    print("Best XGB Params:", study.best_params)
    
    best_xgb = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(**study.best_params))
    ])
    best_xgb.fit(X_train, y_train)
    preds_xgb = best_xgb.predict(X_test)
    acc_xgb = accuracy_score(y_test, preds_xgb)
    print(f"XGBoost Accuracy: {acc_xgb:.4f}")
    print(classification_report(y_test, preds_xgb))
    
    joblib.dump(best_xgb, Path(models_dir) / "xgb_pipeline.joblib")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed/phase6_dataset_v2.csv")
    parser.add_argument("--models_dir", type=str, default="phases/phase_6/models")
    args = parser.parse_args()
    
    train_phase6(args.data, args.models_dir)
