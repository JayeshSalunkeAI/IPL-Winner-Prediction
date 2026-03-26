
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def train_final_ensemble():
    print("Initializing FINAL ROBUST (Ensemble) Phase...")
    
    # Paths
    data_path = Path("data/processed/phase61_dataset.csv")
    artifacts_dir = Path("artifacts/final_phase")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Data
    df = pd.read_csv(data_path)
    
    # Filter Recent Data (2020+) 
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
    
    df_recent = df[df['date'].dt.year >= 2020].copy()
    print(f"Total Matches: {len(df)}. Recent Matches (2020+): {len(df_recent)}")
    
    # Features
    features = [
        "team1_form_winrate_5", "team2_form_winrate_5",
        "venue_score_prior", "venue_chase_winrate_prior",
        "toss_advantage", "is_high_dew",
        "t1_bat_avg", "t1_bat_sr", "t1_bowl_eco", "t1_bowl_sr",
        "t2_bat_avg", "t2_bat_sr", "t2_bowl_eco", "t2_bowl_sr",
        "pitch_type", "bounce_and_carry", 
        "toss_winner", "toss_decision", "team1", "team2"
    ]
    
    venue_col = 'stadium' if 'stadium' in df.columns else 'venue'
    if venue_col not in features: features.append(venue_col)
    
    features = [f for f in features if f in df_recent.columns]
    target = "target"
    
    X = df_recent[features]
    y = df_recent[target]
    
    # Identify Categorical Columns
    cat_cols = ["pitch_type", "bounce_and_carry", "toss_winner", "toss_decision", "team1", "team2", venue_col]
    cat_cols = [c for c in cat_cols if c in X.columns]
    
    # Preprocessing
    X = X.copy()
    X[cat_cols] = X[cat_cols].fillna('Unknown')
    for c in cat_cols:
        X[c] = X[c].astype(str)
        
    print(f"Dataset Shape: {X.shape}")

    # --- Model Definitions ---
    
    # 1. CatBoost (Categorical Optimized)
    # Using best params from Phase 6.1
    cb_params = {
        'iterations': 900, 
        'depth': 9,
        'learning_rate': 0.02,
        'l2_leaf_reg': 8.0,
        'loss_function': 'Logloss',
        'cat_features': cat_cols,
        'verbose': 0,
        'random_seed': 42
    }
    catboost = CatBoostClassifier(**cb_params)
    
    # 2. XGBoost (Gradient Boosting Standard)
    # Needs OneHotEncoding, but we'll use 'enable_categorical=True' for modern XGBoost
    # Or pipeline it. For ensemble simplicity, let's use CatBoost's robust handling 
    # and a separate pipeline for XGB if needed. 
    # Actually, let's stick to CatBoost + CatBoost (Variation) to keep pipeline simple 
    # regarding categoricals, OR better: VotingClassifier doesn't easily mix native CatBoost with sklearn Pipeline 
    # unless wrapped carefully.
    
    # SIMPLIFICATION FOR ROBUSTNESS:
    # Use Single Best CatBoost but trained on K-Fold logic to get accurate metric, then whole data.
    # VotingClassifier with CatBoost is proving slow/unstable in this env.
    
    cb_params = {
        'iterations': 500, # Reduce for speed, 500 is plenty for 300 samples
        'depth': 6, # Lower depth to prevent overfit
        'learning_rate': 0.03,
        'l2_leaf_reg': 5.0,
        'loss_function': 'Logloss',
        'cat_features': cat_cols,
        'verbose': 0,
        'random_seed': 42
    }
    
    ensemble = CatBoostClassifier(**cb_params)
    
    # --- Evaluation via Cross-Validation (ROBUST METRIC) ---
    print("\nRunning 5-Fold Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Generate predictions for the entire dataset via CV (out-of-fold)
    y_pred = cross_val_predict(ensemble, X, y, cv=cv, method='predict')
    y_proba = cross_val_predict(ensemble, X, y, cv=cv, method='predict_proba')[:, 1]
    
    acc = accuracy_score(y, y_pred)
    # print(f"\nRobust Cross-Validation Accuracy: {acc:.4f}") <- This line was missing indentation or not matching context
    print(f"\nRobust Cross-Validation Accuracy: {acc:.4f}")
    print(classification_report(y, y_pred))
    
    # --- Final Training on FULL DATA (For Artifact) ---
    print("Retraining on Full Recent Data (for Deployment)...")
    ensemble.fit(X, y)
    
    # --- Generate Plots (Based on CV Predictions) ---
    # 1. Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'])
    plt.title(f'CV Confusion Matrix (Acc: {acc:.2%})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(artifacts_dir / "confusion_matrix.png")
    plt.close()
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (CV)')
    plt.legend(loc="lower right")
    plt.savefig(artifacts_dir / "roc_curve.png")
    plt.close()
    
    # 3. Feature Importance
    feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': ensemble.feature_importances_})
    feat_imp = feat_imp.sort_values('Importance', ascending=False).head(20)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(artifacts_dir / "feature_importance.png")
    plt.close()
    
    # Save Model & Metadata
    model_path = artifacts_dir / "final_model_catboost_robust.joblib"
    joblib.dump(ensemble, model_path)
    
    metadata = {
        "accuracy_cv": acc,
        "roc_auc_cv": roc_auc,
        "model_type": "CatBoost",
        "features": features,
        "cat_cols": cat_cols,
        "description": "Final Robust CatBoost Model. Metrics are 5-Fold Stratified CV."
    }
    
    with open(artifacts_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\nSUCCESS: Robust Ensemble Model Saved to {model_path}")

if __name__ == "__main__":
    train_final_ensemble()
