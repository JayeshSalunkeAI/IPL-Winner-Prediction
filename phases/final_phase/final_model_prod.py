
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier

def train_final_production():
    print("Initializing FINAL PRODUCTION Training (Fast Mode)...")
    
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
    
    X = df_recent[features]
    y = df_recent["target"]
    
    cat_cols = [c for c in ["pitch_type", "bounce_and_carry", "toss_winner", "toss_decision", "team1", "team2", venue_col] if c in X.columns]
    
    X = X.copy()
    X[cat_cols] = X[cat_cols].fillna('Unknown')
    for c in cat_cols: X[c] = X[c].astype(str)

    # 1. Generate Evaluation Metrics via Single Split (Faster than 5-Fold CV for artifacts)
    # We use Stratified Split 80/20 to give a fair report
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42)
    
    cb_params = {
        'iterations': 800,
        'depth': 6,
        'learning_rate': 0.03,
        'l2_leaf_reg': 5.0,
        'loss_function': 'Logloss',
        'cat_features': cat_cols,
        'verbose': 0,
        'random_seed': 42
    }
    
    model = CatBoostClassifier(**cb_params)
    print("Training Validation Model...")
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50, verbose=0)
    
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    
    print(f"\nValidation Accuracy (80/20 Split): {acc:.4f}")
    print(classification_report(y_test, preds))
    
    # 2. Generate Plots
    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'])
    plt.title(f'Confusion Matrix (Acc: {acc:.2%})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(artifacts_dir / "confusion_matrix.png")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(artifacts_dir / "roc_curve.png")
    
    # Feature Importance
    feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    feat_imp = feat_imp.sort_values('Importance', ascending=False).head(20)
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(artifacts_dir / "feature_importance.png")
    plt.close('all') # Close all figs
    
    # 3. Train PROD Model (Full Data)
    print("\nTraining Final Production Model on ALL Recent Data...")
    prod_model = CatBoostClassifier(**cb_params)
    prod_model.fit(X, y, verbose=0) # No early stopping, use full iterations or trusted params
    
    # Save
    model_path = artifacts_dir / "final_model_catboost_prod.joblib"
    joblib.dump(prod_model, model_path)
    
    metadata = {
        "validation_accuracy": acc,
        "roc_auc": roc_auc,
        "features": features,
        "cat_cols": cat_cols,
        "params": cb_params,
        "description": "Production CatBoost Model trained on 100% of post-2020 data."
    }
    with open(artifacts_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Artifacts saved to {artifacts_dir}")

if __name__ == "__main__":
    train_final_production()
