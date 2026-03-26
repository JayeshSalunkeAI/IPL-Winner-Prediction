
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

def train_final_model():
    print("Initializing FINAL Phase Training...")
    
    # Paths
    data_path = Path("data/processed/phase61_dataset.csv")
    artifacts_dir = Path("artifacts/final_phase")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Data
    df = pd.read_csv(data_path)
    
    # Filter Recent Data (2020+) to ensure relevance
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
    
    df_recent = df[df['date'].dt.year >= 2020].copy()
    print(f"Total Matches: {len(df)}. Recent Matches (2020+): {len(df_recent)}")
    
    # Features (Same as Phase 6.1)
    features = [
        "team1_form_winrate_5", "team2_form_winrate_5",
        "venue_score_prior", "venue_chase_winrate_prior",
        "toss_advantage", "is_high_dew",
        "t1_bat_avg", "t1_bat_sr", "t1_bowl_eco", "t1_bowl_sr",
        "t2_bat_avg", "t2_bat_sr", "t2_bowl_eco", "t2_bowl_sr",
        "pitch_type", "bounce_and_carry", 
        "toss_winner", "toss_decision", "team1", "team2"
    ]
    
    # Handle Venue/Stadium
    venue_col = 'stadium' if 'stadium' in df.columns else 'venue'
    if venue_col not in features: features.append(venue_col)
    
    features = [f for f in features if f in df_recent.columns]
    target = "target"
    
    X = df_recent[features]
    y = df_recent[target]
    
    # Categorical Features
    cat_cols = ["pitch_type", "bounce_and_carry", "toss_winner", "toss_decision", "team1", "team2", venue_col]
    cat_cols = [c for c in cat_cols if c in X.columns]
    
    # Fill Unknowns
    X = X.copy()
    X[cat_cols] = X[cat_cols].fillna('Unknown')
    for c in cat_cols:
        X[c] = X[c].astype(str)
        
    # Split Data (Time Series Split: Last 20% for Final Holdout)
    split_idx = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train Size: {len(X_train)}, Test Size: {len(X_test)}")
    
    # Best Params from Phase 6.1 (CatBoost)
    # {'iterations': 855, 'depth': 9, 'learning_rate': 0.0205, 'l2_leaf_reg': 8.26}
    best_params = {
        'iterations': 1000, # Increased slightly for robustness
        'depth': 9,
        'learning_rate': 0.02,
        'l2_leaf_reg': 8.26,
        'loss_function': 'Logloss',
        'cat_features': cat_cols,
        'verbose': 100,
        'early_stopping_rounds': 50,
        'random_seed': 42
    }
    
    # Train Model
    print("\nTraining Final CatBoost Model...")
    model = CatBoostClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=(X_test, y_test))
    
    # Evaluate
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    
    print(f"\nFinal Model Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))
    
    # --- Generate Plots ---
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'])
    plt.title(f'Confusion Matrix (Acc: {acc:.2%})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(artifacts_dir / "confusion_matrix.png")
    plt.close()
    
    # 2. Feature Importance
    plt.figure(figsize=(10, 8))
    feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    feat_imp = feat_imp.sort_values('Importance', ascending=False).head(20)
    sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis')
    plt.title('Top 20 Important Features')
    plt.tight_layout()
    plt.savefig(artifacts_dir / "feature_importance.png")
    plt.close()
    
    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(artifacts_dir / "roc_curve.png")
    plt.close()
    
    # Save Model & Metadata
    model_path = artifacts_dir / "final_model_catboost.joblib"
    joblib.dump(model, model_path)
    
    metadata = {
        "accuracy": acc,
        "params": best_params,
        "features": features,
        "cat_cols": cat_cols,
        "roc_auc": roc_auc,
        "description": "Final CatBoost Model (Phase 7) trained on post-2020 data with complete granular features."
    }
    
    with open(artifacts_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\nArtifacts Saved to {artifacts_dir}")
    print("- Confusion Matrix")
    print("- Feature Importance Plot")
    print("- ROC Curve")
    print("- Model Joblib")
    print("- Metadata JSON")

if __name__ == "__main__":
    train_final_model()
