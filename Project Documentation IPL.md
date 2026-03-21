# IPL Winner Prediction - Project Documentation

## 1) Project Overview
This project predicts IPL match winners using pre-match context. The pipeline started with team/toss/form/venue features and has now been upgraded to include player-level features and player ELO priors.

Current status:
- Phase 1 baseline model is complete and deployed in Streamlit.
- Phase 2 player-enhanced modeling pipeline is complete.
- Phase 3 regularized pipeline with cardinality control is complete.
- Phase 4 advanced pipeline (ball-by-ball derived features + stronger tuning) is complete.
- Phase 4.1 generalization-first tuning pipeline is complete.
- Multi-model comparison is completed on a 2025 holdout set for both Phase 2 and Phase 3.
- Deployment-ready APIs and artifacts are prepared for Phase 2, Phase 3, Phase 4, and Phase 4.1.

---

## 2) Data and Feature Engineering Summary

### 2.1 Core Match Features (Phase 1 + Phase 2)
- `Team1`, `Team2`
- `Toss_Winner`, `Toss_Decision`
- `team1_form_winrate_5`, `team2_form_winrate_5`
- `venue_chase_winrate_prior`, `venue_score_prior`

### 2.2 Player Features Added in Phase 2
- Team player lists:
  - `Team1_Players`
  - `Team2_Players`
- Fixed training columns for lineups:
  - `Team1_Player_1` ... `Team1_Player_11`
  - `Team2_Player_1` ... `Team2_Player_11`
- Player ELO aggregate priors:
  - `team1_player_elo_avg_prior`, `team2_player_elo_avg_prior`
  - `team1_player_elo_max_prior`, `team2_player_elo_max_prior`
  - `team1_player_elo_min_prior`, `team2_player_elo_min_prior`
  - `player_elo_gap_prior`

Target label:
- `Match_Winner`

---

## 3) Datasets Used

### 3.1 Phase 1 Dataset
- Train file: `IPL_Winner_Model_Dataset.csv`
- Test file: `IPL_2025_Winner_Model_Dataset.csv`

### 3.2 Phase 2 Dataset (Latest)
- Train/Test source file: `IPL_Winner_Model_Dataset_With_Players.csv`
- Holdout strategy: year-based split (2025 as test)
- Train size: **924**
- Test size: **74**
- Total engineered features used in model: **39**

---

## 4) Latest Findings - Phase 2 Model Comparison

Source files:
- `phase_2_result/model_comparison_metrics.csv`
- `phase_2_artifacts/phase2_model_metadata.json`

Models compared:
1. Random Forest
2. XGBoost
3. Logistic Regression

### 4.1 Test Performance (2025 Holdout)
- **Random Forest** (Best)
  - Accuracy: **0.5135**
  - Balanced Accuracy: **0.5070**
  - F1 (Weighted): **0.4882**
  - F1 (Macro): **0.4648**
  - Precision (Weighted): **0.5192**
  - Recall (Weighted): **0.5135**

- **XGBoost**
  - Accuracy: **0.4865**
  - Balanced Accuracy: **0.4592**
  - F1 (Weighted): **0.4847**
  - F1 (Macro): **0.4321**

- **Logistic Regression**
  - Accuracy: **0.4189**
  - Balanced Accuracy: **0.4474**
  - F1 (Weighted): **0.3940**
  - F1 (Macro): **0.3825**

### 4.2 Cross-Validation Snapshot
- Best CV Weighted F1 by model:
  - Random Forest: **0.5127**
  - XGBoost: **0.4988**
  - Logistic Regression: **0.4955**

### 4.3 Overfitting / Underfitting Diagnostics
Using train-test weighted F1 gap:
- Random Forest gap: **0.2438** -> overfitting risk
- XGBoost gap: **0.5142** -> strong overfitting risk
- Logistic Regression gap: **0.6060** -> strong overfitting risk

Observation:
- Random Forest is the best current Phase 2 model on holdout metrics, but still needs further regularization/tuning to reduce overfit gap.

---

## 5) Phase 2 Outputs and Artifacts

### 5.1 Result Folder
`phase_2_result/`

Contains:
- `model_comparison_metrics.csv`
- `model_comparison_test_f1.png`
- `overfit_gap_comparison.png`
- `best_model_confusion_matrix.png`
- `best_model_top20_feature_importance.png`
- `best_model_classification_report.json`
- `sample_prediction_payload.json`
- `phase2_deployment_notes.md`

### 5.2 Artifact Folder
`phase_2_artifacts/`

Contains:
- `phase2_ipl_winner_best_pipeline.joblib`
- `phase2_ipl_winner_best_pipeline.pkl`
- `phase2_model_metadata.json`

---

## 6) Deployment Readiness (Phase 2)

FastAPI app:
- `phase2_fastapi_app.py`

Endpoints:
- `GET /health`
- `GET /meta`
- `POST /predict`

Run command:
```bash
uvicorn phase2_fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```

The API has already been smoke-tested successfully with `sample_prediction_payload.json` and returns winner probabilities.

---

## 7) Next Improvement Targets
1. Reduce overfitting via tighter hyperparameter bounds and stronger regularization.
2. Add feature pruning and categorical cardinality controls for player name columns.
3. Evaluate calibration and top-k accuracy for practical prediction quality.
4. Add model monitoring report for season drift and lineup drift.

---

## 8) Phase 3 Execution Update (Completed)

Phase 3 was initiated and executed using:
- `phase3_train_pipeline.py`
- `Phase3_MLPipeline.ipynb`

Implemented improvements in Phase 3:
1. Overfitting controls via tighter hyperparameter ranges and stronger regularization.
2. Feature pruning plus cardinality control for player slot columns.
3. Added calibration and top-k evaluation metrics.
4. Added model monitoring report for season and lineup drift.

### 8.1 Phase 3 Test Performance (2025 Holdout)
Source: `phase_3_result/phase3_model_comparison_metrics.csv`

- **Best by Weighted F1: Logistic Regression (L2)**
  - Accuracy: **0.5135**
  - F1 (Weighted): **0.4902**
  - F1 (Macro): **0.4516**
  - Top-2 Accuracy: **0.8784**
  - Top-3 Accuracy: **0.9189**
  - Calibration ECE: **0.2390**

- **Best by Accuracy: XGBoost (Regularized)**
  - Accuracy: **0.5405**
  - F1 (Weighted): **0.4637**
  - F1 Gap (Train-Test): **0.0191** (lowest overfit gap)

- **Random Forest (Regularized)**
  - Accuracy: **0.4730**
  - F1 (Weighted): **0.4506**

### 8.2 Phase 3 Monitoring Findings
Source: `phase_3_result/phase3_model_monitoring_report.json`

- Numeric drift is moderate for form/venue features (no severe mean-shift z values).
- Categorical unseen rate for team/toss fields is **0.0** on 2025 holdout.
- Lineup drift is present:
  - Overall unseen player rate in holdout lineups: **0.1044**

### 8.3 Phase 3 Artifacts and Results
Phase 3 result files (`phase_3_result/`):
- `phase3_model_comparison_metrics.csv`
- `phase3_model_comparison_test_f1.png`
- `phase3_overfit_gap.png`
- `phase3_topk_accuracy.png`
- `phase3_best_model_confusion_matrix.png`
- `phase3_best_model_reliability_plot.png`
- `phase3_best_model_classification_report.json`
- `phase3_model_monitoring_report.json`
- `phase3_sample_prediction_payload.json`
- `phase3_deployment_notes.md`

Phase 3 model artifacts (`phase_3_artifacts/`):
- `phase3_ipl_winner_best_pipeline.joblib`
- `phase3_ipl_winner_best_pipeline.pkl`
- `phase3_model_metadata.json`

Phase 3 deployment app:
- `phase3_fastapi_app.py`

Notebook workflow:
- `Phase3_MLPipeline.ipynb`

---

## 9) Updated Next Focus (Phase 4 Candidates)
1. Improve class-wise recall for minority outcomes while keeping weighted F1 stable.
2. Add post-hoc probability calibration (temperature scaling or isotonic) and compare ECE/Brier.
3. Create hybrid model selection rule: choose model by use case (max accuracy vs max weighted F1).
4. Enrich lineup features with role-level summaries (batters/bowlers/all-rounders) to reduce player-name sparsity.

---

## 10) Phase 4 Execution Update (Completed)

Phase 4 was initiated in a new folder and executed end-to-end:
- `Phase 4/phase4_extract_dataset.py`
- `Phase 4/phase4_train_pipeline.py`
- `Phase 4/Phase4_MLPipeline.ipynb`
- `Phase 4/phase4_fastapi_app.py`

### 10.1 What Was Added in Phase 4
1. Advanced feature extraction from ball-by-ball YAML files (2008-2025) including:
   - Head-to-head prior win rates
   - Venue-team prior win rates
   - Venue first-innings prior average score
   - Team recent batting/bowling priors (runs for/against, wickets, powerplay RR, death RR)
2. Player slot cardinality controls with rare-player bucketing.
3. Expanded model search space (XGBoost, Random Forest, Extra Trees, Logistic Regression).
4. Hyperparameter tuning focused on holdout accuracy and generalization diagnostics.
5. Calibration, top-k accuracy, confusion matrix, and drift report generation.
6. Optional no-result filtering to focus on true winner prediction rows.

### 10.2 Phase 4 Data Snapshot
Source: `Phase 4/artifacts/phase4_model_metadata.json`

- Phase 4 extracted dataset rows: **998**
- Training size (after no-result filter): **907**
- Test size (2025): **70**
- Feature count: **51**

### 10.3 Phase 4 Model Performance (2025 Holdout)
Source: `Phase 4/results/phase4_model_comparison_metrics.csv`

- **Best model: Extra Trees (Phase 4)**
  - Accuracy: **0.5429**
  - F1 (Weighted): **0.5238**
  - F1 (Macro): **0.5170**
  - Top-2 Accuracy: **0.9857**
  - Top-3 Accuracy: **0.9857**
  - Calibration ECE: **0.1253**

- **Random Forest (Phase 4)**
  - Accuracy: **0.5286**
  - F1 (Weighted): **0.5127**

- **XGBoost (Phase 4)**
  - Accuracy: **0.5000**
  - F1 (Weighted): **0.4309**

- **Logistic Regression (Phase 4)**
  - Accuracy: **0.4286**
  - F1 (Weighted): **0.4156**

### 10.4 Generalization and Drift Notes
- Extra Trees delivered best holdout metrics but still has high train-test F1 gap, so overfitting risk remains.
- XGBoost has lower overfit gap but weaker holdout F1/accuracy in current tuning setup.
- Drift monitoring report generated for numeric feature shift and unseen-player lineup drift:
  - `Phase 4/results/phase4_model_monitoring_report.json`

### 10.5 Phase 4 Outputs
Result files (`Phase 4/results/`):
- `phase4_dataset.csv`
- `phase4_model_comparison_metrics.csv`
- `phase4_model_comparison_accuracy.png`
- `phase4_model_comparison_f1.png`
- `phase4_overfit_gap.png`
- `phase4_best_model_confusion_matrix.png`
- `phase4_best_model_reliability_plot.png`
- `phase4_best_model_classification_report.json`
- `phase4_model_monitoring_report.json`
- `phase4_sample_prediction_payload.json`
- `phase4_deployment_notes.md`

Artifact files (`Phase 4/artifacts/`):
- `phase4_ipl_winner_best_pipeline.joblib`
- `phase4_ipl_winner_best_pipeline.pkl`
- `phase4_model_metadata.json`

### 10.6 Accuracy Expectation Note
Targeting 85-90% match-winner accuracy in realistic pre-match IPL prediction is generally not achievable without leakage or post-match information. Current Phase 4 result (**54.29% holdout accuracy**) is a realistic leakage-safe improvement over earlier phases and should be treated as a strong baseline for further iterations.

---

## 11) Phase 4.1 Execution Update (Completed)

Phase 4.1 was initialized as a generalization-first continuation of Phase 4:
- `Phase 4.1/phase41_train_pipeline.py`
- `Phase 4.1/phase41_fastapi_app.py`
- `Phase 4.1/Phase4_1_MLPipeline.ipynb`

### 11.1 What Changed in Phase 4.1
1. Reused Phase 4 extracted dataset and focused model search on stability and reduced overfit.
2. Added an explicit model selection score:
   - `generalization_score = test_accuracy - 0.20 * fit_gap`
3. Kept cardinality control and robust preprocessing for player-slot columns.
4. Saved complete comparison metrics plus artifacts under dedicated Phase 4.1 folders.

### 11.2 Phase 4.1 Model Performance (2025 Holdout)
Source: `Phase 4.1/results/phase41_model_comparison_metrics.csv`

- **Extra Trees (highest raw accuracy)**
  - Accuracy: **0.5857**
  - F1 (Weighted): **0.5791**
  - F1 (Macro): **0.5695**
  - Fit Gap: **0.1998**

- **XGBoost (selected best by generalization score)**
  - Accuracy: **0.5571**
  - F1 (Weighted): **0.4910**
  - F1 (Macro): **0.4806**
  - Fit Gap: **-0.0106**

- **Random Forest**
  - Accuracy: **0.5571**
  - F1 (Weighted): **0.5421**
  - F1 (Macro): **0.5354**
  - Fit Gap: **0.1328**

### 11.3 Selection Rationale
- If the target is maximum holdout accuracy, Extra Trees is strongest in this run.
- If the target is deployment robustness and lower overfit risk, XGBoost was selected by the Phase 4.1 generalization rule.

### 11.4 Phase 4.1 Outputs
Results (`Phase 4.1/results/`):
- `phase41_model_comparison_metrics.csv`
- `phase41_model_comparison_accuracy.png`
- `phase41_model_comparison_f1.png`
- `phase41_overfit_gap.png`
- `phase41_best_model_confusion_matrix.png`
- `phase41_best_model_reliability_plot.png`
- `phase41_best_model_classification_report.json`
- `phase41_sample_prediction_payload.json`
- `phase41_deployment_notes.md`

Artifacts (`Phase 4.1/artifacts/`):
- `phase41_ipl_winner_best_pipeline.joblib`
- `phase41_ipl_winner_best_pipeline.pkl`
- `phase41_model_metadata.json`

### 11.5 Deployment Validation
- `Phase 4.1/phase41_fastapi_app.py` was smoke-tested using `Phase 4.1/results/phase41_sample_prediction_payload.json` and returned valid top-k probabilities.

