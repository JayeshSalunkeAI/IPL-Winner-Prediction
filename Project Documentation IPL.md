# IPL Winner Prediction - Project Documentation

## 1) Overview

This repository predicts IPL match winners from pre-match data.
The project evolved through phased improvements in features, regularization, model selection, and deployment readiness.

This document reflects the phase-based structure and latest validated runs.

---

## 2) Repository Structure

```text
data/
  raw/
    IPL_2025_Winner_Model_Dataset.csv
    IPL_Dataset(2008-2024).csv
    IPL_Player_ELO_Ratings.csv
    IPL_Winner_Model_Dataset.csv
    IPL_Winner_Model_Dataset_With_Players.csv
    ipl_male/*.yaml

phases/
  phase_1/
    app/
      streamlit_app.py
    scripts/
      extract_2025_match_data.py
      Player_Data_extraction.py
    notebooks/
      Data_Exploration.ipynb
      Data_Preparation.ipynb
      Pipeline_ML.ipynb
    artifacts/
      ipl_winner_xgb_pipeline.joblib
      ipl_winner_xgb_pipeline.pkl
      ipl_winner_model_metadata.json

  phase_2/
    phase2_train_pipeline.py
    Phase2_MLPipeline.ipynb
    results/
    artifacts/

  phase_3/
    phase3_train_pipeline.py
    phase3_transforms.py
    Phase3_MLPipeline.ipynb
    results/
    artifacts/

  phase_4/
    phase4_extract_dataset.py
    phase4_train_pipeline.py
    phase4_transforms.py
    Phase4_MLPipeline.ipynb
    results/
    artifacts/

  phase_4_1/
    phase41_train_pipeline.py
    phase41_transforms.py
    phase41_fastapi_app.py
    Phase4_1_MLPipeline.ipynb
    results/
    artifacts/

  phase_5/
    phase5_train_pipeline.py
    phase5_transforms.py
    Phase5_MLPipeline.ipynb
    results/
    artifacts/

  phase_5_1/
    phase51_extract_dataset.py
    phase51_train_pipeline.py
    phase51_transforms.py
    Phase5_1_MLPipeline.ipynb
    results/
    artifacts/

live_system/
  ipl_live_predictor/
    data/
      dummy_matches.json
    src/
      config.py
      data_provider.py
      model_runtime.py
      state_engine.py
      storage.py
      engine.py
      main.py
    state/
      live_state.db
    README.md
```

Cleanup actions completed:
- Removed root-level clutter and grouped files by phase.
- Moved datasets and YAML corpus into `data/raw/`.
- Removed unnecessary legacy FastAPI apps from earlier phases.
- Kept only the final deployment API in `phases/phase_4_1/`.
- Added separate live automation system in `live_system/ipl_live_predictor/`.

---

## 3) Phase-by-Phase Evolution

## Phase 1

Goal:
- Build a baseline pre-match winner model and deploy a simple UI.

What was added:
- Core features: teams, toss winner/decision, short-form trend, venue priors.
- Streamlit app for baseline prediction.

Current location:
- App: `phases/phase_1/app/streamlit_app.py`
- Scripts: `phases/phase_1/scripts/`
- Artifacts: `phases/phase_1/artifacts/`

## Phase 2

Goal:
- Add player-aware features and compare multiple models.

What was improved:
- Player slot columns (`Team1_Player_1..11`, `Team2_Player_1..11`).
- Player ELO prior aggregates.
- Multi-model comparison (RF, XGB, Logistic).
- Overfit diagnostics and saved artifacts/reports.

Current location:
- Training script: `phases/phase_2/phase2_train_pipeline.py`
- Notebook: `phases/phase_2/Phase2_MLPipeline.ipynb`
- Results: `phases/phase_2/results/`
- Artifacts: `phases/phase_2/artifacts/`

## Phase 3

Goal:
- Improve generalization and reduce overfitting from Phase 2.

What was improved:
- Stronger regularization.
- Categorical cardinality control for player columns.
- Calibration and top-k evaluation.
- Drift monitoring outputs.

Current location:
- Training script: `phases/phase_3/phase3_train_pipeline.py`
- Transformer module: `phases/phase_3/phase3_transforms.py`
- Notebook: `phases/phase_3/Phase3_MLPipeline.ipynb`
- Results: `phases/phase_3/results/`
- Artifacts: `phases/phase_3/artifacts/`

## Phase 4

Goal:
- Enrich priors using ball-by-ball data and expand tuned model search.

What was improved:
- Added advanced priors (head-to-head, venue-team, first-innings venue average, team recent scoring/bowling rates).
- Retained player-slot control pipeline.
- Expanded search across XGBoost, Extra Trees, Random Forest, Logistic Regression.

Current location:
- Dataset extraction: `phases/phase_4/phase4_extract_dataset.py`
- Training script: `phases/phase_4/phase4_train_pipeline.py`
- Notebook: `phases/phase_4/Phase4_MLPipeline.ipynb`
- Results: `phases/phase_4/results/`
- Artifacts: `phases/phase_4/artifacts/`

## Phase 4.1

Goal:
- Prioritize deployment stability and lower overfit risk.

What was improved:
- Stronger generalization-first tuning.
- Explicit selection objective:
  - `generalization_score = test_accuracy - 0.20 * max(fit_gap, 0)`
- Final deployable API retained for this phase only.

Current location:
- Training script: `phases/phase_4_1/phase41_train_pipeline.py`
- API: `phases/phase_4_1/phase41_fastapi_app.py`
- Notebook: `phases/phase_4_1/Phase4_1_MLPipeline.ipynb`
- Results: `phases/phase_4_1/results/`
- Artifacts: `phases/phase_4_1/artifacts/`

## Phase 5

Goal:
- Improve raw predictive performance with stronger ensemble candidates.

What was improved:
- Added LightGBM, CatBoost, and stacking variants.
- Added faster search strategy and overfit-aware model selection support.
- Preserved reports and artifacts in the same style as previous phases.

Current location:
- Training script: `phases/phase_5/phase5_train_pipeline.py`
- Notebook: `phases/phase_5/Phase5_MLPipeline.ipynb`
- Results: `phases/phase_5/results/`
- Artifacts: `phases/phase_5/artifacts/`

## Phase 5.1

Goal:
- Improve 2025 generalization with recency-aware signals and season-weighted training.

What was improved:
- Added season-weighted training (higher weights for 2022 to 2024).
- Added new extracted priors:
  - Recent XI continuity.
  - Player availability/injury proxy.
  - Toss-decision x venue interaction priors.
  - Venue phase splits (powerplay/death, first/second innings).
- Added seeded stacking ensemble (probability averaging over multiple seeds).
- Added two-stage optimization:
  - Fast search on stacking, CatBoost, XGBoost.
  - Full search on top-2 candidates from fast stage.

Current location:
- Dataset extraction: `phases/phase_5_1/phase51_extract_dataset.py`
- Training script: `phases/phase_5_1/phase51_train_pipeline.py`
- Notebook: `phases/phase_5_1/Phase5_1_MLPipeline.ipynb`
- Results: `phases/phase_5_1/results/`
- Artifacts: `phases/phase_5_1/artifacts/`

## Live Predictor System

Goal:
- Run automated match-by-match prediction and update priors through the season.

What it does:
- Ingests feed payload (dummy feed now; API field left empty).
- Predicts pre-match winner with confidence and stores history.
- Finalizes completed matches and updates rolling state for next matches.
- Persists state in SQLite for fast retrieval.

State updates covered:
- Team form and rolling scoring/bowling rates.
- Player ELO.
- Venue priors and venue-team priors.
- Head-to-head priors.
- Toss-decision x venue priors.
- Lineup continuity history.

Current location:
- `live_system/ipl_live_predictor/`

---

## 4) Latest Metrics (2025 Holdout)

Phase 4.1:
- Best raw accuracy (legacy): around `0.5857`

Phase 5:
- Best model: `stacking_gen`
- Test accuracy: `0.5857`
- Weighted F1: `0.5852`
- Fit gap (weighted F1): `0.1850`

Phase 5.1:
- Best model run: `catboost_gen_full`
- Test accuracy: `0.5286`
- Weighted F1: `0.5062`
- Fit gap (weighted F1): `0.1645`

Current practical deployment choice:
- Keep Phase 5 best model for predictions until Phase 5.1 surpasses it.

---

## 5) Validation and Execution Status

Validated in the cleaned structure:
- Phase notebooks run with updated local paths.
- Phase 4.1 API smoke-tested with sample payload.
- Scripts updated to use structure-safe default paths rooted at project root.
- Live predictor dummy-flow tested end-to-end:
  - Predictions written with confidence.
  - Completed match finalized.
  - SQLite state tables updated (ELO, team history, venue priors, h2h).

Final API for deployment:
- `phases/phase_4_1/phase41_fastapi_app.py`

Live automation entrypoint:
- `live_system/ipl_live_predictor/src/main.py`

---


