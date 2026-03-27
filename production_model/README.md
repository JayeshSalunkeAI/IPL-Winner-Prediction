# IPL Winner Prediction - Production Model

This repository contains the **Final Production Model** for predicting the outcome of IPL matches (Team A vs. Team B). It is structured to be used as a standalone module for automation (e.g., periodic model retraining, daily predictions).

## 📂 Directory Structure

```plaintext
production_model/
├── data/                  # Input datasets
│   ├── training_data.csv  # Historical match data (2008-21 July 2024) used for training
│   └── squads_2026.csv    # Current squad strengths for feature lookups (future use)
├── model/                 # Trained model artifacts
│   ├── model.joblib       # The serialized CatBoost model file
│   └── metadata.json      # Model configuration, accuracy metrics, and feature list
├── plots/                 # Performance visualizations
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
├── scripts/               # Operational scripts
│   ├── train_model.py         # Retrains the model using data/training_data.csv
│   ├── predict_match.py       # Manual one-off prediction CLI
│   ├── auto_predict_trigger.py# Trigger-based live pre-match prediction (CricAPI)
│   ├── run_today.sh           # One-command launcher for today's match prediction
│   ├── test_cricapi_key.py    # API connectivity and schema check
│   ├── manual_fallback_predict.py # Manual prediction flow if API automation fails
│   ├── record_match_result.py # Save actual winner after match end
│   ├── model_health_check.py  # Health report for production model
│   ├── db_report.py           # Latest predictions + running hit-rate report
│   └── ops_db.py              # SQLite DB helpers
└── requirements.txt       # Python dependencies
```

## 🛠️ Setup & Installation

To run this model in a new environment:

1.  **Clone or Copy this folder.**
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

### 0. Automation Quick Start (Step-by-Step)

Run these steps in order:

1. Open terminal in `production_model`.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Export API key:
```bash
export CRICAPI_KEY="YOUR_API_KEY"
```
4. Test API access:
```bash
python scripts/test_cricapi_key.py
```
5. Trigger automated pre-match prediction (5-10 mins before start):
```bash
./scripts/run_today.sh --team1 RCB --team2 SRH
```
6. Read output JSON in `results/`.
7. (After match end) store actual result so next predictions use latest form:
```bash
python scripts/record_match_result.py --match-id <MATCH_ID> --winner "TEAM_NAME"
```

### Daily Operations (Recommended)

Use this checklist on every match day:

1. Validate API and quota:
```bash
python scripts/test_cricapi_key.py
```
2. Start auto trigger:
```bash
./scripts/run_today.sh --team1 RCB --team2 SRH
```
3. If auto fails, run manual fallback:
```bash
python scripts/manual_fallback_predict.py --team1 "RCB" --team2 "SRH" --venue "M Chinnaswamy Stadium, Bengaluru" --toss-winner "RCB" --toss-decision field --match-id manual-rcb-srh
```
4. After match end, write actual result:
```bash
python scripts/record_match_result.py --match-id manual-rcb-srh --winner "SRH"
```
5. Review tracking report:
```bash
python scripts/db_report.py --limit 10
```
6. Check model health weekly:
```bash
python scripts/model_health_check.py
```

### 1. Running a Prediction (Inference)

Use the `predict_match.py` script to get a probability score for a match.

**Command:**
```bash
python scripts/predict_match.py \
  --team1 "RCB" \
  --team2 "CSK" \
  --venue "M Chinnaswamy Stadium" \
  --toss_winner "RCB" \
  --decision "field"
```

**Output:**
```plaintext
Predicting: RCB vs CSK at M Chinnaswamy Stadium
Loading model from .../model/model.joblib...
========================================
PREDICTION: RCB Wins!
Confidence: 68.42%
========================================
```

**Note on Features:**
The current prediction script uses **static average values** for dynamic features (e.g., `form_winrate_5`, `venue_score_prior`) as placeholders. For a fully automated pipeline, you must integrate a feature engineering step that calculates:
*   `team1_form_winrate_5`: Team 1's win rate in the last 5 matches before this date.
*   `venue_score_prior`: Average first innings score at this venue in the last 10 matches.
*   `toss_advantage`: Venue-specific win probability for the toss winner.

*(Logic for these calculations can be found in the original project's `Data_Preparation.ipynb`, which is not included here to keep this repo lightweight.)*

### 2. Retraining the Model

To update the model with new match results (e.g., after the 2026 season ends), append the new data to `data/training_data.csv` and run:

```bash
python scripts/train_model.py
```

This will:
1.  Load the dataset.
2.  Filter for recent matches (2020+).
3.  Train a robust **CatBoost Classifier**.
4.  Save the new `model.joblib` and updated plots.

### 3. Automatic Live Trigger (Recommended)

Use this when toss happens before the match and you want prediction in the final 5 to 10 minutes before start.

1. Set API key in environment:
```bash
export CRICAPI_KEY="YOUR_API_KEY"
```

2. Run trigger script:
```bash
python scripts/auto_predict_trigger.py \
  --series-search "Indian Premier League" \
  --team1 RCB \
  --team2 SRH \
  --min-before 5 \
  --max-before 10 \
  --poll-seconds 60 \
  --timeout-minutes 120
```

Fast one-command launcher:
```bash
export CRICAPI_KEY="YOUR_API_KEY"
./scripts/run_today.sh --team1 RCB --team2 SRH
```

You can still pass any advanced flags through the launcher (for example `--match-id`, `--no-squad-check`).

How it works:
1. Finds your upcoming match from CricAPI.
2. Polls match info until toss data is available.
3. Waits for the configured pre-match window (for example 5-10 minutes before start).
4. Builds model features from historical dataset + live toss/venue fields.
5. Outputs winner prediction and saves JSON output in `results/`.

For a 7:00 PM match, keep `--min-before 5 --max-before 10`; the script will emit prediction around 6:50-6:55 PM.

Optional low-quota mode (skip squad call):
```bash
python scripts/auto_predict_trigger.py --no-squad-check
```

Check key validity and endpoint health:
```bash
python scripts/test_cricapi_key.py
```

If automation fails, use manual fallback:
```bash
python scripts/manual_fallback_predict.py \
  --team1 "Mumbai Indians" \
  --team2 "Kolkata Knight Riders" \
  --venue "Wankhede Stadium, Mumbai" \
  --toss-winner "Mumbai Indians" \
  --toss-decision field \
  --match-id manual-001
```

After match completes, store actual winner (used for next-match calculations):
```bash
python scripts/record_match_result.py --match-id manual-001 --winner "Mumbai Indians" --first-innings-score 182 --second-innings-score 176
```

Check production model health:
```bash
python scripts/model_health_check.py
```

Show last predictions and running hit-rate:
```bash
python scripts/db_report.py --limit 10
```

Operational database:
1. SQLite DB path: `data/ops_matches.db`
2. Auto and manual predictions are both written to this DB.
3. Recorded match results are reused in next predictions (recent form + venue blend).

Important free-tier note:
1. CricAPI free key has daily hit limits.
2. If you see `Blocking since hits today exceeded hits limit`, wait for daily reset or move to a paid plan.

## 🧯 Troubleshooting

1. `CRICAPI_KEY is not set`:
```bash
export CRICAPI_KEY="YOUR_API_KEY"
```
2. API blocked by daily limit:
Use manual fallback for that match and resume auto mode after quota reset.
3. Wrong team name in manual mode:
Use full names used in training data (example: `Royal Challengers Bengaluru`, not `Royal Challengers Bangalore`).
4. Script command not found:
Activate environment and run from project root:
```bash
source .venv/bin/activate
cd production_model
```

## 📊 Model Performance

The current model (trained on data up to 2024) achieves:
*   **Accuracy:** ~69% (Stratified Validation)
*   **ROC AUC:** 0.75
*   **Key Drivers:** `team_form`, `venue_avg_score`, `toss_advantage`.

## 📝 Dependencies

*   `pandas`
*   `numpy`
*   `scikit-learn`
*   `catboost`
*   `joblib`
*   `matplotlib`
*   `seaborn`
