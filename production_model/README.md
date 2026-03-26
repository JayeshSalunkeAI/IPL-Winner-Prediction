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
│   ├── train_model.py     # Retrains the model using data/training_data.csv
│   └── predict_match.py   # CLI tool to run a prediction
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
