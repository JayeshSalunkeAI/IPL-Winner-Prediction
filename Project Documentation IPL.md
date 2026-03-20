# IPL Winner Prediction — Project Documentation

## 1) Project Overview
This project prepares a machine-learning-ready IPL match dataset by cleaning and transforming historical match data from 2008 to 2024.

Primary goal:
- Build a consistent feature dataset for predicting the match winner.

Current status:
- Data extraction and feature engineering pipeline is implemented in `Data_Extraction.ipynb`.
- Final processed dataset is generated and saved as `IPL_Winner_Model_Dataset.csv`.

---

## 2) Current Workspace Structure
- `Data_Extraction.ipynb` — main notebook containing full data preprocessing + feature engineering pipeline.
- `IPL_Dataset(2008-2024).csv` — base/source tabular match dataset.
- `IPL_Winner_Model_Dataset.csv` — final engineered dataset used for modeling.
- `ipl_male/` — raw IPL YAML match files (1169 files).
- `README.md` — minimal project readme currently containing title.
- `Project Documentation IPL.docx` — existing document version.
- `Project Documentation IPL.md` — this clean, up-to-date documentation.

---

## 3) Work Completed So Far

### 3.1 Data Loading and Initial Exploration
Done in notebook:
- Imported `pandas`, `numpy`, and `requests`.
- Loaded source CSV (`IPL_Dataset(2008-2024).csv`) into DataFrame `df`.
- Performed basic exploration:
  - `head()`
  - `describe()`
  - `shape`
  - `info()`

### 3.2 Time Ordering + Team Parsing
- Converted `Date` to datetime.
- Sorted records by `Date` to preserve chronological order.
- Split `Teams` column into:
  - `Team1`
  - `Team2`

### 3.3 Recent Form Feature Engineering
Implemented rolling team form feature:
- Built long-format match table (two rows per match: each team perspective).
- Computed `win` indicator.
- Used rolling window of previous **N = 5** matches with shift(1) to avoid leakage.
- Created:
  - `team1_form_winrate_5`
  - `team2_form_winrate_5`

### 3.4 Venue-Based Prior Features
Implemented venue chasing tendency features:
- Derived batting order using toss winner + toss decision (`bat` / `field`).
- Maintained incremental venue stats with Laplace smoothing (`alpha = 1`).
- Added prior-only features per match:
  - `venue_chase_winrate_prior`
  - `venue_score_prior` where `score = 2 * chase_rate - 1`

### 3.5 Team Name Standardization
Standardized historical team names:
- `Delhi Daredevils` → `Delhi Capitals`
- `Kings XI Punjab` → `Punjab Kings`
- `Royal Challengers Bangalore` → `Royal Challengers Bengaluru`

Applied standardization to:
- `Teams`
- `Toss_Winner`
- `Match_Winner`
- Then re-generated `Team1` and `Team2` from updated `Teams`.

### 3.6 Defunct Team Filtering
Removed matches involving defunct teams:
- Deccan Chargers
- Gujarat Lions
- Kochi Tuskers Kerala
- Pune Warriors
- Rising Pune Supergiants

### 3.7 Final Modeling Dataset Creation
Selected and ordered final columns into `new_df`:
1. `Match_ID`
2. `Date`
3. `Teams`
4. `Team1`
5. `Team2`
6. `Toss_Winner`
7. `Toss_Decision`
8. `team1_form_winrate_5`
9. `team2_form_winrate_5`
10. `venue_chase_winrate_prior`
11. `venue_score_prior`
12. `Match_Winner`

### 3.8 Local Output Save
- Saved `new_df` to local project folder as:
  - `IPL_Winner_Model_Dataset.csv`

---

## 4) Data Snapshot (Current)

Source dataset:
- File: `IPL_Dataset(2008-2024).csv`
- Rows: **1073**

Final engineered dataset:
- File: `IPL_Winner_Model_Dataset.csv`
- Rows: **902**
- Columns: **12**

Intermediate engineered DataFrame (`df2`) before final column selection:
- Rows: **902**
- Columns: **26**

Raw YAML archive:
- Folder: `ipl_male/`
- YAML files: **1169**

---

## 5) Features Ready for Prediction
Core predictive features currently prepared:
- Team identity (`Team1`, `Team2`)
- Toss context (`Toss_Winner`, `Toss_Decision`)
- Recent form (`team1_form_winrate_5`, `team2_form_winrate_5`)
- Venue tendency priors (`venue_chase_winrate_prior`, `venue_score_prior`)

Target label:
- `Match_Winner`

---

## 6) Notes and Known Considerations
- `team1_form_winrate_5` has initial missing values for very early team history (expected due to limited prior matches).
- Venue prior features are generated chronologically and use only prior information, reducing leakage risk.
- Team-name harmonization improves consistency across seasons.

---

## 7) Suggested Next Steps
1. Add train/validation/test split strategy based on time (chronological split).
2. Encode categorical features (`Team1`, `Team2`, toss fields) safely.
3. Train baseline models (Logistic Regression, Random Forest, XGBoost/LightGBM).
4. Evaluate using accuracy + log loss + confusion matrix.
5. Track feature importance and calibration.
6. Add a dedicated training notebook/script and update README with usage steps.

---

## 8) How to Reproduce Current Output
In `Data_Extraction.ipynb`, run all cells in order and ensure final save cell executes:

```python
output_path = "IPL_Winner_Model_Dataset.csv"
new_df.to_csv(output_path, index=False)
print(f"Saved: {output_path}")
```

This regenerates the final dataset in the project root.
