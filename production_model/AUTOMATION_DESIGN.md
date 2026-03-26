# IPL 2026 Automation Pipeline Design

This document outlines the architecture for a fully automated prediction system for IPL 2026 using the `production_model`.

## 1. System Architecture

The system consists of three main components:
1.  **Monitor/Scheduler:** Checks for upcoming matches.
2.  **Data Fetcher:** Retrieves live match info (Toss, Playing XI).
3.  **Predictor Engine:** Calculates features dynamically and runs the model.

```mermaid
graph TD
    A[Scheduler (Cron/Task)] -->|Every 15 mins| B{Is there a match soon?}
    B -->|Yes| C{Is Toss Done?}
    C -->|Yes| D[Fetch Live Data: Toss, XI, Pitch]
    D --> E[Feature Engine]
    E -->|Combine 2026 History + Live Data| F[Construct Feature Vector]
    F --> G[Run Model Prediction]
    G --> H[Output: Notification/Dashboard]
    B -->|No| I[Sleep]
```

## 2. Implementation Strategy

### Step 1: Data Source Integration (The Hard Part)
You need a reliable source for live match data.
*   **Paid APIs:** SportMonks, CricAPI (Best reliability).
*   **Free/Scraping:** Using unofficial libraries (risky but free).
*   **Manual Override:** A simple GUI/CLI to input Toss details manually if API fails.

### Step 2: The `2026_season.csv` Database
You must maintain a CSV file that records *every completed match* in 2026.
*   **Why?** The model uses "Last 5 Matches Form".
*   **How?** After every match finishes, append the result to `data/2026_season_results.csv`.
*   **The `FeatureEngine`:** I have included a `scripts/feature_engine.py` skeleton that shows how to load history + 2026 results to calculate current form dynamically.

### Step 3: Automation Script (`run_pipeline.py`)
This script would look like this (pseudo-code):

```python
import schedule
import time
from scripts.feature_engine import FeatureEngine
from scripts.predict_match import predict

def check_match():
    match = get_today_match() # From API or Schedule CSV
    if match and is_toss_time(match):
        toss_info = get_toss_result() # From API
        
        # KEY STEP: Calculate dynamic features
        features = FeatureEngine().construct_features(
            team1=match.team1,
            team2=match.team2,
            venue=match.venue,
            toss_winner=toss_info.winner,
            toss_decision=toss_info.decision
        )
        
        # Predict
        result = model.predict(features)
        send_telegram_notification(result)

schedule.every(10).minutes.do(check_match)

while True:
    schedule.run_pending()
    time.sleep(1)
```

## 3. Deployment
1.  **Server:** A small VPS (AWS EC2 free tier, DigitalOcean) or a Raspberry Pi.
2.  **Environment:** Set up `requirements.txt`.
3.  **Process Manager:** Use `pm2` or `systemd` to keep the script running 24/7.

## 4. Manual Fallback
Always keep `scripts/predict_match.py` handy. If the automation fails (API down), run the command manually:
```bash
python scripts/predict_match.py --team1 "RCB" --team2 "MI" ...
```
