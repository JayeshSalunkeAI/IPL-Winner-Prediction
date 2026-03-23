# IPL Live Predictor Automation

This folder contains a standalone automation system to serve predictions from the best trained model and keep match-state priors updated as the season progresses.

## What it does

- Loads best model artifact (currently Phase 5 model by default).
- Reads a feed payload (dummy JSON for now; API URL placeholder available in config).
- Predicts pre-match winner + confidence score.
- Stores prediction and probability distribution into SQLite for fast retrieval.
- On completed matches, updates rolling team form, player ELO, h2h, venue priors, toss-decision priors, and lineup continuity state.
- Uses updated state for next match predictions.

## Folder structure

- src/config.py: runtime paths and API placeholder config.
- src/data_provider.py: feed loader (dummy provider).
- src/model_runtime.py: model artifact loader + prediction call.
- src/state_engine.py: pre-match feature assembly + post-match state updates.
- src/storage.py: SQLite schema and persistence helpers.
- src/engine.py: orchestration logic.
- src/main.py: CLI runner.
- data/dummy_matches.json: dummy feed payload shape.
- state/live_state.db: generated SQLite database.

## API placeholder

In `src/config.py`, API field is intentionally empty by default:

- `IPL_API_BASE_URL` environment variable

You can wire actual API later without changing core prediction/state logic.

## Run

From repository root:

```bash
source .venv/bin/activate
python live_system/ipl_live_predictor/src/main.py --mode once
```

Loop mode:

```bash
python live_system/ipl_live_predictor/src/main.py --mode loop --interval-seconds 120
```

## Query history quickly

```bash
sqlite3 live_system/ipl_live_predictor/state/live_state.db "SELECT match_id, team1, team2, predicted_winner, confidence, actual_winner, status FROM matches ORDER BY match_date;"
```

## Notes

- Dummy feed already includes one upcoming and one completed example.
- Feature calculations are aligned with your training priors (team form, venue chase prior, h2h, player ELO, toss/venue interactions, lineup continuity).
