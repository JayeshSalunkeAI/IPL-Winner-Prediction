# Phase 2 Deployment Notes

## Train

```bash
python phases/phase_2/phase2_train_pipeline.py
```

## Artifacts

- `phases/phase_2/artifacts/phase2_ipl_winner_best_pipeline.joblib`
- `phases/phase_2/artifacts/phase2_ipl_winner_best_pipeline.pkl`
- `phases/phase_2/artifacts/phase2_model_metadata.json`

## Sample payload

```json
{
  "features": {
    "Team1": "Kolkata Knight Riders",
    "Team2": "Royal Challengers Bengaluru",
    "Toss_Winner": "Royal Challengers Bengaluru",
    "Toss_Decision": "field",
    "team1_form_winrate_5": 0.8,
    "team2_form_winrate_5": 0.4,
    "venue_chase_winrate_prior": 0.5,
    "venue_score_prior": 0.0,
    "Team1_Players": "PlayerA|PlayerB|...",
    "Team2_Players": "PlayerX|PlayerY|...",
    "team1_player_elo_avg_prior": 1012.4,
    "team2_player_elo_avg_prior": 1008.1,
    "team1_player_elo_max_prior": 1060.0,
    "team2_player_elo_max_prior": 1042.5,
    "team1_player_elo_min_prior": 970.0,
    "team2_player_elo_min_prior": 968.0,
    "player_elo_gap_prior": 4.3,
    "Team1_Player_1": "PlayerA",
    "Team1_Player_2": "PlayerB",
    "Team1_Player_3": "PlayerC",
    "Team1_Player_4": "PlayerD",
    "Team1_Player_5": "PlayerE",
    "Team1_Player_6": "PlayerF",
    "Team1_Player_7": "PlayerG",
    "Team1_Player_8": "PlayerH",
    "Team1_Player_9": "PlayerI",
    "Team1_Player_10": "PlayerJ",
    "Team1_Player_11": "PlayerK",
    "Team2_Player_1": "PlayerX",
    "Team2_Player_2": "PlayerY",
    "Team2_Player_3": "PlayerZ",
    "Team2_Player_4": "PlayerM",
    "Team2_Player_5": "PlayerN",
    "Team2_Player_6": "PlayerO",
    "Team2_Player_7": "PlayerP",
    "Team2_Player_8": "PlayerQ",
    "Team2_Player_9": "PlayerR",
    "Team2_Player_10": "PlayerS",
    "Team2_Player_11": "PlayerT"
  }
}
```

Use `phases/phase_2/results/sample_prediction_payload.json` as the base payload.
