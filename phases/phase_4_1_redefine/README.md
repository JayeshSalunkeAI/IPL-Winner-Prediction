# Phase 4.1 Redefine

This folder rebuilds Phase 4.1 with improved player performance features scraped
from the IPL stats site for seasons 2020 to 2026.

## Pipeline

1. Scrape season batting and bowling stats from `https://www.iplt20.com/stats/<year>` feeds.
2. Clean and combine player performance into an Elo-like rating table.
3. Train an ExtraTrees generalization model using the Phase 4.1 feature schema.

## Commands

Run from repo root.

```bash
python phases/phase_4_1_redefine/scrape_ipl_player_stats.py \
  --years 2020 2021 2023 2024 2025 2026 \
  --out-dir phases/phase_4_1_redefine/data

python phases/phase_4_1_redefine/build_player_ratings.py \
  --batting-csv phases/phase_4_1_redefine/data/ipl_toprunsscorers_2020_2026.csv \
  --bowling-csv phases/phase_4_1_redefine/data/ipl_mostwickets_2020_2026.csv \
  --out-csv phases/phase_4_1_redefine/data/player_performance_ratings_2020_2026.csv

python phases/phase_4_1_redefine/train_phase41r_extra_trees.py \
  --input phases/phase_4/results/phase4_dataset.csv \
  --player-ratings phases/phase_4_1_redefine/data/player_performance_ratings_2020_2026.csv \
  --results-dir phases/phase_4_1_redefine/results \
  --artifacts-dir phases/phase_4_1_redefine/artifacts \
  --years 2020 2021 2023 2024 2025 \
  --drop-no-result
```

## Outputs

- `data/ipl_toprunsscorers_2020_2026.csv`
- `data/ipl_mostwickets_2020_2026.csv`
- `data/player_performance_ratings_2020_2026.csv`
- `artifacts/phase41r_extratrees_pipeline.joblib`
- `artifacts/phase41r_model_metadata.json`
- `results/phase41r_metrics.json`
