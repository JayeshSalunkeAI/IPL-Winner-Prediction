# Model Comparison

This folder keeps the live comparison workflow between the production model and phase 4.1.

## Files

- `compare_models.py`: loads a Cricbuzz-scraped match row and predicts with both models.
- `comparison_table.csv`: upsert table with the latest prediction comparison per match.
- `model_paths.json`: reference paths for the two models used in the comparison.

## Match Day Flow

Daily one-command automation:
- Refresh feature stores only:
   - `production_model/scripts/run_phase41_daily_refresh.sh`
- Refresh feature stores and predict one match id:
   - `production_model/scripts/run_phase41_daily_refresh.sh --match-id <MATCH_ID>`

Cron examples (edit with `crontab -e`):
- Refresh stores every day at 10:00 AM:
   - `0 10 * * * cd /home/vectone/MyProjects/IPL-Winner-Prediction && ./production_model/scripts/run_phase41_daily_refresh.sh >> /tmp/phase41_refresh.log 2>&1`
- Refresh + predict a known match id at 3:30 PM:
   - `30 15 * * * cd /home/vectone/MyProjects/IPL-Winner-Prediction && ./production_model/scripts/run_phase41_daily_refresh.sh --match-id 151818 >> /tmp/phase41_predict.log 2>&1`

0. Refresh Phase 4.1 scorecard-based feature stores (recommended before prediction).
   - Build/refresh links: `python build_match_ids.py --output data/raw/cricbuzz_match_links.csv --max-depth 2`
   - Build feature stores: `python production_model/scripts/update_phase41_feature_store.py --links-csv data/raw/cricbuzz_match_links.csv`
   - Outputs:
     - `production_model/data/phase41_scorecard_history.csv`
     - `production_model/data/phase41_match_feature_store.csv`
     - `production_model/data/phase41_team_feature_store.csv`

1. Scrape the match data with the Cricbuzz scraper.
   - Live match: `python current_scraper.py --match-id <MATCH_ID> --output data/raw/cricbuzz_current_matches.csv`
   - Historical backfill: `python historical_scraper.py --output data/raw/cricbuzz_historical_matches.csv`
2. Run the comparison job.
   - Example: `python "production_model/Model Comparison/compare_models.py" --input data/raw/cricbuzz_current_matches.csv`
3. Review `comparison_table.csv` in this folder.

## Notes

- The production model uses the operational feature builder and recent form from `production_model/data/ops_matches.db`.
- Phase 4.1 now consumes till-date scorecard-derived priors from `production_model/data/phase41_match_feature_store.csv` for form, H2H, venue stats, runs, wickets, powerplay RR, death RR, and player Elo aggregates.
- If a match result is already present in the scraped row, the table also stores `actual_winner`, `production_correct`, and `phase41_correct`.
