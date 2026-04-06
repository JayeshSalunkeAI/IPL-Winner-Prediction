# Model Comparison

This folder keeps the live comparison workflow between the production model and phase 4.1.

## Files

- `compare_models.py`: loads a Cricbuzz-scraped match row and predicts with both models.
- `comparison_table.csv`: upsert table with the latest prediction comparison per match.
- `model_paths.json`: reference paths for the two models used in the comparison.

## Match Day Flow

1. Scrape the match data with the Cricbuzz scraper.
   - Live match: `python current_scraper.py --match-id <MATCH_ID> --output data/raw/cricbuzz_current_matches.csv`
   - Historical backfill: `python historical_scraper.py --output data/raw/cricbuzz_historical_matches.csv`
2. Run the comparison job.
   - Example: `python "production_model/Model Comparison/compare_models.py" --input data/raw/cricbuzz_current_matches.csv`
3. Review `comparison_table.csv` in this folder.

## Notes

- The production model uses the operational feature builder and recent form from `production_model/data/ops_matches.db`.
- Phase 4.1 uses the player-slot columns from the scraped XI plus filler values for missing engineered priors.
- If a match result is already present in the scraped row, the table also stores `actual_winner`, `production_correct`, and `phase41_correct`.
