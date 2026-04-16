#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

MATCH_ID=""
MAX_DEPTH="2"
SERIES_KEY="Indian Premier League 2026"
START_ID="149618"
SLEEP_SECS="0.1"
FORCE_REFRESH="0"

usage() {
  cat <<'EOF'
Usage:
  run_phase41_daily_refresh.sh [options]

Options:
  --match-id <id>          Optional. If provided, scrape and predict this match after refresh.
  --max-depth <n>          Match link crawl depth. Default: 2
  --series-key <text>      Target series filter for feature store. Default: Indian Premier League 2026
  --start-id <id>          Minimum match id to process. Default: 149618
  --sleep <seconds>        Sleep between scorecard requests in feature builder. Default: 0.1
  --force-refresh          Force re-scrape all links even if already in history store.
  -h, --help               Show this help message.

Examples:
  run_phase41_daily_refresh.sh
  run_phase41_daily_refresh.sh --match-id 151818
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --match-id)
      MATCH_ID="${2:-}"
      shift 2
      ;;
    --max-depth)
      MAX_DEPTH="${2:-2}"
      shift 2
      ;;
    --series-key)
      SERIES_KEY="${2:-Indian Premier League 2026}"
      shift 2
      ;;
    --start-id)
      START_ID="${2:-149618}"
      shift 2
      ;;
    --sleep)
      SLEEP_SECS="${2:-0.1}"
      shift 2
      ;;
    --force-refresh)
      FORCE_REFRESH="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 2
      ;;
  esac
done

echo "[1/3] Refreshing Cricbuzz match links..."
"$PYTHON_BIN" "$ROOT_DIR/build_match_ids.py" \
  --output "$ROOT_DIR/data/raw/cricbuzz_match_links.csv" \
  --max-depth "$MAX_DEPTH"

echo "[2/3] Updating Phase 4.1 feature stores..."
UPDATE_ARGS=(
  "$ROOT_DIR/production_model/scripts/update_phase41_feature_store.py"
  --links-csv "data/raw/cricbuzz_match_links.csv"
  --start-id "$START_ID"
  --series-key "$SERIES_KEY"
  --sleep "$SLEEP_SECS"
)
if [[ "$FORCE_REFRESH" == "1" ]]; then
  UPDATE_ARGS+=(--force-refresh)
fi
"$PYTHON_BIN" "${UPDATE_ARGS[@]}"

if [[ -n "$MATCH_ID" ]]; then
  echo "[3/3] Scraping and predicting match $MATCH_ID..."
  SCRAPE_OUT="$ROOT_DIR/data/raw/cricbuzz_today_${MATCH_ID}.csv"
  "$PYTHON_BIN" "$ROOT_DIR/current_scraper.py" \
    --match-id "$MATCH_ID" \
    --scorecard-url "https://www.cricbuzz.com/live-cricket-scorecard/${MATCH_ID}/match" \
    --squads-url "https://www.cricbuzz.com/cricket-match-squads/${MATCH_ID}/match" \
    --output "$SCRAPE_OUT"

  "$PYTHON_BIN" "$ROOT_DIR/production_model/Model Comparison/compare_models.py" \
    --input "$SCRAPE_OUT" \
    --output "$ROOT_DIR/production_model/Model Comparison/comparison_table.csv"
else
  echo "[3/3] Prediction skipped (no --match-id provided)."
fi

echo "Done."
