#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Prefer local venv python if available; fallback to python3.
if [[ -x "$BASE_DIR/../.venv/bin/python" ]]; then
  PYTHON_BIN="$BASE_DIR/../.venv/bin/python"
elif [[ -x "$BASE_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$BASE_DIR/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

if [[ -z "${CRICAPI_KEY:-}" ]]; then
  echo "ERROR: CRICAPI_KEY is not set."
  echo "Set it first: export CRICAPI_KEY=\"YOUR_KEY\""
  exit 2
fi

# Defaults can be overridden by env vars or CLI args passed through "$@".
SERIES_SEARCH="${SERIES_SEARCH:-Indian Premier League}"
MIN_BEFORE="${MIN_BEFORE:-5}"
MAX_BEFORE="${MAX_BEFORE:-10}"
POLL_SECONDS="${POLL_SECONDS:-60}"
TIMEOUT_MINUTES="${TIMEOUT_MINUTES:-120}"

exec "$PYTHON_BIN" "$BASE_DIR/scripts/auto_predict_trigger.py" \
  --series-search "$SERIES_SEARCH" \
  --min-before "$MIN_BEFORE" \
  --max-before "$MAX_BEFORE" \
  --poll-seconds "$POLL_SECONDS" \
  --timeout-minutes "$TIMEOUT_MINUTES" \
  "$@"
