#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WREN_UI_DIR="$PROJECT_ROOT/WrenAI/wren-ui"
NODE_RUNTIME_BIN="$PROJECT_ROOT/.local/node-runtime/node_modules/node/bin"

if [[ ! -x "$NODE_RUNTIME_BIN/node" || ! -f "$WREN_UI_DIR/node_modules/next/dist/bin/next" ]]; then
  echo "Frontend dependencies are missing. See README.md: Recovered WrenAI frontend." >&2
  exit 1
fi

export PATH="$NODE_RUNTIME_BIN:$PATH"
export TZ=UTC
export NEXT_TELEMETRY_DISABLED=1
export TELEMETRY_ENABLED=false
export DB_TYPE=sqlite
export SQLITE_FILE="${SQLITE_FILE:-$WREN_UI_DIR/db.sqlite3}"
export WREN_DATA_ONLY_MODE="${WREN_DATA_ONLY_MODE:-true}"
export WREN_DUCKDB_FILE="${WREN_DUCKDB_FILE:-$PROJECT_ROOT/.local/wren-data/wren.duckdb}"
export WREN_SAMPLE_DATA_DIR="${WREN_SAMPLE_DATA_DIR:-$PROJECT_ROOT/.local/wren-data/ecommerce}"
export AUTOMIND_API_URL="${AUTOMIND_API_URL:-http://127.0.0.1:8000/predict/ecommerce-good-review}"
export AUTOMIND_HEART_DISEASE_API_URL="${AUTOMIND_HEART_DISEASE_API_URL:-http://127.0.0.1:8000/predict/heart-disease}"

if [[ "$WREN_DATA_ONLY_MODE" == true ]]; then
  "$PROJECT_ROOT/venv/bin/python" "$PROJECT_ROOT/scripts/download_wren_sample.py"
fi

cd "$WREN_UI_DIR"
node node_modules/knex/bin/cli.js migrate:latest
if [[ "${WREN_UI_MODE:-dev}" == production ]]; then
  if [[ ! -f .next/BUILD_ID ]]; then
    echo "Production build is missing. Build WrenAI/wren-ui first, or use the default dev mode." >&2
    exit 1
  fi
  # Next's standalone server needs its public and compiled static assets beside it.
  mkdir -p .next/standalone/.next
  cp -R .next/static .next/standalone/.next/
  cp -R public .next/standalone/
  export PORT="${WREN_UI_PORT:-3001}"
  export HOSTNAME=127.0.0.1
  exec node .next/standalone/server.js
fi
exec node node_modules/next/dist/bin/next dev --hostname 127.0.0.1 --port "${WREN_UI_PORT:-3001}"
