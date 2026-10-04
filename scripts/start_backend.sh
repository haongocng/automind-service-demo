#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -x venv/bin/python ]]; then
  echo "Missing venv. Run: python3 -m venv venv && venv/bin/python -m pip install -r requirements.txt" >&2
  exit 1
fi

export LLM_INSIGHT_ENABLED="${LLM_INSIGHT_ENABLED:-false}"
exec venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port "${AUTOMIND_PORT:-8000}"
