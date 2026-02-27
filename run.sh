#!/usr/bin/env bash
set -euo pipefail

# activate venv if not already active
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    else
        echo "No virtual environment found. Run ./setup.sh first."
        exit 1
    fi
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "Starting Graph ML Lab on http://${HOST}:${PORT}"
echo "Press Ctrl+C to stop."
echo ""

python -m uvicorn backend.app:app --host "$HOST" --port "$PORT" --reload
