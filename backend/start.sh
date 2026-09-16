#!/bin/sh
set -e

PORT="${PORT:-8000}"

# Navigate to backend directory safely from any starting location
if [ -d "/app/backend" ]; then
  cd /app/backend
elif [ -d "backend" ]; then
  cd backend
fi

echo "[Binocrypt] Starting production server on port ${PORT} from $(pwd)..."
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
