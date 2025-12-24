#!/usr/bin/env bash
set -euo pipefail

# Helper to run the local Christmas Scenes server on port 4000
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

: "${SWARMUI:=http://swarmui:7801}"
: "${IMAGE_MODEL:=Flux/flux1-schnell-fp8}"
: "${IMAGE_CFGSCALE:=1.0}"
: "${IMAGE_STEPS:=6}"
: "${IMAGE_WIDTH:=1024}"
: "${IMAGE_HEIGHT:=1024}"
: "${PORT:=4000}"
: "${REFRESH_SECONDS:=10}"

export SWARMUI IMAGE_MODEL IMAGE_CFGSCALE IMAGE_STEPS IMAGE_WIDTH IMAGE_HEIGHT PORT REFRESH_SECONDS

echo "Starting Christmas Scenes server on port $PORT"
if [ -f chatbot/requirements.txt ]; then
  echo "Installing requirements (if needed)..."
  python3 -m pip install -r chatbot/requirements.txt || true
fi

python3 server.py
