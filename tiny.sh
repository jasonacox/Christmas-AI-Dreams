#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="jasonacox/christmas-ai-dreams:latest"
CONTAINER="christmas-ai-dreams-tiny"

# Remove any existing container with the same name
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

# Pull latest image if available
docker pull "$IMAGE_NAME" || true

# Run container with hard-coded envs
docker run -d --name "christmas-ai-dreams-tiny" --restart unless-stopped --network host \
  -e IMAGE_PROVIDER=swarmui \
  -e SWARMUI=http://localhost:7801 \
  -e IMAGE_MODEL="Flux/flux1-schnell-fp8" \
  -e IMAGE_CFGSCALE=1.0 \
  -e IMAGE_STEPS=6 \
  -e IMAGE_WIDTH=1024 \
  -e IMAGE_HEIGHT=1024 \
  -e REFRESH_SECONDS=60 \
  "jasonacox/christmas-ai-dreams:latest"
  

echo "Started $CONTAINER (listening on host:4000)"

docker logs --tail 50 -f "$CONTAINER"
