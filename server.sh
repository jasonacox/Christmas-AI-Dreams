#!/usr/bin/env bash
set -euo pipefail

# Reliable Docker runner for the Christmas AI Dreams service
# - stops and removes any existing container with the configured name
# - deletes the existing image for the chosen tag (if requested)
# - pulls the image and (re)starts the container with --restart=unless-stopped

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

IMAGE_NAME="jasonacox/christmas-ai-dreams"
TAG="${1:-latest}"
CONTAINER_NAME="${CONTAINER_NAME:-christmas-ai-dreams}"

# Set defaults (same pattern as run_local_server.sh) and export them so
# docker run -e VAR will inherit the values from this script's environment.
: "${IMAGE_PROVIDER:=swarmui}"
: "${SWARMUI:=http://host.docker.internal:7801}"
: "${IMAGE_MODEL:=Flux/flux1-schnell-fp8}"
: "${IMAGE_CFGSCALE:=1.0}"
: "${IMAGE_STEPS:=6}"
: "${IMAGE_WIDTH:=1024}"
: "${IMAGE_HEIGHT:=1024}"
: "${IMAGE_SEED:=-1}"
: "${IMAGE_TIMEOUT:=300}"
: "${REFRESH_SECONDS:=60}"
: "${OPENAI_IMAGE_API_KEY:=}"
: "${OPENAI_IMAGE_API_BASE:=https://api.openai.com/v1}"
: "${OPENAI_IMAGE_MODEL:=dall-e-3}"
: "${OPENAI_IMAGE_SIZE:=1024x1024}"
: "${PORT:=4000}"

export IMAGE_PROVIDER SWARMUI IMAGE_MODEL IMAGE_CFGSCALE IMAGE_STEPS IMAGE_WIDTH IMAGE_HEIGHT IMAGE_SEED IMAGE_TIMEOUT REFRESH_SECONDS OPENAI_IMAGE_API_KEY OPENAI_IMAGE_API_BASE OPENAI_IMAGE_MODEL OPENAI_IMAGE_SIZE PORT

# (We pass explicit environment variable NAMES to docker so values come from exported env)

echo "Managing Docker service: ${IMAGE_NAME}:${TAG} -> container ${CONTAINER_NAME}"

# Stop and remove existing container if present
if docker ps -a --format '{{.Names}}' | grep -xq "$CONTAINER_NAME"; then
  echo "Stopping and removing existing container: $CONTAINER_NAME"
  docker rm -f "$CONTAINER_NAME" || true
fi

# Pull the image (if available on registry)
echo "Pulling image ${IMAGE_NAME}:${TAG} (if available)"
docker pull ${IMAGE_NAME}:${TAG} || true

# Detect whether host.docker.internal is available; on many Linux hosts it is not
# and containers should use host networking to reach services running on the host.
USE_HOST_NETWORK=0
if [ "$(uname -s)" = "Linux" ]; then
  if ! getent hosts host.docker.internal >/dev/null 2>&1; then
    echo "host.docker.internal not available — using host networking so container can reach host services"
    USE_HOST_NETWORK=1
  fi
fi

# Decide SWARMUI URL to pass into the container depending on networking
if [ "$USE_HOST_NETWORK" -eq 1 ]; then
  SWARMUI_FOR_RUN="http://127.0.0.1:7801"
else
  SWARMUI_FOR_RUN="${SWARMUI:-http://host.docker.internal:7801}"
fi

# Build docker run command with explicit env vars and defaults
if [ "$USE_HOST_NETWORK" -eq 1 ]; then
  DOCKER_RUN=(docker run -d --name "$CONTAINER_NAME" --restart unless-stopped --network host)
else
  DOCKER_RUN=(docker run -d --name "$CONTAINER_NAME" --restart unless-stopped -p ${PORT}:4000)
fi

# Basic provider selection and SwarmUI defaults
DOCKER_RUN+=( -e "IMAGE_PROVIDER=${IMAGE_PROVIDER:-swarmui}" )
DOCKER_RUN+=( -e "SWARMUI=${SWARMUI_FOR_RUN}" )

# Image generation model & parameters
DOCKER_RUN+=( -e "IMAGE_MODEL=${IMAGE_MODEL:-Flux/flux1-schnell-fp8}" )
DOCKER_RUN+=( -e "IMAGE_CFGSCALE=${IMAGE_CFGSCALE:-1.0}" )
DOCKER_RUN+=( -e "IMAGE_STEPS=${IMAGE_STEPS:-6}" )
DOCKER_RUN+=( -e "IMAGE_WIDTH=${IMAGE_WIDTH:-1024}" )
DOCKER_RUN+=( -e "IMAGE_HEIGHT=${IMAGE_HEIGHT:-1024}" )
DOCKER_RUN+=( -e "IMAGE_SEED=${IMAGE_SEED:--1}" )
DOCKER_RUN+=( -e "IMAGE_TIMEOUT=${IMAGE_TIMEOUT:-300}" )

# Refresh / UI
DOCKER_RUN+=( -e "REFRESH_SECONDS=${REFRESH_SECONDS:-60}" )

# OpenAI-compatible settings (if using openai provider)
DOCKER_RUN+=( -e "OPENAI_IMAGE_API_KEY=${OPENAI_IMAGE_API_KEY:-}" )
DOCKER_RUN+=( -e "OPENAI_IMAGE_API_BASE=${OPENAI_IMAGE_API_BASE:-https://api.openai.com/v1}" )
DOCKER_RUN+=( -e "OPENAI_IMAGE_MODEL=${OPENAI_IMAGE_MODEL:-dall-e-3}" )
DOCKER_RUN+=( -e "OPENAI_IMAGE_SIZE=${OPENAI_IMAGE_SIZE:-1024x1024}" )

DOCKER_RUN+=( ${IMAGE_NAME}:${TAG} )

echo "Starting container: ${CONTAINER_NAME} listening on port ${PORT}"
# shellcheck disable=SC2086
${DOCKER_RUN[@]}

echo "Container started. Showing recent logs (press Ctrl-C to exit follow):"

docker logs --tail 50 -f "$CONTAINER_NAME"
