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
: "${SWARMUI:=http://localhost:7801}"
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

# Build docker run command with explicit env vars and defaults
DOCKER_RUN=(docker run -d --name "$CONTAINER_NAME" --restart unless-stopped -p ${PORT}:4000)

# Basic provider selection and SwarmUI defaults
DOCKER_RUN+=( -e "IMAGE_PROVIDER=${IMAGE_PROVIDER:-swarmui}" )
DOCKER_RUN+=( -e "SWARMUI=${SWARMUI:-http://localhost:7801}" )

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
