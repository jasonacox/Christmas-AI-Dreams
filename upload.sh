#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="jasonacox/christmas-ai-dreams"

echo "Build and push ${IMAGE_NAME} to Docker Hub"

# Accept optional tag as first arg, or prompt
TAG=${1:-}
if [ -z "$TAG" ]; then
    # Try to read VERSION from server.py
    if [ -f server.py ]; then
        # Robustly extract the first quoted string on the VERSION= line
        VERSION_FROM_FILE=$(grep -E '^\s*VERSION\s*=' server.py | head -n1 | sed -E 's/.*["'"'\'"']([^"'"'\'"']+)["'"'\'"'].*/\1/') || true
    else
        VERSION_FROM_FILE=""
    fi
    if [ -n "$VERSION_FROM_FILE" ]; then
        TAG=$VERSION_FROM_FILE
        echo "Using version from server.py: $TAG"
    else
        read -p "Enter tag to push (default: latest): " TAG
        TAG=${TAG:-latest}
    fi
fi

echo "Tag: ${TAG}"

# Confirm computed tag with the user before building
echo "Computed tag: ${TAG}"
read -p "Use this tag? [Y/n] " CONFIRM
CONFIRM=${CONFIRM:-Y}
if [[ ! "$CONFIRM" =~ ^([yY])$ ]]; then
    read -p "Enter tag to push (default: latest): " TAG_INPUT
    TAG=${TAG_INPUT:-latest}
fi

echo "Building ${IMAGE_NAME}:${TAG}"
docker buildx build --platform linux/amd64,linux/arm64,linux/arm/v7 --push -t ${IMAGE_NAME}:${TAG} .

if [ "$TAG" != "latest" ]; then
    echo "Also tagging as latest"
    docker buildx build --platform linux/amd64,linux/arm64,linux/arm/v7 --push -t ${IMAGE_NAME}:latest .
fi

echo "Verify image manifests"
docker buildx imagetools inspect ${IMAGE_NAME}:${TAG} | grep Platform || true

echo "Done."

