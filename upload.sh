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

echo "Verify image manifests for ${IMAGE_NAME}:${TAG}"
docker buildx imagetools inspect ${IMAGE_NAME}:${TAG} | grep Platform || true

# Create and push a git tag for this version (skip if tag is 'latest')
if [ "${TAG}" != "latest" ]; then
    # Normalize tag name: use as-is if it starts with 'v', otherwise prefix with 'v'
    if [[ "${TAG}" =~ ^v ]]; then
        TAG_NAME="${TAG}"
    else
        TAG_NAME="v${TAG}"
    fi

    if git rev-parse --git-dir >/dev/null 2>&1; then
        if git rev-parse --verify "refs/tags/${TAG_NAME}" >/dev/null 2>&1; then
            echo "Git tag ${TAG_NAME} already exists locally. Pushing to origin..."
        else
            echo "Creating annotated git tag ${TAG_NAME}"
            git tag -a "${TAG_NAME}" -m "Release ${TAG_NAME}" || true
        fi
        echo "Pushing tag ${TAG_NAME} to origin"
        git push origin "${TAG_NAME}" || true
    else
        echo "Not a git repository, skipping git tag creation"
    fi
fi

echo "Done."

