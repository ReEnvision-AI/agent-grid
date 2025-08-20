#!/bin/bash

# This script builds and pushes a Docker image to a container registry.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---

# Default value for WITH_FLASH, can be overridden by command-line argument
WITH_FLASH=${1:-"false"}

# --- Main Script ---

echo "Starting deployment script..."

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    set -a # automatically export all variables
    source .env
    set +a # stop exporting
fi

# Check for required environment variables
if [ -z "$CR_PAT" ]; then
    echo "Error: CR_PAT is not set." >&2
    exit 1
fi

if [ -z "$CR_USER" ]; then
    echo "Error: CR_USER is not set." >&2
    exit 1
fi

# Define the version for the Docker Image, default to 1.0.0
VERSION=${AGENT_GRID_VERSION:-"1.0.0"}
echo "Building Agent Grid container version $VERSION"

IMAGE_NAME="ghcr.io/reenvision-ai/agent-grid"

# Set the builder image and image name based on WITH_FLASH
if [ "$WITH_FLASH" = "true" ]; then
    echo "Building with Flash Attention"
    BUILDER_IMAGE="ghcr.io/reenvision-ai/base-image:1.1.0-flash"
    IMAGE_WITH_VERSION="${IMAGE_NAME}:${VERSION}-flash"
    LATEST_IMAGE="${IMAGE_NAME}:latest-flash"
else
    echo "Building without Flash Attention"
    BUILDER_IMAGE="ghcr.io/reenvision-ai/base-image:1.1.0"
    IMAGE_WITH_VERSION="${IMAGE_NAME}:${VERSION}"
    LATEST_IMAGE="${IMAGE_NAME}:latest"
fi

# Login to Github Container Registry using podman
echo "Logging into ghcr.io..."
echo "$CR_PAT" | podman login ghcr.io -u "$CR_USER" --password-stdin

# Build the image with podman
echo "Building image: $IMAGE_WITH_VERSION"
podman build --build-arg BUILDER_IMAGE="$BUILDER_IMAGE" -t "$IMAGE_WITH_VERSION" .

# Push the image to the registry
echo "Pushing image: $IMAGE_WITH_VERSION"
podman push "$IMAGE_WITH_VERSION"

# If the version is not 'latest' and not a 'beta' version,
# then tag and push it as the latest version as well.
if [[ "$VERSION" != "latest" && "$VERSION" != *"beta"* ]]; then
    echo "Tagging image as latest..."
    podman tag "$IMAGE_WITH_VERSION" "$LATEST_IMAGE"

    echo "Pushing latest image: $LATEST_IMAGE"
    podman push "$LATEST_IMAGE"
fi

echo "Deployment successful."
