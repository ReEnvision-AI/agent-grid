#!/bin/bash
set -e

# Build script for Agent Grid bootstrap server image
# Creates a lightweight CPU-only image for DHT bootstrap nodes

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

# Get the version from the Python package
VERSION=$(python3 -c "import sys; sys.path.insert(0, 'src'); import agentgrid; print(agentgrid.__version__)")
echo "Building Agent Grid container version $VERSION"


IMAGE_NAME="reenvision-ai/agent-grid"
IMAGE_WITH_VERSION="${IMAGE_NAME}:${VERSION}-bootstrap"


echo "Building Agent Grid bootstrap server image: ${IMAGE_WITH_VERSION}"

# Build the bootstrap image
podman build --format docker \
    -f Dockerfile.bootstrap \
    -t "${IMAGE_WITH_VERSION}" \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    .

echo "Build complete: ${IMAGE_WITH_VERSION}"

# Show image size
echo "Image size:"
podman images "${IMAGE_WITH_VERSION}" --format "table {{.Repository}}	{{.Tag}}	{{.Size}}"

CONTAINER_REGISTRY=${CONTAINER_REGISTERY:-"ghcr.io"}
# Optional: Tag for registry
if [[ -n "${CONTAINER_REGISTRY}" ]]; then
    echo "Logging into ${CONTAINER_REGISTRY}..."
    echo "$CR_PAT" | podman login "${CONTAINER_REGISTRY}" -u "$CR_USER" --password-stdin

    REGISTRY_IMAGE="${CONTAINER_REGISTRY}/${IMAGE_WITH_VERSION}"
    podman tag "${IMAGE_WITH_VERSION}" "${REGISTRY_IMAGE}"
    echo "Tagged for registry: ${REGISTRY_IMAGE}"
    podman push "${REGISTRY_IMAGE}"
fi

echo "Bootstrap image ready for deployment!"
echo ""
echo "To run locally (with default identity_0.key):"
echo "podman run -p 8788:8788 ${IMAGE_WITH_VERSION}"
echo ""
echo "To run with a specific identity file (e.g., identity_1.key):"
echo "podman run -p 8788:8788 -e IDENTITY_FILENAME=identity_1.key ${IMAGE_WITH_VERSION}"
echo ""
echo "To run with custom peers:"
echo "podman run -p 8788:8788 ${IMAGE_WITH_VERSION} agent-grid-bootstrap --initial_peers /ip4/x.x.x.x/tcp/8788/p2p/..."
