#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "--- Setting up Python runtime for Electron app ---"

# --- 1. Define constants and detect environment ---
PYTHON_VERSION="3.11.9"
RELEASE_TAG="20240415"
BASE_URL="https://github.com/indygreg/python-build-standalone/releases/download/${RELEASE_TAG}"

TARGET_DIR="electron/python-runtime"
OS=$(uname -s)
ARCH=$(uname -m)

PYTHON_DIST=""
# Default to the 'macos' extra, which pulls in 'inference' dependencies.
# This is suitable for cross-platform CPU-based execution.
INSTALL_EXTRA="macos"

case "$OS" in
    Darwin)
        case "$ARCH" in
            arm64)
                PYTHON_DIST="cpython-${PYTHON_VERSION}+${RELEASE_TAG}-aarch64-apple-darwin-install_only.tar.gz"
                ;;
            x86_64)
                PYTHON_DIST="cpython-${PYTHON_VERSION}+${RELEASE_TAG}-x86_64-apple-darwin-install_only.tar.gz"
                ;;
        esac
        ;;
    Linux)
        case "$ARCH" in
            x86_64)
                PYTHON_DIST="cpython-${PYTHON_VERSION}+${RELEASE_TAG}-x86_64-unknown-linux-gnu-install_only.tar.gz"
                # For Linux, we can use the 'full' install to get GPU support if available.
                INSTALL_EXTRA="full"
                ;;
            # Add other Linux architectures if needed, e.g., aarch64
        esac
        ;;
    # Add Windows (e.g., MINGW64_NT*) support here if needed in the future.
esac

if [ -z "$PYTHON_DIST" ]; then
    echo "Error: Unsupported OS/Architecture: $OS/$ARCH"
    exit 1
fi

DOWNLOAD_URL="${BASE_URL}/${PYTHON_DIST}"
INSTALL_CMD="pip install .[${INSTALL_EXTRA}]"

# --- 2. Clean and create target directory ---
echo "Preparing target directory: $TARGET_DIR"
if [ -d "$TARGET_DIR" ]; then
    echo "Removing existing runtime directory to ensure a clean setup."
    rm -rf "$TARGET_DIR"
fi
mkdir -p "$TARGET_DIR"

# --- 3. Download and extract Python ---
TEMP_FILE=$(mktemp)
echo "Downloading Python distribution for $OS $ARCH..."
echo "URL: $DOWNLOAD_URL"
curl -L -o "$TEMP_FILE" "$DOWNLOAD_URL"

echo "Extracting Python to $TARGET_DIR..."
# The archive extracts its content into a 'python' directory.
tar -xzf "$TEMP_FILE" -C "$TARGET_DIR"
# Move the contents from the nested 'python' directory to our target directory.
mv "$TARGET_DIR"/python/* "$TARGET_DIR"/
rmdir "$TARGET_DIR"/python

# --- 4. Install agent-grid and dependencies ---
PIP_PATH="$TARGET_DIR/bin/pip"
echo "Using pip at $PIP_PATH to install project dependencies..."
echo "Running command: $INSTALL_CMD (from project root)"

# It's good practice to upgrade pip first.
"$PIP_PATH" install --upgrade pip

# Execute the installation. This assumes the script is run from the project root.
"$PIP_PATH" install ".[${INSTALL_EXTRA}]"

# --- 5. Cleanup ---
echo "Cleaning up temporary download file..."
rm "$TEMP_FILE"

echo ""
echo "--- Python runtime setup complete! ---"
echo "Environment created at: $TARGET_DIR"
