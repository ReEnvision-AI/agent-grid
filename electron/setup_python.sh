#!/bin/bash

set -euo pipefail

echo "--- Setting up Python runtime for Electron app ---"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_VERSION="3.11.9"
RELEASE_TAG="20240415"
BASE_URL="https://github.com/indygreg/python-build-standalone/releases/download/${RELEASE_TAG}"

OS="$(uname -s)"
ARCH="$(uname -m)"

INSTALL_EXTRA="macos"
PLATFORM_KEY=""
PYTHON_DIST=""

case "${OS}" in
  Darwin)
    case "${ARCH}" in
      arm64)
        PLATFORM_KEY="darwin-arm64"
        PYTHON_DIST="cpython-${PYTHON_VERSION}+${RELEASE_TAG}-aarch64-apple-darwin-install_only.tar.gz"
        ;;
      x86_64)
        PLATFORM_KEY="darwin-x64"
        PYTHON_DIST="cpython-${PYTHON_VERSION}+${RELEASE_TAG}-x86_64-apple-darwin-install_only.tar.gz"
        ;;
    esac
    ;;
  Linux)
    case "${ARCH}" in
      x86_64)
        PLATFORM_KEY="linux-x64"
        PYTHON_DIST="cpython-${PYTHON_VERSION}+${RELEASE_TAG}-x86_64-unknown-linux-gnu-install_only.tar.gz"
        INSTALL_EXTRA="full"
        ;;
      arm64|aarch64)
        PLATFORM_KEY="linux-arm64"
        PYTHON_DIST="cpython-${PYTHON_VERSION}+${RELEASE_TAG}-aarch64-unknown-linux-gnu-install_only.tar.gz"
        INSTALL_EXTRA="full"
        ;;
    esac
    ;;
esac

if [[ -z "${PYTHON_DIST}" || -z "${PLATFORM_KEY}" ]]; then
  echo "Error: Unsupported OS/Architecture combination: ${OS}/${ARCH}" >&2
  exit 1
fi

DOWNLOAD_URL="${BASE_URL}/${PYTHON_DIST}"
TARGET_ROOT="${PROJECT_ROOT}/electron/python-runtime/${PLATFORM_KEY}"
PYTHON_DIR="${TARGET_ROOT}"
VENV_PATH="${TARGET_ROOT}/venv"
PIP_INSTALL_SPEC=".[${INSTALL_EXTRA}]"

echo "Preparing target directory: ${TARGET_ROOT}"
rm -rf "${TARGET_ROOT}"
mkdir -p "${TARGET_ROOT}"

TEMP_TARBALL="$(mktemp)"
cleanup() {
  rm -f "${TEMP_TARBALL}"
}
trap cleanup EXIT
echo "Downloading Python ${PYTHON_VERSION} runtime from:"
echo "  ${DOWNLOAD_URL}"
curl -fL -o "${TEMP_TARBALL}" "${DOWNLOAD_URL}"

echo "Extracting runtime..."
tar -xzf "${TEMP_TARBALL}" --strip-components=1 -C "${TARGET_ROOT}"

if [[ ! -x "${PYTHON_DIR}/bin/python3" ]]; then
  echo "Error: Extracted runtime missing python executable at ${PYTHON_DIR}/bin/python3" >&2
  exit 1
fi

echo "Creating bundled virtual environment at ${VENV_PATH}"
rm -rf "${VENV_PATH}"
"${PYTHON_DIR}/bin/python3" -m venv "${VENV_PATH}"

VENV_PYTHON="${VENV_PATH}/bin/python3"
VENV_PIP="${VENV_PATH}/bin/pip"

echo "Upgrading seed tools inside virtual environment..."
"${VENV_PIP}" install --upgrade pip setuptools wheel

echo "Installing Agent Grid (${PIP_INSTALL_SPEC}) into bundled venv..."
(
  cd "${PROJECT_ROOT}"
  "${VENV_PIP}" install --no-cache-dir "${PIP_INSTALL_SPEC}"
)

CODESIGN_IDENTITY="${CODESIGN_IDENTITY:-}"
if [[ -n "${CODESIGN_IDENTITY}" ]]; then
  echo "Codesigning bundled runtime with identity: ${CODESIGN_IDENTITY}"
  sign_targets=()
  while IFS= read -r -d '' candidate; do
    if file "${candidate}" | grep -qE 'Mach-O (64-bit|universal)'; then
      sign_targets+=("${candidate}")
    fi
  done < <(find "${PYTHON_DIR}" "${VENV_PATH}" -type f -print0)

  if [[ ${#sign_targets[@]} -eq 0 ]]; then
    echo "Warning: No Mach-O binaries detected to sign under ${PYTHON_DIR}."
  else
    for target in "${sign_targets[@]}"; do
      codesign --force --options runtime --timestamp --sign "${CODESIGN_IDENTITY}" "${target}"
    done
  fi
else
  echo "Skipping codesign step; set CODESIGN_IDENTITY to sign the runtime before packaging."
fi

ARCHIVE_PATH="${TARGET_ROOT}.tar.gz"
echo "Creating compressed runtime archive at ${ARCHIVE_PATH}"
tar -czf "${ARCHIVE_PATH}" -C "${TARGET_ROOT}" .

if [[ "${KEEP_UNPACKED_RUNTIME:-0}" != "1" ]]; then
  echo "Removing unpacked runtime directory (set KEEP_UNPACKED_RUNTIME=1 to keep it)."
  rm -rf "${TARGET_ROOT}"
fi

echo ""
echo "--- Python runtime setup complete ---"
if [[ -d "${TARGET_ROOT}" ]]; then
  echo "Bundled environment created at: ${TARGET_ROOT}"
fi
echo "Runtime archive available at: ${ARCHIVE_PATH}"
