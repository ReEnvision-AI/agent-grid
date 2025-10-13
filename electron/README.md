# Agent Grid Electron Companion

This Electron app provides a tray UI for starting/stopping the Agent Grid server on macOS or Windows. It uses the `agentgrid.launcher.server` Python helper introduced in the backend refactor.

## Quick Start (when network access is available)

```bash
cd electron
npm install
npm start
```

To produce a signed and notarized macOS build with the embedded Python runtime, make sure the Developer ID certificate is available on the build host and configure the required environment variables before running the make command:

```bash
export CSC_NAME="Developer ID Application: Your Name (TEAMID)"
export APPLE_ID="apple-id@example.com"
export APPLE_APP_SPECIFIC_PASSWORD="xxxx-xxxx-xxxx-xxxx"
export APPLE_TEAM_ID="TEAMID"
npm run make
```

`npm run make` uses Electron Forge to package the app, sign it with the supplied Developer ID certificate, and submit it to Apple for notarization when the credentials above are present. Artifacts are written under `out/make`. If you only need an unsigned build (or are missing credentials), set `CSC_IDENTITY_AUTO_DISCOVERY=false` before running `npm run make` and Forge will skip the signing/notarization steps. If the build machine has no outbound network, make sure the matching `electron` npm package is already installed so Electron Forge does not attempt to download it at build time.

## Bundled Python Runtime

To build the Electron app, a self-contained Python environment is required. This environment is not checked into Git. Instead, a setup script is provided to create it for you.

**First-time Setup:**

Before you can run or package the app, you must generate the Python runtime. From the **project root directory**, run:

```bash
./electron/setup_python.sh
```

This script will:
1.  Detect your operating system and architecture.
2.  Download a standalone Python 3.11 environment.
3.  Extract it into the `electron/python-runtime/<platform>` directory (for example `darwin-arm64`).
4.  Create a pre-baked virtual environment with the required `agent-grid` dependencies (`[macos]` or `[full]`).

On first launch, the Electron app simply copies this bundled virtual environment into the user data directory, so it no longer needs to build a Python runtime or download packages on the target machine. If the bundle is missing, the app will refuse to start—always regenerate it before packaging. The Forge configuration copies everything under `python-runtime/` into the app bundle’s `Contents/Resources/python-runtime/` directory, so make sure that tree exists before running `npm run make`.

If you previously launched an older build, delete `~/Library/Application Support/Agent Grid/python-runtime` before running the refreshed app so it can copy the updated runtime bundle.

The `electron/python-runtime` directory is ignored by Git. If you make changes to the Python source code in `src/agentgrid` or change dependencies in `pyproject.toml`, you should re-run the script to generate a fresh environment.

> **Note:** The setup script now emits a compressed archive (`python-runtime/<platform>.tar.gz`). The Electron app extracts this archive into the user’s data directory the first time it runs. Set `KEEP_UNPACKED_RUNTIME=1` when invoking `./electron/setup_python.sh` if you also want to keep the unpacked runtime directory around for local debugging.

Electron Forge only copies the archive files into the `.app` bundle. Keeping the unpacked runtime directory in `electron/python-runtime/` is fine for local testing, but it will be ignored when packaging so the notarized build does not contain unsigned binaries.

If you plan to notarize the build, run `./electron/setup_python.sh` on a machine that has access to your Developer ID Application certificate and provide the signing identity via `CODESIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)"`. The script signs every Mach-O binary in the bundled runtime before it is archived, which keeps notarization from flagging the embedded Python libraries as unsigned.

The preferences window still allows pointing to an alternate interpreter when debugging or developing against a custom virtual environment.

## Project Structure

```
electron/
 ├─ main.js          # Electron main process (tray + process controller)
 ├─ preload.js       # Safe IPC bridge exposed to renderer
 ├─ renderer/
 │   ├─ index.html   # Preferences window UI
 │   ├─ renderer.js  # Vanilla JS logic for UI
 │   └─ styles.css   # Simple styling
 ├─ config/
 │   └─ defaults.json  # Initial settings
 ├─ package.json
 └─ README.md
```

## Current Status
- No external network calls; models list is hard-coded in `main.js` and the settings defaults.
- The UI is minimal (plain HTML/CSS/JS). You can replace it with React/Vue later.
- Process management streams stdout/stderr to `~/Library/Application Support/AgentGrid/logs/` (per Electron `userData`).

## Next Steps
1. Run `npm install` once you have internet access.
2. Consider adding React with Vite if you want a richer UI.
3. Integrate OS notifications when the server starts/stops or fails.
4. Package with `electron-forge` (update `forge.config.js` as needed).
