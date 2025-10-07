# Agent Grid Electron Companion

This Electron app provides a tray UI for starting/stopping the Agent Grid server on macOS or Windows. It uses the `agentgrid.launcher.server` Python helper introduced in the backend refactor.

## Quick Start (when network access is available)

```bash
cd electron
npm install
npm start
```

To package the desktop app with the embedded Python runtime:

```bash
CSC_IDENTITY_AUTO_DISCOVERY=false npm run dist
```

The environment variable disables automatic code-signing discovery; provide your signing identity when you are ready to notarize. If the build machine has no outbound network, make sure the matching `electron` npm package is already installed so `electron-builder` does not attempt to download it at build time.

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

On first launch, the Electron app simply copies this bundled virtual environment into the user data directory, so it no longer needs to build a Python runtime or download packages on the target machine. If the bundle is missing, the app will refuse to start—always regenerate it before packaging. The `electron/package.json` build config copies everything under `python-runtime/` into the app bundle’s `Contents/Resources/python-runtime/` directory, so make sure that tree exists before running `npm run dist`.

If you previously launched an older build, delete `~/Library/Application Support/agent-grid-electron/python-runtime` before running the refreshed app so it can copy the updated runtime bundle.

The `electron/python-runtime` directory is ignored by Git. If you make changes to the Python source code in `src/agentgrid` or change dependencies in `pyproject.toml`, you should re-run the script to generate a fresh environment.

> **Note:** The setup script now emits a compressed archive (`python-runtime/<platform>.tar.gz`). The Electron app extracts this archive into the user’s data directory the first time it runs. Set `KEEP_UNPACKED_RUNTIME=1` when invoking `./electron/setup_python.sh` if you also want to keep the unpacked runtime directory around for local debugging.

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
4. Package with `electron-builder` (configure `electron-builder.yml` or extend `package.json` scripts).
