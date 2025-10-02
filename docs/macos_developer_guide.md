# Agent Grid macOS Menu Bar App – Developer Guide

This guide walks through setting up and extending the macOS status bar companion to Agent Grid. It assumes you already have the Python backend checked out (this repository) and want to work on the Swift/SwiftUI front end located under `macos/AgentGridMenuApp/`.

## 1. Prerequisites
- macOS 13 Ventura or later (Apple Silicon or Intel).
- Xcode 15 or later.
- Python 3.11+ with the Agent Grid dependencies installed (e.g., via the repo’s `.venv`).
- A working `.env` and `models` file for `agentgrid` (used by the backend launcher).

Optional but recommended:
- Homebrew (`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`)
- `pyenv`/`uv` for isolated Python versions.

## 2. Repository Layout
```
repo/
 ├─ docs/
 │   ├─ macos_menu_app_plan.md        # UX + architecture outline
 │   └─ macos_developer_guide.md      # (this file)
 ├─ macos/
 │   └─ AgentGridMenuApp/
 │       ├─ AgentGridMenuApp.swift
 │       ├─ ConfigManager.swift
 │       ├─ StatusBarController.swift
 │       ├─ ServerProcessController.swift
 │       ├─ ModelDeviceDiscovery.swift
 │       ├─ PreferencesView.swift
 │       ├─ LogManager.swift
 │       └─ README.md
 └─ src/
     └─ agentgrid/launcher/
         ├─ __init__.py
         ├─ server.py                 # JSON-configurable launcher
         └─ discovery.py              # Model/device helpers
```

## 3. Creating the Xcode Project
1. Open Xcode → “Create a new project” → App → Name it `AgentGridMenuApp`, team/bundle identifier as desired.
2. Set the interface to SwiftUI, lifecycle to SwiftUI App, language Swift.
3. Target macOS 13 or later.
4. Delete the template Swift files Xcode generated and replace them with the ones from `macos/AgentGridMenuApp/`.
5. Add the files to the project (drag from Finder or File ➜ Add Files to “AgentGridMenuApp”…).
6. Ensure the “Embed in target” checkbox is checked when adding.

## 4. Wiring the Python Backend
The Swift scaffolding expects a Python interpreter and uses the new launcher helpers.

### Python Environment
- If you’re using the repo’s virtual environment, note the path (e.g., `/Users/you/source/repos/agent-grid/.venv/bin/python`).
- In macOS Preferences, point the app to this Python path.

### Runtime Config
`ConfigManager` writes a JSON file consumed by `agentgrid.launcher.server`. The default configuration uses:
```
converted_model_name_or_path: value selected in Preferences
torch_dtype: float16 (adjust later)
device: mps
port: 31331
identity_path: ./dev.id
warmup_tokens_interval: (optional)
new_swarm: true
```
Adjust the defaults or expose additional settings in the preferences UI as needed.

## 5. Swift Components Overview
- **AgentGridMenuApp.swift**: App entry point, sets up `AppDelegate`, Preferences scene.
- **StatusBarController.swift**: Manages the NSStatusItem, menu actions, and state-dependent icons.
- **ServerProcessController.swift**: Launches/stops the Python process, streams stdout/stderr, and exposes server state via `@Published`.
- **ConfigManager.swift**: Reads/writes preferences, builds JSON runtime config, placeholder Keychain storage for the HF token.
- **ModelDeviceDiscovery.swift**: Calls the Python discovery helper to list models/devices.
- **PreferencesView.swift**: SwiftUI form for editing Python path, models, env file, warmup interval, etc.
- **LogManager.swift**: Writes stdout/stderr to rotating log files under `~/Library/Application Support/AgentGrid/Logs` and reveals them in Finder.

## 6. Running the App
1. Build and run in Xcode.
2. The menu bar icon should appear; open Preferences to set Python path, models file, select a model, optionally warmup interval, and HF token.
3. Choose “Start Server…” from the status bar menu. The app writes the runtime config and spawns the Python launcher.
4. Check logs (View Logs) for progress or errors.

## 7. Development Checklist
- [ ] Implement real Keychain handling in `KeychainHelper` (replace the placeholder UserDefaults solution).
- [ ] Surface device choices (`ModelDeviceDiscovery`) in the Preferences UI.
- [ ] Parse stdout/stderr more robustly for state transitions (e.g., use structured log markers).
- [ ] Handle process restarts and auto-start on login if desired (using LaunchAgents).
- [ ] Add unit tests for `ConfigManager`, `ServerProcessController` (mocking `Process`).
- [ ] Consider using `swift-log` or a third-party logging framework for the macOS app itself.
- [ ] Harden bundle for distribution (signing, entitlements, notarization).

## 8. Common Issues
- **Python path invalid**: `ConfigError.pythonNotFound` will surface in the status menu. Verify the interpreter exists and is executable.
- **Models list empty**: Ensure `models` file is accessible; tap Refresh in Preferences; check that `python -m agentgrid.launcher.discovery --list-models` works manually.
- **Server fails immediately**: View logs for the Python process; typically missing `.env` or HF token.
- **Stuck processes**: “Stop Server” sends SIGTERM then SIGINT/SIGKILL fallback after 5s. If processes persist, check the log and adjust `ServerProcessController.stopServer()` as needed.

## 9. Next Steps
- Finish integrating the Swift components (UI updates, error prompts, notifications).
- Add sample `.env` / `models` templates for new users.
- Extend the macOS docs to cover packaging/signing when the UI stabilizes.

Refer to `docs/macos_menu_app_plan.md` for UX details and roadmap. Contributions welcome—coordinate via issues/PRs and note which part of the macOS stack you’re modifying.
