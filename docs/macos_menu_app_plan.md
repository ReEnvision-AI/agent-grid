# Agent Grid macOS Menu Bar App Plan

## Goals
- Provide a native macOS status bar app that can start/stop an Agent Grid server with minimal setup.
- Surface server status, logs, and configuration in a user-friendly way.
- Reuse the new Python launcher helpers so macOS and Linux share the same backend behaviour.
- Keep the CLI scripts fully functional for advanced users.

## User Experience
### Status Bar Menu
- Icon reflects state: ⭕️ Stopped, 🟡 Starting, 🟢 Running, 🔴 Error.
- Menu items:
  1. `Start Server…` – opens a configuration sheet.
  2. `Stop Server` – visible when running.
  3. `Select Model` – quick picker populated from the models file/helper.
  4. `Preferences…` – window with advanced settings (Python path, env files, warmup interval, auto-start).
  5. `View Logs` – opens log directory in Finder or embedded log window.
  6. `Check Devices` – optional submenu showing CUDA/MPS availability from the helper script.
  7. `Quit` – stops the server if running, then exits.

### Preferences Window
- **General** tab: autostart on login, default device/quant settings, warmup interval toggle.
- **Paths** tab: Python interpreter picker, `.env` file, `models` file, working directory.
- **Tokens** tab: Hugging Face token stored in Keychain (prompt if missing).
- **Logs** tab: log history, button to reveal in Finder, copy diagnostics.

### UX Notes
- Show a toast/notification when server starts/stops or fails.
- In error states, menu item displays the latest stderr summary.
- Provide a one-time onboarding flow guiding the user to locate `.env` and `models` files or create them.

## Architecture
```
+-------------------------------+
| macOS App (Swift/SwiftUI)     |
|  - StatusBarController        |
|  - PreferencesView            |
|  - LogWindowController        |
+---------------+---------------+
                |
                | Process IPC (Foundation.Process)
                v
+-------------------------------+
| Python Runtime (venv)         |
|  - agentgrid.launcher.server  |
|  - agentgrid.launcher.discovery|
+-------------------------------+
```

### Components
- **StatusBarController**
  - Owns `NSStatusItem`, updates icon/menu based on `ServerState`.
  - Handles menu actions, triggers start/stop via `ServerProcessController`.

- **ServerProcessController**
  - Wraps `Process` for `python -m agentgrid.launcher.server` with JSON config.
  - Redirects stdout/stderr to a logging pipeline, updates state via Combine publishers.
  - Implements graceful shutdown (TERM + wait + KILL fallback).

- **ConfigManager**
  - Stores settings in `UserDefaults`/`Application Support/AgentGrid/config.json`.
  - Performs validation (Python path existence, file accessibility).
  - Exposes computed config dict for JSON export to the Python launcher.

- **DiscoveryBridge**
  - Runs `python -m agentgrid.launcher.discovery --list-models` and `--probe-devices`.
  - Caches results with expiration to avoid spamming subprocess creation.

- **LogManager**
  - Receives stdout/stderr data, writes to rotating files in `~/Library/Logs/AgentGrid`.
  - Publishes log streams to UI.

### Process Lifecycle
1. User selects `Start Server…`.
2. Gather config via sheet/preferences -> produce JSON payload.
3. Launch `Process` with:
   ```
   $PYTHON_BIN -m agentgrid.launcher.server /path/to/generated-config.json
   ```
4. Subscribe to stdout/stderr pipes, parse `INFO/ERROR` for state transitions.
5. `Stop Server` sends SIGTERM; if Process refuses to die within N seconds, send SIGKILL.
6. On exit, update state, clear warmup counters, show notification.

## Config Serialization
- App generates a temporary JSON file in `~/Library/Application Support/AgentGrid/runtime-config.json` containing:
  ```json
  {
    "converted_model_name_or_path": "unsloth/gpt-oss-20b-BF16",
    "torch_dtype": "float16",
    "device": "mps",
    "port": 31331,
    "token": "${HF_TOKEN}",
    "cache_dir": "~/Library/Application Support/AgentGrid/cache",
    "warmup_tokens_interval": 2048,
    "new_swarm": true,
    "identity_path": "./dev.id"
  }
  ```
- The launcher handles validation and defaulting; the app only ensures user-friendly inputs.

## Development Steps
1. **Project Setup**
   - Create Xcode project (Swift/SwiftUI), target macOS 13+.
   - Add status bar entry and placeholder menu.
   - Integrate Sparkle or similar later (optional).

2. **Process Controller**
   - Build Swift wrapper for launching/stopping Python using `Process`.
   - Implement Combine publishers for `ServerState` and log lines.

3. **Preferences & Persisted Settings**
   - Build SwiftUI views for General/Paths/Tokens.
   - Store in `UserDefaults` (with Keychain for secrets); add validation UI feedback.

4. **Discovery Integration**
   - Hook the `discovery` helper to populate model/device lists.
   - Cache results and allow manual refresh.

5. **Logging & Diagnostics**
   - Build log window/console, support copying diagnostics.
   - Rotate logs (e.g., keep last 10 files of 1MB each).

6. **Error Handling/Notifications**
   - Add NSUserNotificationCenter alerts (or modern API) on failures.
   - Show retry option in the menu when a launch fails.

7. **Packaging**
   - Add Build Script to embed config templates (sample `.env`, readme).
   - Plan for notarization; include instructions for user to select Python or provide bundled interpreter later.

8. **Testing**
   - Manual test matrix: Apple Silicon + Intel macOS, CPU vs MPS.
   - Automated Swift unit tests for ConfigManager, ProcessController mocks.
   - Ensure CLI scripts remain unaffected (Python tests still pass).

## Open Questions
- Should we bundle a specific Python runtime? For initial MVP we rely on system/venv; bundling can come later.
- How to surface log location to the user? Possibly a “Reveal in Finder” action.
- Should the app support multiple concurrent servers? Out of scope for MVP; UI assumes one server.

## Next Actions
1. Scaffold Xcode project with status bar item and placeholder menu.
2. Implement `ServerProcessController` prototype that launches the new Python launcher and streams logs.
3. Hook discovery helper for model list in UI.

These steps set the stage for iterating on UI/UX while leveraging the backend refactor completed earlier.
