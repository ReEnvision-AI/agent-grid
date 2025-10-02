# Agent Grid Menu Bar App (macOS)

This directory contains a Swift/SwiftUI scaffolding for the macOS menu bar app described in `docs/macos_menu_app_plan.md`.

## Getting Started
1. Open Xcode and create a new **App** project named `AgentGridMenuApp` targeting macOS 13 or later, SwiftUI + Swift language.
2. Replace the generated sources with the files in this directory.
3. Ensure the app target is configured with the bundle identifier you plan to use.
4. Add `Python` as a required dependency in your developer documentation (the app expects a Python executable path).

### Structure
```
macos/AgentGridMenuApp/
 ├─ AgentGridMenuApp.swift        # App entry point & AppDelegate hookup
 ├─ StatusBarController.swift     # Manages NSStatusItem and menus
 ├─ ServerProcessController.swift # Launches/stops Python process
 ├─ ConfigManager.swift           # Persists user settings (UserDefaults/Keychain)
 ├─ ModelDeviceDiscovery.swift    # Bridges to Python discovery helpers
 ├─ PreferencesView.swift         # SwiftUI preferences window
 └─ README.md                     # This file
```

After copying these files into your Xcode project, verify the following build settings:
- Deployment target: macOS 13+
- Use Swift concurrency features where helpful (Combine, async/await)
- Add the `App Sandbox` entitlement only after deciding on runtime requirements.

The Swift files include TODOs for future work (e.g., wiring Keychain storage) and provide placeholder Combine publishers to feed the UI.

## Menu Bar Only Setup
- In the app target's Info "Custom macOS Application Property" add `Application is agent (UIElement)` (key `LSUIElement`) and set it to `YES` so the app stays in the menu bar without a dock icon.
