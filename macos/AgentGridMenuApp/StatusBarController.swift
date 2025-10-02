import AppKit
import Combine

final class StatusBarController {
    private let statusItem: NSStatusItem
    private let menu = NSMenu()
    private let serverController: ServerProcessController
    private let configManager: ConfigManager
    private var cancellables = Set<AnyCancellable>()

    init(serverController: ServerProcessController, configManager: ConfigManager) {
        self.serverController = serverController
        self.configManager = configManager
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        statusItem.button?.image = NSImage(systemSymbolName: "bolt.horizontal", accessibilityDescription: "Agent Grid")
        statusItem.menu = menu
        buildMenu()
        observeServerState()
    }

    private func buildMenu() {
        menu.autoenablesItems = false

        let startItem = NSMenuItem(title: "Start Server…", action: #selector(startServer), keyEquivalent: "s")
        startItem.target = self
        menu.addItem(startItem)

        let stopItem = NSMenuItem(title: "Stop Server", action: #selector(stopServer), keyEquivalent: "q")
        stopItem.target = self
        stopItem.tag = MenuTag.stop.rawValue
        menu.addItem(stopItem)

        menu.addItem(NSMenuItem.separator())

        let modelsItem = NSMenuItem(title: "Select Model", action: #selector(selectModel), keyEquivalent: "m")
        modelsItem.target = self
        menu.addItem(modelsItem)

        let logsItem = NSMenuItem(title: "View Logs", action: #selector(openLogs), keyEquivalent: "l")
        logsItem.target = self
        menu.addItem(logsItem)

        let prefsItem = NSMenuItem(title: "Preferences…", action: #selector(openPreferences), keyEquivalent: ",")
        prefsItem.target = self
        menu.addItem(prefsItem)

        menu.addItem(NSMenuItem.separator())

        let quitItem = NSMenuItem(title: "Quit Agent Grid", action: #selector(quitApp), keyEquivalent: "")
        quitItem.target = self
        menu.addItem(quitItem)
    }

    private func observeServerState() {
        serverController.$state
            .receive(on: RunLoop.main)
            .sink { [weak self] state in
                self?.updateMenu(for: state)
            }
            .store(in: &cancellables)
    }

    private func updateMenu(for state: ServerProcessController.State) {
        if let button = statusItem.button {
            switch state {
            case .stopped:
                button.image = NSImage(systemSymbolName: "bolt.horizontal", accessibilityDescription: "Agent Grid")
            case .starting:
                button.image = NSImage(systemSymbolName: "bolt.horizontal.circle", accessibilityDescription: "Starting")
            case .running:
                button.image = NSImage(systemSymbolName: "bolt.horizontal.fill", accessibilityDescription: "Running")
            case .failed:
                button.image = NSImage(systemSymbolName: "bolt.slash", accessibilityDescription: "Error")
            }
        }

        if let stopItem = menu.item(withTag: MenuTag.stop.rawValue) {
            stopItem.isEnabled = state == .running || state == .starting
        }
    }

    @objc private func startServer() {
        serverController.startServer()
    }

    @objc private func stopServer() {
        serverController.stopServer()
    }

    @objc private func selectModel() {
        ModelSelectionWindow.present(configManager: configManager, serverController: serverController)
    }

    @objc private func openPreferences() {
        NSApp.sendAction(Selector(("showPreferencesWindow:")), to: nil, from: nil)
    }

    @objc private func openLogs() {
        serverController.revealLogsInFinder()
    }

    @objc private func quitApp() {
        NSApp.terminate(nil)
    }
}

private enum MenuTag: Int {
    case stop = 1
}
