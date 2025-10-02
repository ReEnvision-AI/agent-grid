import SwiftUI
import Combine
import AppKit

@main
struct AgentGridMenuApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        Settings {
            PreferencesView(viewModel: appDelegate.preferencesViewModel)
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    private(set) lazy var configManager = ConfigManager()
    private(set) lazy var serverController = ServerProcessController(configManager: configManager)
    private(set) lazy var preferencesViewModel = PreferencesViewModel(configManager: configManager, serverController: serverController)
    private var statusBarController: StatusBarController?

    func applicationDidFinishLaunching(_ notification: Notification) {
        statusBarController = StatusBarController(serverController: serverController, configManager: configManager)
        serverController.bootstrap()
    }

    func applicationWillTerminate(_ notification: Notification) {
        serverController.stopServer()
    }
}
