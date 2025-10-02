import SwiftUI
import AppKit
import Combine

struct ModelSelectionWindow: View {
    @ObservedObject var viewModel: ModelSelectionViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Select a Model")
                .font(.headline)
            Picker("Model", selection: $viewModel.selectedModel) {
                ForEach(viewModel.models, id: \.self) { model in
                    Text(model).tag(model)
                }
            }
            .pickerStyle(.menu)

            HStack {
                Spacer()
                Button("Cancel") { viewModel.cancel() }
                Button("Select") { viewModel.confirm() }
                    .disabled(viewModel.selectedModel.isEmpty)
            }
        }
        .padding(16)
        .frame(width: 320)
        .onAppear { viewModel.refreshModels() }
    }
}

final class ModelSelectionViewModel: ObservableObject {
    @Published var selectedModel: String
    @Published var models: [String] = []

    private let configManager: ConfigManager
    private let serverController: ServerProcessController
    private weak var window: NSWindow?

    init(configManager: ConfigManager, serverController: ServerProcessController, window: NSWindow?) {
        self.configManager = configManager
        self.serverController = serverController
        self.window = window
        self.selectedModel = UserDefaults.standard.string(forKey: "selectedModel") ?? ""
        ModelDeviceDiscovery.shared.$models
            .receive(on: RunLoop.main)
            .assign(to: &self.$models)
    }

    func refreshModels() {
        if let pythonURL = try? configManager.pythonExecutable() {
            ModelDeviceDiscovery.shared.refreshModels(modelsFile: configManager.modelsFileURL(), python: pythonURL)
        }
    }

    func confirm() {
        configManager.update(selectedModel: selectedModel)
        window?.close()
    }

    func cancel() {
        window?.close()
    }
}

extension ModelSelectionWindow {
    static func present(configManager: ConfigManager, serverController: ServerProcessController) {
        let window = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 320, height: 140),
                             styleMask: [.titled, .closable],
                             backing: .buffered,
                             defer: false)
        window.title = "Select Model"
        let viewModel = ModelSelectionViewModel(configManager: configManager, serverController: serverController, window: window)
        window.contentView = NSHostingView(rootView: ModelSelectionWindow(viewModel: viewModel))
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
}
