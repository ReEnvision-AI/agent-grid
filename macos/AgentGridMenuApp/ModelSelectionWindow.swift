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
    @Published var models: [String]

    private let configManager: ConfigManager
    private weak var window: NSWindow?

    init(configManager: ConfigManager, window: NSWindow?) {
        self.configManager = configManager
        self.window = window
        let defaults = ConfigManager.defaultModels
        self.models = defaults
        let current = configManager.currentSelectedModel()
        if defaults.contains(current) {
            self.selectedModel = current
        } else {
            self.selectedModel = defaults.first ?? ""
        }
    }

    func refreshModels() {
        models = ConfigManager.defaultModels
        if !models.contains(selectedModel) {
            selectedModel = models.first ?? ""
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
    static func present(configManager: ConfigManager) {
        let panel = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 320, height: 140),
                            styleMask: [.titled, .closable],
                            backing: .buffered,
                            defer: false)
        panel.title = "Select Model"
        let viewModel = ModelSelectionViewModel(configManager: configManager, window: panel)
        panel.contentView = NSHostingView(rootView: ModelSelectionWindow(viewModel: viewModel))
        panel.center()
        panel.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
}
