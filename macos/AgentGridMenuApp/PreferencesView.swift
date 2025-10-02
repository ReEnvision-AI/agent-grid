import SwiftUI
import Combine

final class PreferencesViewModel: ObservableObject {
    @Published var pythonPath: String
    @Published var modelsPath: String
    @Published var envPath: String
    @Published var selectedModel: String
    @Published var warmupInterval: Int
    @Published var availableModels: [String] = []

    private let configManager: ConfigManager
    private let serverController: ServerProcessController
    private var cancellables = Set<AnyCancellable>()

    init(configManager: ConfigManager, serverController: ServerProcessController) {
        self.configManager = configManager
        self.serverController = serverController
        pythonPath = UserDefaults.standard.string(forKey: "pythonPath") ?? "/usr/bin/python3"
        modelsPath = UserDefaults.standard.string(forKey: "modelsPath") ?? "models"
        envPath = UserDefaults.standard.string(forKey: "envPath") ?? ".env"
        selectedModel = UserDefaults.standard.string(forKey: "selectedModel") ?? ""
        warmupInterval = UserDefaults.standard.integer(forKey: "warmupInterval")

        ModelDeviceDiscovery.shared.$models
            .receive(on: RunLoop.main)
            .assign(to: &self.$availableModels)
    }

    func refreshModels() {
        guard let pythonURL = try? configManager.pythonExecutable() else { return }
        ModelDeviceDiscovery.shared.refreshModels(modelsFile: URL(fileURLWithPath: modelsPath), python: pythonURL)
    }

    func save() {
        configManager.update(pythonPath: pythonPath)
        configManager.update(modelsPath: modelsPath)
        configManager.update(envPath: envPath)
        configManager.update(warmupInterval: warmupInterval)
        configManager.update(selectedModel: selectedModel)
    }
}

struct PreferencesView: View {
    @ObservedObject var viewModel: PreferencesViewModel

    var body: some View {
        Form {
            Section(header: Text("General")) {
                Picker("Model", selection: $viewModel.selectedModel) {
                    ForEach(viewModel.availableModels, id: \.self) { model in
                        Text(model).tag(model)
                    }
                }
                Button("Refresh Models") {
                    viewModel.refreshModels()
                }
                Stepper(value: $viewModel.warmupInterval, in: 0...32768, step: 512) {
                    Text("Warmup tokens interval: \(viewModel.warmupInterval)")
                }
            }

            Section(header: Text("Paths")) {
                TextField("Python", text: $viewModel.pythonPath)
                TextField("Models", text: $viewModel.modelsPath)
                TextField(".env", text: $viewModel.envPath)
            }

            Section(header: Text("Token")) {
                SecureField("Hugging Face Token", text: Binding(
                    get: { KeychainHelper.shared.hfToken ?? "" },
                    set: { KeychainHelper.shared.hfToken = $0 }
                ))
            }
        }
        .padding(16)
        .frame(width: 420)
        .onAppear {
            viewModel.refreshModels()
        }
        .toolbar {
            ToolbarItem(placement: .confirmationAction) {
                Button("Save") {
                    viewModel.save()
                }
            }
        }
    }
}
