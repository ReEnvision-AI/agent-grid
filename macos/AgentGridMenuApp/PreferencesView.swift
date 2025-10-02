import SwiftUI
import Combine

final class PreferencesViewModel: ObservableObject {
    @Published var pythonPath: String
    @Published var modelsPath: String
    @Published var envPath: String
    @Published var selectedModel: String
    @Published var warmupInterval: Int
    @Published var device: String
    @Published var torchDType: String
    @Published var quantType: String
    @Published var port: String
    @Published var identityPath: String
    @Published var initialPeers: String
    @Published var newSwarm: Bool
    @Published var availableModels: [String]
    @Published var availableDevices: [String] = []

    private let configManager: ConfigManager
    private let serverController: ServerProcessController
    private var cancellables = Set<AnyCancellable>()

    init(configManager: ConfigManager, serverController: ServerProcessController) {
        self.configManager = configManager
        self.serverController = serverController
        pythonPath = configManager.currentPythonPath()
        modelsPath = configManager.currentModelsPath()
        envPath = configManager.currentEnvPath()
        selectedModel = configManager.currentSelectedModel()
        warmupInterval = configManager.currentWarmupInterval()
        device = configManager.currentDevice()
        torchDType = configManager.currentTorchDType()
        quantType = configManager.currentQuantType()
        port = String(configManager.currentPort())
        identityPath = configManager.currentIdentityPath()
        initialPeers = configManager.currentInitialPeers()
        newSwarm = configManager.currentNewSwarm()
        availableModels = ConfigManager.defaultModels
        availableDevices = ["cpu", "mps", "cuda"]
        if !availableModels.contains(selectedModel) {
            selectedModel = availableModels.first ?? ""
        }

        ModelDeviceDiscovery.shared.$devices
            .receive(on: RunLoop.main)
            .sink { [weak self] devices in
                guard let self = self else { return }
                let available = devices
                    .filter { $0.value.available }
                    .map { $0.key }
                self.availableDevices = available.isEmpty ? ["cpu", "mps", "cuda"] : available
                if !self.availableDevices.contains(self.device) {
                    self.device = self.availableDevices.first ?? "cpu"
                }
            }
            .store(in: &cancellables)
    }

    func refreshModels() {
        availableModels = ConfigManager.defaultModels
        if !availableModels.contains(selectedModel) {
            selectedModel = availableModels.first ?? ""
        }
    }

    func refreshDevices() {
        ModelDeviceDiscovery.shared.refreshDevices()
    }

    func save() {
        configManager.update(pythonPath: pythonPath)
        configManager.update(modelsPath: modelsPath)
        configManager.update(envPath: envPath)
        configManager.update(warmupInterval: warmupInterval)
        configManager.update(selectedModel: selectedModel)
        configManager.update(device: device)
        configManager.update(torchDType: torchDType)
        configManager.update(quantType: quantType)
        configManager.update(port: Int(port) ?? 31331)
        configManager.update(identityPath: identityPath)
        configManager.update(initialPeers: initialPeers)
        configManager.update(newSwarm: newSwarm)
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
                Button("Refresh Models") { viewModel.refreshModels() }
                Stepper(value: $viewModel.warmupInterval, in: 0...32768, step: 512) {
                    Text("Warmup tokens interval: \(viewModel.warmupInterval)")
                }
            }

            Section(header: Text("Server")) {
                Picker("Device", selection: $viewModel.device) {
                    ForEach(viewModel.availableDevices, id: \.self) { device in
                        Text(device.uppercased()).tag(device)
                    }
                }
                Button("Detect Devices") { viewModel.refreshDevices() }
                TextField("Torch dtype", text: $viewModel.torchDType)
                TextField("Quant type", text: $viewModel.quantType)
                TextField("Port", text: $viewModel.port)
                TextField("Identity path", text: $viewModel.identityPath)
                Toggle("Start new swarm", isOn: $viewModel.newSwarm)
                TextField("Initial peers (comma separated)", text: $viewModel.initialPeers)
                    .disabled(viewModel.newSwarm)
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
        .frame(width: 460)
        .onAppear {
            viewModel.refreshModels()
            viewModel.refreshDevices()
        }
        .toolbar {
            ToolbarItem(placement: .confirmationAction) {
                Button("Save") { viewModel.save() }
            }
        }
    }
}
