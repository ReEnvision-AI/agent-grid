import Foundation
import AppKit

enum ConfigError: Error {
    case pythonNotFound
    case invalidModelsFile
    case unableToWriteRuntimeConfig
}

final class ConfigManager {
    static let defaultModels = [
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",
        "unsloth/gpt-oss-20b-BF16"
    ]

    struct RuntimeConfiguration: Codable {
        var converted_model_name_or_path: String
        var torch_dtype: String
        var device: String
        var port: Int
        var token: String?
        var identity_path: String
        var warmup_tokens_interval: Int?
        var new_swarm: Bool
        var quant_type: String?
        var initial_peers: [String]?
    }

    private enum Keys {
        static let pythonPath = "pythonPath"
        static let modelsPath = "modelsPath"
        static let envPath = "envPath"
        static let selectedModel = "selectedModel"
        static let warmupInterval = "warmupInterval"
        static let device = "device"
        static let torchDType = "torchDType"
        static let quantType = "quantType"
        static let port = "port"
        static let identityPath = "identityPath"
        static let initialPeers = "initialPeers"
        static let newSwarm = "newSwarm"
    }

    private let defaults = UserDefaults.standard
    private let fileManager = FileManager.default

    func load() {
        defaults.register(defaults: [
            Keys.pythonPath: "/usr/bin/python3",
            Keys.modelsPath: "models",
            Keys.envPath: ".env",
            Keys.selectedModel: ConfigManager.defaultModels.first ?? "",
            Keys.warmupInterval: 0,
            Keys.device: defaultDevice(),
            Keys.torchDType: defaultTorchDType(),
            Keys.quantType: "none",
            Keys.port: 31331,
            Keys.identityPath: "./dev.id",
            Keys.initialPeers: "",
            Keys.newSwarm: true
        ])
    }

    // MARK: - Current values

    func currentPythonPath() -> String { defaults.string(forKey: Keys.pythonPath) ?? "/usr/bin/python3" }
    func currentModelsPath() -> String { defaults.string(forKey: Keys.modelsPath) ?? "models" }
    func currentEnvPath() -> String { defaults.string(forKey: Keys.envPath) ?? ".env" }
    func currentSelectedModel() -> String {
        let saved = defaults.string(forKey: Keys.selectedModel) ?? ""
        if !saved.isEmpty { return saved }
        return ConfigManager.defaultModels.first ?? ""
    }
    func currentWarmupInterval() -> Int { defaults.integer(forKey: Keys.warmupInterval) }
    func currentDevice() -> String { defaults.string(forKey: Keys.device) ?? defaultDevice() }
    func currentTorchDType() -> String { defaults.string(forKey: Keys.torchDType) ?? defaultTorchDType() }
    func currentQuantType() -> String { defaults.string(forKey: Keys.quantType) ?? "none" }
    func currentPort() -> Int {
        let value = defaults.integer(forKey: Keys.port)
        return value > 0 ? value : 31331
    }
    func currentIdentityPath() -> String { defaults.string(forKey: Keys.identityPath) ?? "./dev.id" }
    func currentInitialPeers() -> String { defaults.string(forKey: Keys.initialPeers) ?? "" }
    func currentNewSwarm() -> Bool { defaults.bool(forKey: Keys.newSwarm) }

    // MARK: - Paths

    func pythonExecutable() throws -> URL {
        let url = URL(fileURLWithPath: currentPythonPath())
        if fileManager.isExecutableFile(atPath: url.path) { return url }
        throw ConfigError.pythonNotFound
    }

    func modelsFileURL() -> URL { URL(fileURLWithPath: currentModelsPath()) }
    func environmentFileURL() -> URL { URL(fileURLWithPath: currentEnvPath()) }
    func workingDirectory() -> URL { URL(fileURLWithPath: fileManager.currentDirectoryPath) }

    // MARK: - Environment

    func environmentVariables() -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        env["AGENTGRID_MODELS_FILE"] = modelsFileURL().path
        env["AGENTGRID_ENV_PATH"] = environmentFileURL().path
        if let token = KeychainHelper.shared.hfToken, !token.isEmpty {
            env["HF_TOKEN"] = token
        }
        env.merge(parseEnvFile()) { _, new in new }
        return env
    }

    // MARK: - Runtime Config

    func buildRuntimeConfiguration() throws -> RuntimeConfiguration {
        let model = try resolveModelSelection()
        return RuntimeConfiguration(
            converted_model_name_or_path: model,
            torch_dtype: currentTorchDType(),
            device: currentDevice(),
            port: currentPort(),
            token: KeychainHelper.shared.hfToken,
            identity_path: currentIdentityPath(),
            warmup_tokens_interval: warmupInterval(),
            new_swarm: currentNewSwarm(),
            quant_type: currentQuantType(),
            initial_peers: peersList()
        )
    }

    func warmupInterval() -> Int? {
        let value = currentWarmupInterval()
        return value > 0 ? value : nil
    }

    func writeRuntimeConfig(_ config: RuntimeConfiguration) throws -> URL {
        let support = try fileManager.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let dir = support.appendingPathComponent("AgentGrid", isDirectory: true)
        try fileManager.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("runtime-config.json")
        let data = try JSONEncoder().encode(config)
        try data.write(to: url, options: .atomic)
        return url
    }

    // MARK: - Updates

    func update(selectedModel: String) { defaults.set(selectedModel, forKey: Keys.selectedModel) }
    func update(pythonPath: String) { defaults.set(pythonPath, forKey: Keys.pythonPath) }
    func update(modelsPath: String) { defaults.set(modelsPath, forKey: Keys.modelsPath) }
    func update(envPath: String) { defaults.set(envPath, forKey: Keys.envPath) }
    func update(warmupInterval: Int) { defaults.set(warmupInterval, forKey: Keys.warmupInterval) }
    func update(device: String) { defaults.set(device, forKey: Keys.device) }
    func update(torchDType: String) { defaults.set(torchDType, forKey: Keys.torchDType) }
    func update(quantType: String) { defaults.set(quantType, forKey: Keys.quantType) }
    func update(port: Int) { defaults.set(port, forKey: Keys.port) }
    func update(identityPath: String) { defaults.set(identityPath, forKey: Keys.identityPath) }
    func update(initialPeers: String) { defaults.set(initialPeers, forKey: Keys.initialPeers) }
    func update(newSwarm: Bool) { defaults.set(newSwarm, forKey: Keys.newSwarm) }

    // MARK: - Helpers

    private func resolveModelSelection() throws -> String {
        let model = currentSelectedModel()
        if !model.isEmpty { return model }
        if let fallback = ConfigManager.defaultModels.first { return fallback }
        throw ConfigError.invalidModelsFile
    }

    private func peersList() -> [String]? {
        if currentNewSwarm() { return [] }
        let separators = CharacterSet(charactersIn: ",\n")
        let tokens = currentInitialPeers()
            .components(separatedBy: separators)
            .flatMap { $0.components(separatedBy: CharacterSet.whitespaces) }
        let peers = tokens
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        return peers.isEmpty ? nil : peers
    }

    private func parseEnvFile() -> [String: String] {
        var result: [String: String] = [:]
        let url = environmentFileURL()
        guard fileManager.fileExists(atPath: url.path),
              let contents = try? String(contentsOf: url, encoding: .utf8) else {
            return result
        }
        for rawLine in contents.split(separator: "\n") {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty || line.hasPrefix("#") { continue }
            let parts = line.split(separator: "=", maxSplits: 1)
            guard parts.count == 2 else { continue }
            let key = String(parts[0]).trimmingCharacters(in: .whitespacesAndNewlines)
            let value = String(parts[1]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !key.isEmpty { result[key] = value }
        }
        return result
    }

    private func defaultDevice() -> String {
        #if os(macOS)
        return "mps"
        #else
        return "cpu"
        #endif
    }

    private func defaultTorchDType() -> String {
        defaultDevice() == "cuda" ? "bfloat16" : "float16"
    }
}

final class KeychainHelper {
    static let shared = KeychainHelper()

    var hfToken: String? {
        get { UserDefaults.standard.string(forKey: "hfTokenTemp") }
        set { UserDefaults.standard.set(newValue, forKey: "hfTokenTemp") }
    }
}
