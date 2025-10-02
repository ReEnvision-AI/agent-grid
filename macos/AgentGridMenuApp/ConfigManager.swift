import Foundation
import AppKit

enum ConfigError: Error {
    case pythonNotFound
    case invalidModelsFile
    case unableToWriteRuntimeConfig
}

final class ConfigManager {
    struct RuntimeConfiguration: Codable {
        var converted_model_name_or_path: String
        var torch_dtype: String
        var device: String
        var port: Int
        var token: String
        var identity_path: String
        var warmup_tokens_interval: Int?
        var new_swarm: Bool
    }

    private enum Keys {
        static let pythonPath = "pythonPath"
        static let modelsPath = "modelsPath"
        static let envPath = "envPath"
        static let selectedModel = "selectedModel"
        static let warmupInterval = "warmupInterval"
    }

    private let defaults = UserDefaults.standard
    private let fileManager = FileManager.default

    func load() {
        defaults.register(defaults: [
            Keys.pythonPath: "/usr/bin/python3",
            Keys.modelsPath: "models",
            Keys.envPath: ".env",
            Keys.warmupInterval: 0
        ])
    }

    func pythonExecutable() throws -> URL {
        let path = defaults.string(forKey: Keys.pythonPath) ?? ""
        let url = URL(fileURLWithPath: path)
        if fileManager.isExecutableFile(atPath: url.path) {
            return url
        }
        throw ConfigError.pythonNotFound
    }

    func modelsFileURL() -> URL {
        URL(fileURLWithPath: defaults.string(forKey: Keys.modelsPath) ?? "models")
    }

    func environmentFileURL() -> URL {
        URL(fileURLWithPath: defaults.string(forKey: Keys.envPath) ?? ".env")
    }

    func workingDirectory() -> URL {
        URL(fileURLWithPath: fileManager.currentDirectoryPath)
    }

    func environmentVariables() -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        if let envPath = defaults.string(forKey: Keys.envPath) {
            env["AGENTGRID_ENV_PATH"] = envPath
        }
        if let token = KeychainHelper.shared.hfToken { env["HF_TOKEN"] = token }
        return env
    }

    func buildRuntimeConfiguration() throws -> RuntimeConfiguration {
        guard let model = defaults.string(forKey: Keys.selectedModel), !model.isEmpty else {
            throw ConfigError.invalidModelsFile
        }
        let token = KeychainHelper.shared.hfToken ?? ""
        return RuntimeConfiguration(
            converted_model_name_or_path: model,
            torch_dtype: "float16",
            device: "mps",
            port: 31331,
            token: token,
            identity_path: "./dev.id",
            warmup_tokens_interval: warmupInterval(),
            new_swarm: true
        )
    }

    func warmupInterval() -> Int? {
        let interval = defaults.integer(forKey: Keys.warmupInterval)
        return interval > 0 ? interval : nil
    }

    func writeRuntimeConfig(_ config: RuntimeConfiguration) throws -> URL {
        let appSupport = try fileManager.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let dir = appSupport.appendingPathComponent("AgentGrid", isDirectory: true)
        try fileManager.createDirectory(at: dir, withIntermediateDirectories: true)
        let fileURL = dir.appendingPathComponent("runtime-config.json")
        let data = try JSONEncoder().encode(config)
        guard let jsonString = String(data: data, encoding: .utf8) else {
            throw ConfigError.unableToWriteRuntimeConfig
        }
        try jsonString.write(to: fileURL, atomically: true, encoding: .utf8)
        return fileURL
    }

    func update(selectedModel: String) {
        defaults.set(selectedModel, forKey: Keys.selectedModel)
    }

    func update(pythonPath: String) {
        defaults.set(pythonPath, forKey: Keys.pythonPath)
    }

    func update(modelsPath: String) {
        defaults.set(modelsPath, forKey: Keys.modelsPath)
    }

    func update(envPath: String) {
        defaults.set(envPath, forKey: Keys.envPath)
    }

    func update(warmupInterval: Int) {
        defaults.set(warmupInterval, forKey: Keys.warmupInterval)
    }
}

final class KeychainHelper {
    static let shared = KeychainHelper()

    private let service = "com.agentgrid.menuapp"
    private let account = "huggingface_token"

    var hfToken: String? {
        get { retrieveToken() }
        set { if let token = newValue { storeToken(token) } }
    }

    private func storeToken(_ token: String) {
        // TODO: Implement Keychain storage.
        // Placeholder uses UserDefaults for now.
        UserDefaults.standard.set(token, forKey: "hfTokenTemp")
    }

    private func retrieveToken() -> String? {
        UserDefaults.standard.string(forKey: "hfTokenTemp")
    }
}
