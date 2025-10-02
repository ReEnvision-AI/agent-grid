import Foundation
import Combine

struct DeviceInfo: Codable {
    let available: Bool
    let deviceCount: Int?
    let devices: [CUDADevice]?

    struct CUDADevice: Codable {
        let id: Int
        let name: String
    }
}

final class ModelDeviceDiscovery: ObservableObject {
    static let shared = ModelDeviceDiscovery()

    @Published private(set) var models: [String] = []
    @Published private(set) var devices: [String: DeviceInfo] = [:]

    private let queue = DispatchQueue(label: "com.agentgrid.discovery")

    func refreshModels(modelsFile: URL, python: URL) {
        queue.async {
            let task = Process()
            task.executableURL = python
            task.arguments = ["-m", "agentgrid.launcher.discovery", "--list-models", "--models-file", modelsFile.path]
            let pipe = Pipe()
            task.standardOutput = pipe
            do {
                try task.run()
                task.waitUntilExit()
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                let output = String(decoding: data, as: UTF8.self)
                let lines = output.split(separator: "\n").map { String($0) }
                DispatchQueue.main.async { [weak self] in
                    self?.models = lines
                }
            } catch {
                // Ignore for now
            }
        }
    }

    func refreshDevices(python: URL) {
        queue.async {
            let task = Process()
            task.executableURL = python
            task.arguments = ["-m", "agentgrid.launcher.discovery", "--probe-devices"]
            let pipe = Pipe()
            task.standardOutput = pipe
            do {
                try task.run()
                task.waitUntilExit()
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                let decoded = try JSONDecoder().decode([String: DeviceInfo].self, from: data)
                DispatchQueue.main.async { [weak self] in
                    self?.devices = decoded
                }
            } catch {
                // Ignore for now; fallback to defaults
            }
        }
    }
}
