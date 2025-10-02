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

    @Published private(set) var models: [String] = ConfigManager.defaultModels
    @Published private(set) var devices: [String: DeviceInfo] = ModelDeviceDiscovery.defaultDevices()

    func refreshModels(modelsFile: URL, python: URL) {
        refreshModels()
    }

    func refreshModels() {
        models = ConfigManager.defaultModels
    }

    func refreshDevices(python: URL) {
        refreshDevices()
    }

    func refreshDevices() {
        devices = ModelDeviceDiscovery.defaultDevices()
    }

    private static func defaultDevices() -> [String: DeviceInfo] {
        return [
            "cpu": DeviceInfo(available: true, deviceCount: 1, devices: nil),
            "mps": DeviceInfo(available: true, deviceCount: 1, devices: nil),
            "cuda": DeviceInfo(available: false, deviceCount: 0, devices: nil)
        ]
    }
}
