import Foundation
import Combine

final class ServerProcessController: ObservableObject {
    enum State: Equatable {
        case stopped
        case starting
        case running
        case failed(message: String)
    }

    static let shared = ServerProcessController(configManager: ConfigManager())

    @Published private(set) var state: State = .stopped
    @Published private(set) var latestLogLine: String = ""

    private let configManager: ConfigManager
    private var process: Process?
    private var stdoutPipe: Pipe?
    private var stderrPipe: Pipe?
    private var cancellables = Set<AnyCancellable>()
    private let logManager = LogManager()

    init(configManager: ConfigManager) {
        self.configManager = configManager
    }

    func bootstrap() {
        configManager.load()
    }

    func startServer() {
        switch state {
        case .running, .starting:
            return
        case .failed(_), .stopped:
            break
        }
        do {
            let configuration = try configManager.buildRuntimeConfiguration()
            let jsonURL = try configManager.writeRuntimeConfig(configuration)
            try launchProcess(configPath: jsonURL)
        } catch {
            state = .failed(message: String(describing: error))
        }
    }

    func stopServer() {
        guard let process = process else { return }
        process.terminate()
        DispatchQueue.global().asyncAfter(deadline: .now() + 5) { [weak self] in
            guard let self = self, let process = self.process, process.isRunning else { return }
            process.interrupt()
            process.terminate()
            process.forceTerminate()
        }
    }

    func revealLogsInFinder() {
        logManager.revealLogsInFinder()
    }

    private func launchProcess(configPath: URL) throws {
        let pythonURL = try configManager.pythonExecutable()
        let process = Process()
        process.executableURL = pythonURL
        process.arguments = ["-m", "agentgrid.launcher.server", configPath.path]
        process.environment = configManager.environmentVariables()
        process.currentDirectoryURL = configManager.workingDirectory()

        stdoutPipe = Pipe()
        stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        self.process = process
        state = .starting

        stdoutPipe?.fileHandleForReading.readabilityHandler = { [weak self] handle in
            guard let line = String(data: handle.availableData, encoding: .utf8), !line.isEmpty else { return }
            self?.handleLog(line: line, stream: LogStream.stdout)
        }
        stderrPipe?.fileHandleForReading.readabilityHandler = { [weak self] handle in
            guard let line = String(data: handle.availableData, encoding: .utf8), !line.isEmpty else { return }
            self?.handleLog(line: line, stream: LogStream.stderr)
        }

        process.terminationHandler = { [weak self] proc in
            DispatchQueue.main.async { self?.processDidTerminate(proc) }
        }

        try process.run()

        DispatchQueue.main.asyncAfter(deadline: .now() + 2) { [weak self] in
            guard let self = self, let process = self.process, process.isRunning else { return }
            if case .starting = self.state {
                self.state = .running
            }
        }
    }

    private func handleLog(line: String, stream: LogStream) {
        latestLogLine = line
        logManager.append(line: line, stream: stream)
        if stream == .stderr, line.lowercased().contains("error") {
            DispatchQueue.main.async { [weak self] in
                self?.state = .failed(message: line.trimmingCharacters(in: .whitespacesAndNewlines))
            }
        } else if stream == .stdout, line.contains("Running a server on") {
            DispatchQueue.main.async { [weak self] in
                self?.state = .running
            }
        }
    }

    private func processDidTerminate(_ process: Process) {
        stdoutPipe?.fileHandleForReading.readabilityHandler = nil
        stderrPipe?.fileHandleForReading.readabilityHandler = nil
        stdoutPipe = nil
        stderrPipe = nil
        self.process = nil
        if case .failed = state {
            return
        }
        state = .stopped
    }
}

private extension Process {
    func forceTerminate() {
        guard isRunning else { return }
        let pid = self.processIdentifier
        kill(pid_t(pid), SIGKILL)
    }
}
