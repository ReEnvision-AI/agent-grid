import Foundation
import AppKit

enum LogStream: String {
    case stdout
    case stderr
}

final class LogManager {
    private let fileManager = FileManager.default
    private lazy var logsDirectory: URL = {
        let appSupport = try? fileManager.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let dir = appSupport?.appendingPathComponent("AgentGrid/Logs", isDirectory: true)
        if let dir = dir {
            try? fileManager.createDirectory(at: dir, withIntermediateDirectories: true)
            return dir
        }
        return URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("AgentGridLogs", isDirectory: true)
    }()

    private lazy var stdoutURL = logsDirectory.appendingPathComponent("stdout.log")
    private lazy var stderrURL = logsDirectory.appendingPathComponent("stderr.log")

    func append(line: String, stream: LogStream) {
        let url = stream == .stdout ? stdoutURL : stderrURL
        let data = (line + "\n").data(using: .utf8) ?? Data()
        if fileManager.fileExists(atPath: url.path) {
            if let handle = try? FileHandle(forWritingTo: url) {
                handle.seekToEndOfFile()
                handle.write(data)
                try? handle.close()
            }
        } else {
            try? data.write(to: url)
        }
    }

    func revealLogsInFinder() {
        NSWorkspace.shared.activateFileViewerSelecting([stdoutURL, stderrURL])
    }
}
