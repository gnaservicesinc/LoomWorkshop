import AppKit
import Darwin
import Foundation

private struct WorkshopDocument: Codable, Sendable {
    var schema = "loom-workshop.project.v1"
    var options = WorkshopOptions()
}
private struct AdapterManifest: Decodable, Sendable {
    struct Base: Decodable, Sendable { let id: String; let version: String }
    struct Model: Decodable, Sendable { let map: String; let format: String; let path: String; let weightsPath: String; let baseSHA256: String }
    struct Category: Decodable, Sendable { let id: String; let version: String; let adapters: [Model] }
    let schema: String; let id: String; let name: String; let version: String
    let baseModel: Base; let categories: [Category]; let files: [BaseManifest.File]
}
private struct ProcessResult: Sendable { let status: Int32; let lastError: String? }

public actor WorkshopProject {
    public nonisolated let sourceFolder: URL
    public nonisolated let projectFolder: URL
    public nonisolated let textureStudio: URL
    private var document: WorkshopDocument
    private var activeProcess: Process?
    private var operationActive = false

    public nonisolated static func inspectStudio(at application: URL) throws -> StudioBase { try readStudio(application).0 }

    public init(sourceFolder: URL, textureStudio: URL) throws {
        guard let path = realpath(sourceFolder.path, nil) else { throw WorkshopError("Choose an existing folder with your material images.") }
        defer { free(path) }
        let root = URL(fileURLWithPath: String(cString: path), isDirectory: true)
        guard (try root.resourceValues(forKeys: [.isDirectoryKey])).isDirectory == true else { throw WorkshopError("Choose a folder containing your material images.") }
        let studio = try readStudio(textureStudio).0
        self.sourceFolder = root
        self.projectFolder = root.appendingPathComponent(".loom-workshop", isDirectory: true)
        self.textureStudio = studio.applicationURL
        try FileManager.default.createDirectory(at: projectFolder, withIntermediateDirectories: true)
        let file = projectFolder.appendingPathComponent("project.json")
        if FileManager.default.fileExists(atPath: file.path) {
            document = try JSONDecoder().decode(WorkshopDocument.self, from: Data(contentsOf: file))
            guard document.schema == "loom-workshop.project.v1" else { throw WorkshopError("This source folder uses a newer Loom Workshop project format.") }
        } else {
            document = WorkshopDocument()
            try JSONEncoder().encode(document).write(to: file, options: .atomic)
        }
    }

    public func snapshot(imageSize: Int? = nil) throws -> WorkshopPlan {
        let studio = try readStudio(textureStudio).0
        let size = imageSize ?? document.options.imageSize
        guard (32...8192).contains(size) else { throw WorkshopError("Choose a training dimension between 32 and 8,192 pixels.") }
        let samples = try currentSamples(source: sourceFolder, studio: studio, imageSize: size)
        let datasets = studio.categories.compactMap { base -> WorkshopDataset? in
            let members = samples.filter { $0.categoryID == base.id }
            guard !members.isEmpty else { return nil }
            let counts = Dictionary(uniqueKeysWithValues: base.mapKinds.map { kind in (kind, members.filter { $0.targets[kind] != nil }.count) })
            return WorkshopDataset(id: base.id, name: base.name, imageCount: members.count, pairedCounts: counts)
        }
        return WorkshopPlan(sourceFolder: sourceFolder, studio: studio, datasets: datasets,
            requiresClassification: samples.contains { $0.categoryID == nil }, options: document.options, imageCount: samples.count)
    }

    public func updateOptions(_ options: WorkshopOptions) throws {
        try validate(options)
        document.options = options
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(document).write(to: projectFolder.appendingPathComponent("project.json"), options: .atomic)
    }

    public func latestAdapter() throws -> URL? {
        let runs = projectFolder.appendingPathComponent("runs", isDirectory: true)
        guard FileManager.default.fileExists(atPath: runs.path) else { return nil }
        let candidates = try FileManager.default.contentsOfDirectory(at: runs, includingPropertiesForKeys: [])
            .map { $0.appendingPathComponent("adapter", isDirectory: true) }
            .filter { FileManager.default.fileExists(atPath: $0.appendingPathComponent("manifest.json").path) }
            .sorted {
                ((try? $0.appendingPathComponent("manifest.json").resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast)
                > ((try? $1.appendingPathComponent("manifest.json").resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast)
            }
        for adapter in candidates where (try? validateAdapter(adapter)) != nil { return adapter }
        return nil
    }

    public func plan(progress: @escaping @Sendable (WorkshopEvent) -> Void) async throws -> WorkshopPlan {
        try beginOperation(); defer { operationActive = false }
        _ = try await run(command: "plan", options: document.options, progress: progress)
        return try snapshot()
    }

    public func train(options: WorkshopOptions, progress: @escaping @Sendable (WorkshopEvent) -> Void) async throws -> URL {
        try beginOperation(); defer { operationActive = false }
        try updateOptions(options)
        let run = try await run(command: "train", options: options, progress: progress)
        let adapter = run.appendingPathComponent("adapter", isDirectory: true)
        _ = try validateAdapter(adapter)
        progress(WorkshopEvent(message: "Your LoRA is ready for Texture Studio", progress: 1))
        return adapter
    }

    public func export(adapter: URL) async throws -> URL {
        try beginOperation(); defer { operationActive = false }
        return try await exportPackage(adapter)
    }

    public func install(adapter: URL) async throws {
        try beginOperation(); defer { operationActive = false }
        let archive = try await exportPackage(adapter)
        let directory = try await unpack(archive)
        defer { try? FileManager.default.removeItem(at: directory) }
        let manifest = try validateAdapter(directory)
        let receipt = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("Texture Studio/LoRAs", isDirectory: true).appendingPathComponent(manifest.id, isDirectory: true)
        if FileManager.default.fileExists(atPath: receipt.appendingPathComponent("manifest.json").path) {
            let installed = try validateAdapter(receipt)
            guard matchingReceipt(installed, expected: manifest) else { throw WorkshopError("A different LoRA occupies this installation folder.") }
            return
        }
        try await Self.openDocument(archive, application: textureStudio)
        for _ in 0..<80 {
            try Task.checkCancellation()
            if FileManager.default.fileExists(atPath: receipt.appendingPathComponent("manifest.json").path) {
                let installed = try validateAdapter(receipt)
                guard matchingReceipt(installed, expected: manifest) else { throw WorkshopError("Texture Studio reported a different LoRA installation.") }
                return
            }
            try await Task.sleep(for: .milliseconds(250))
        }
        throw WorkshopError("The LoRA was sent to Texture Studio, but installation has not completed. Check Texture Studio's LoRA manager, then try again.")
    }

    public func cancel() { if let activeProcess, activeProcess.isRunning { activeProcess.terminate() } }

    private func beginOperation() throws {
        guard !operationActive else { throw WorkshopError("An operation is already running in this source folder. Stop it before starting another.") }
        operationActive = true
    }
    private func matchingReceipt(_ actual: AdapterManifest, expected: AdapterManifest) -> Bool {
        actual.id == expected.id && actual.name == expected.name && actual.version == expected.version &&
        Dictionary(uniqueKeysWithValues: actual.files.map { ($0.path, $0.sha256) }) == Dictionary(uniqueKeysWithValues: expected.files.map { ($0.path, $0.sha256) })
    }
    private func validate(_ options: WorkshopOptions) throws {
        guard !options.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              (32...8192).contains(options.imageSize), (1...100_000).contains(options.epochs),
              (1...128).contains(options.rank), options.learningRate.isFinite, options.learningRate > 0,
              Set(options.datasetIDs).count == options.datasetIDs.count else { throw WorkshopError("Choose a name, valid training dimensions, positive epochs and learning rate for your LoRA.") }
    }
    private func run(command: String, options: WorkshopOptions, progress: @escaping @Sendable (WorkshopEvent) -> Void) async throws -> URL {
        try validate(options)
        let studio = try readStudio(textureStudio).0
        let samples = try currentSamples(source: sourceFolder, studio: studio, imageSize: options.imageSize)
        guard !samples.isEmpty else { throw WorkshopError("This folder has no diffuse images. Add diffuse images with matching material maps.") }
        guard Set(options.datasetIDs).isSubset(of: Set(studio.categories.map(\.id))) else { throw WorkshopError("A selected dataset is absent from this Texture Studio build.") }
        let run = projectFolder.appendingPathComponent("runs/" + UUID().uuidString.lowercased(), isDirectory: true)
        try FileManager.default.createDirectory(at: run, withIntermediateDirectories: true)
        let input: [[String: Any]] = samples.map { sample in
            var value: [String: Any] = ["id": sample.id, "input": sample.input.path, "targets": sample.targets.mapValues(\.path), "normal_convention": sample.normalConvention]
            if let category = sample.categoryID { value["category_id"] = category }
            return value
        }
        let request: [String: Any] = ["schema": "loom-workshop-training-v1", "source_root": sourceFolder.path,
            "output_directory": run.path, "texture_studio_path": textureStudio.path, "base_root": studio.packageURL.path,
            "base_id": studio.modelID, "base_version": studio.version, "adapter_name": options.name,
            "selected_category_ids": options.datasetIDs, "samples": input, "image_size": options.imageSize,
            "epochs": options.epochs, "batch_size": 1, "learning_rate": options.learningRate,
            "rank": options.rank, "alpha": options.rank, "shuffle_each_epoch": options.shuffleEachEpoch,
            "flip_crops": options.flipCrops, "rotate_crops": false, "exaggerate_features": options.exaggerateFeatures,
            "seed": 42, "device": "auto"]
        let requestFile = run.appendingPathComponent("request.json")
        try JSONSerialization.data(withJSONObject: request, options: [.prettyPrinted, .sortedKeys]).write(to: requestFile, options: .atomic)
        let runtime = try runtimeCode()
        let python = projectFolder.appendingPathComponent("runtime/bin/python")
        if !FileManager.default.isExecutableFile(atPath: python.path) {
            progress(.init(message: "Preparing the training runtime in this source folder"))
            try await execute(URL(fileURLWithPath: "/bin/bash"), arguments: [runtime.appendingPathComponent("setup_runtime.sh").path, sourceFolder.path],
                              variables: [:], log: run.appendingPathComponent("setup.log"), progress: progress)
        }
        try Task.checkCancellation()
        try await execute(python, arguments: ["-m", "loom_workshop", command, "--request", requestFile.path],
                          variables: ["PYTHONPATH": runtime.path], log: run.appendingPathComponent(command + ".log"), progress: progress)
        return run
    }

    private func runtimeCode() throws -> URL {
        let candidates = [ProcessInfo.processInfo.environment["LOOM_WORKSHOP_RUNTIME"].map { URL(fileURLWithPath: $0, isDirectory: true) },
            Bundle.main.resourceURL?.appendingPathComponent("Runtime", isDirectory: true),
            URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Runtime", isDirectory: true)].compactMap { $0 }
        guard let runtime = candidates.first(where: { FileManager.default.fileExists(atPath: $0.appendingPathComponent("loom_workshop/__main__.py").path) }) else {
            throw WorkshopError("Loom Workshop's training runtime is missing. Reinstall Loom Workshop.")
        }
        return runtime
    }

    private func execute(_ executable: URL, arguments: [String], variables: [String: String], log: URL,
                         progress: @escaping @Sendable (WorkshopEvent) -> Void) async throws {
        let child = Process()
        child.executableURL = executable; child.arguments = arguments; child.currentDirectoryURL = sourceFolder
        child.environment = ProcessInfo.processInfo.environment.merging(variables) { _, replacement in replacement }
        child.environment?["PYTHONUNBUFFERED"] = "1"; child.environment?["PYTHONDONTWRITEBYTECODE"] = "1"
        let output = Pipe(); child.standardOutput = output; child.standardError = output
        guard FileManager.default.createFile(atPath: log.path, contents: nil) else { throw WorkshopError("The project run log could not be created.") }
        let logger = try FileHandle(forWritingTo: log)
        activeProcess = child
        defer { activeProcess = nil; try? logger.close() }
        try child.run()
        let result = try await withTaskCancellationHandler {
            try await Task.detached {
                var buffer = Data(), lastError: String?
                while true {
                    let data = output.fileHandleForReading.availableData
                    if data.isEmpty { break }
                    try logger.write(contentsOf: data); buffer.append(data)
                    while let end = buffer.firstIndex(of: 10) {
                        let line = Data(buffer[..<end]); buffer.removeSubrange(...end)
                        guard let event = try? JSONSerialization.jsonObject(with: line) as? [String: Any] else { continue }
                        let type = event["event"] as? String ?? "status"
                        if let message = event["message"] as? String {
                            if type == "error" { lastError = message }
                            progress(WorkshopEvent(message: message, progress: (event["progress"] as? NSNumber)?.doubleValue,
                                epoch: event["epoch"] as? Int, loss: (event["loss"] as? NSNumber)?.doubleValue))
                        }
                    }
                }
                child.waitUntilExit()
                return ProcessResult(status: child.terminationStatus, lastError: lastError)
            }.value
        } onCancel: { if child.isRunning { child.terminate() } }
        try Task.checkCancellation()
        if result.status == 130 { throw CancellationError() }
        guard result.status == 0 else { throw WorkshopError((result.lastError ?? "The operation could not finish. Your original images and installed base are preserved.") + "\nRun details: " + log.path) }
    }

    private func validateAdapter(_ root: URL) throws -> AdapterManifest {
        let (_, base) = try readStudio(textureStudio)
        let manifest = try JSONDecoder().decode(AdapterManifest.self, from: Data(contentsOf: root.appendingPathComponent("manifest.json")))
        guard manifest.schema == "texture-loom.adapter.v1", UUID(uuidString: manifest.id) != nil,
              !manifest.name.isEmpty, !manifest.categories.isEmpty,
              manifest.baseModel.id == base.id, manifest.baseModel.version == base.version,
              Set(manifest.categories.map(\.id)).count == manifest.categories.count else { throw WorkshopError("This LoRA belongs to a different Texture Loom model or version.") }
        try verifyFiles(manifest.files, root: root)
        let inventory = Set(manifest.files.map(\.path))
        for category in manifest.categories {
            guard let installed = base.categories.first(where: { $0.id == category.id && $0.version == category.version }),
                  !category.adapters.isEmpty, Set(category.adapters.map(\.map)).count == category.adapters.count else { throw WorkshopError("A LoRA dataset is absent from this Texture Studio version.") }
            for adapter in category.adapters {
                guard adapter.format == "coreml", let model = installed.models.first(where: { $0.map == adapter.map }),
                      try fileSHA(modelPath(model.weightsPath, in: textureStudio.appendingPathComponent("Contents/Resources/TextureLoom"))) == adapter.baseSHA256 else { throw WorkshopError("This LoRA needs its exact base model weights.") }
                for path in [adapter.path, adapter.weightsPath] {
                    let artifact = try modelPath(path, in: root)
                    guard FileManager.default.fileExists(atPath: artifact.path), inventory.contains(where: { $0 == path || $0.hasPrefix(path + "/") }) else { throw WorkshopError("A LoRA artifact is missing: \(path).") }
                }
            }
        }
        return manifest
    }

    private func exportPackage(_ adapter: URL) async throws -> URL {
        if (try? adapter.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory != true {
            let directory = try await unpack(adapter)
            defer { try? FileManager.default.removeItem(at: directory) }
            _ = try validateAdapter(directory)
            return adapter
        }
        let ownedRuns = projectFolder.appendingPathComponent("runs", isDirectory: true).standardizedFileURL.resolvingSymlinksInPath().path + "/"
        guard adapter.standardizedFileURL.resolvingSymlinksInPath().path.hasPrefix(ownedRuns) else {
            throw WorkshopError("This LoRA belongs to another source project. Reopen its source folder to export or install it.")
        }
        let manifest = try validateAdapter(adapter)
        let exports = projectFolder.appendingPathComponent("exports", isDirectory: true)
        try FileManager.default.createDirectory(at: exports, withIntermediateDirectories: true)
        let safeName = manifest.name.replacingOccurrences(of: "[^A-Za-z0-9 ._-]", with: "", options: .regularExpression).trimmingCharacters(in: .whitespacesAndNewlines)
        let archive = exports.appendingPathComponent((safeName.isEmpty ? "Texture LoRA" : safeName) + "-" + manifest.id.prefix(8) + ".loomlora")
        let staging = exports.appendingPathComponent("." + UUID().uuidString + ".zip")
        defer { try? FileManager.default.removeItem(at: staging) }
        try await execute(URL(fileURLWithPath: "/usr/bin/ditto"), arguments: ["-c", "-k", "--norsrc", adapter.path, staging.path], variables: [:], log: exports.appendingPathComponent("export.log"), progress: { _ in })
        if FileManager.default.fileExists(atPath: archive.path) { try FileManager.default.removeItem(at: archive) }
        try FileManager.default.moveItem(at: staging, to: archive)
        return archive
    }

    private func unpack(_ archive: URL) async throws -> URL {
        let destination = projectFolder.appendingPathComponent("imports/" + UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        let listing = try await Self.capture("/usr/bin/zipinfo", ["-1", archive.path])
        let permissions = try await Self.capture("/usr/bin/zipinfo", ["-l", archive.path])
        let names = listing.split(separator: "\n").map(String.init)
        guard !names.isEmpty, names.count <= 10_000, !permissions.split(separator: "\n").contains(where: { $0.hasPrefix("l") }) else { throw WorkshopError("The LoRA archive has an invalid file inventory.") }
        for name in names {
            guard !name.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) }) else { throw WorkshopError("The LoRA archive has an invalid filename.") }
            _ = try modelPath(name.hasSuffix("/") ? String(name.dropLast()) : name, in: destination)
        }
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: false)
        do {
            try await execute(URL(fileURLWithPath: "/usr/bin/ditto"), arguments: ["-x", "-k", archive.path, destination.path], variables: [:], log: projectFolder.appendingPathComponent("imports/import.log"), progress: { _ in })
            return destination
        } catch { try? FileManager.default.removeItem(at: destination); throw error }
    }
    private nonisolated static func capture(_ executable: String, _ arguments: [String]) async throws -> String {
        try await Task.detached {
            let child = Process(); child.executableURL = URL(fileURLWithPath: executable); child.arguments = arguments
            let pipe = Pipe(); child.standardOutput = pipe; child.standardError = pipe
            try child.run(); let data = pipe.fileHandleForReading.readDataToEndOfFile(); child.waitUntilExit()
            guard child.terminationStatus == 0 else { throw WorkshopError("The LoRA archive could not be opened.") }
            return String(decoding: data, as: UTF8.self)
        }.value
    }
    @MainActor private static func openDocument(_ url: URL, application: URL) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            let configuration = NSWorkspace.OpenConfiguration(); configuration.activates = true
            NSWorkspace.shared.open([url], withApplicationAt: application, configuration: configuration) { _, error in
                if let error { continuation.resume(throwing: error) } else { continuation.resume() }
            }
        }
    }
}
