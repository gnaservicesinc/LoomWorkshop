import AppKit
import Foundation
import Observation
import UniformTypeIdentifiers
import WorkshopCore

@MainActor @Observable
final class WorkshopModel {
    private(set) var studio: StudioBase?
    private(set) var studioURL = URL(fileURLWithPath: "/Applications/Texture Studio.app", isDirectory: true)
    private(set) var studioIssue: String?
    private(set) var sourceFolder: URL?
    private(set) var folderIssue: String?
    private(set) var plan: WorkshopPlan?
    var options = WorkshopOptions()
    var selectedDatasetID = ""
    private(set) var busy = false
    private(set) var status = ""
    private(set) var progress: Double?
    private(set) var events: [WorkshopEvent] = []
    private(set) var trainedAdapter: URL?
    private(set) var trainedName = ""
    private(set) var exportedAdapter: URL?
    private(set) var installed = false
    var errorMessage: String?
    @ObservationIgnored private var project: WorkshopProject?
    @ObservationIgnored private var operation: Task<Void, Never>?
    @ObservationIgnored private var saveOperation: Task<Void, Never>?
    @ObservationIgnored private var generation = UUID()
    @ObservationIgnored private var polling = false
    @ObservationIgnored private let rememberLocations: Bool

    init(rememberLocations: Bool = true) { self.rememberLocations = rememberLocations }

    var availableDatasets: [WorkshopDataset] {
        plan?.datasets.filter { $0.pairedCounts.values.contains { $0 > 0 } } ?? []
    }
    var selectedDatasetIDs: [String] {
        selectedDatasetID.isEmpty ? availableDatasets.map(\.id) : [selectedDatasetID]
    }
    var trainingOptionsIssue: String? {
        do { try options.validate(); return nil }
        catch { return error.localizedDescription }
    }
    var canTrain: Bool {
        studio != nil && plan != nil && !busy && plan?.requiresClassification == false &&
        plan?.options.imageSize == options.imageSize && trainingOptionsIssue == nil &&
        !selectedDatasetIDs.isEmpty && (selectedDatasetID.isEmpty || availableDatasets.contains { $0.id == selectedDatasetID }) &&
        !options.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    func start() {
        guard !busy, studio == nil else { return }
        let arguments = ProcessInfo.processInfo.arguments
        let explicitStudio = argument("--texture-studio", in: arguments).map { URL(fileURLWithPath: $0, isDirectory: true) }
        let remembered = rememberLocations ? UserDefaults.standard.string(forKey: "TextureStudioApplication") : nil
        let url = explicitStudio ?? remembered.map { URL(fileURLWithPath: $0, isDirectory: true) } ?? studioURL
        let explicitSource = argument("--source-folder", in: arguments).map { URL(fileURLWithPath: $0, isDirectory: true) }
        let lastSource = rememberLocations ? UserDefaults.standard.string(forKey: "SourceImageFolder") : nil
        let rememberedSource = lastSource.map { URL(fileURLWithPath: $0, isDirectory: true) }
            .flatMap { FileManager.default.fileExists(atPath: $0.path) ? $0 : nil }
        let source = explicitSource ?? rememberedSource
        inspectStudio(url, thenOpen: source)
    }

    func chooseStudio() {
        let panel = NSOpenPanel()
        panel.title = "Choose Texture Studio"
        panel.prompt = "Choose Texture Studio"
        panel.allowedContentTypes = [.applicationBundle]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.directoryURL = URL(fileURLWithPath: "/Applications", isDirectory: true)
        guard panel.runModal() == .OK, let url = panel.url else { return }
        inspectStudio(url)
    }

    func inspectStudio(_ url: URL, thenOpen source: URL? = nil) {
        operation?.cancel()
        saveOperation?.cancel()
        let oldProject = project
        generation = UUID()
        let current = generation
        studioURL = url
        studio = nil
        studioIssue = nil
        clearProject()
        busy = true
        status = "Checking Texture Studio"
        progress = nil
        operation = Task {
            if let oldProject { await oldProject.cancel() }
            do {
                let base = try await Task.detached(priority: .userInitiated) {
                    try WorkshopProject.inspectStudio(at: url)
                }.value
                guard generation == current, !Task.isCancelled else { return }
                studio = base
                studioURL = base.applicationURL
                busy = false
                status = "Texture Studio is ready"
                if rememberLocations { UserDefaults.standard.set(base.applicationURL.path, forKey: "TextureStudioApplication") }
                if let source { openSourceFolder(source) }
            } catch {
                guard generation == current else { return }
                busy = false
                status = "Texture Studio needs a built-in Texture Loom model"
                studioIssue = error.localizedDescription
            }
        }
    }

    func chooseSourceFolder() {
        guard studio != nil, !busy else { return }
        let panel = NSOpenPanel()
        panel.title = "Choose your material image folder"
        panel.prompt = "Choose Image Folder"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.directoryURL = sourceFolder
        guard panel.runModal() == .OK, let url = panel.url else { return }
        openSourceFolder(url)
    }

    func openSourceFolder(_ url: URL) {
        guard let studio else { return }
        operation?.cancel()
        saveOperation?.cancel()
        let oldProject = project
        generation = UUID()
        let current = generation
        clearProject()
        sourceFolder = url
        busy = true
        status = "Reading your images"
        progress = nil
        operation = Task {
            if let oldProject { await oldProject.cancel() }
            do {
                let opened = try WorkshopProject(sourceFolder: url, textureStudio: studio.applicationURL)
                let value = try await opened.snapshot()
                let adapter = try await opened.latestAdapter()
                guard current == generation, !Task.isCancelled else { return }
                project = opened
                options = value.options
                if options.datasetIDs.count == 1 { selectedDatasetID = options.datasetIDs[0] }
                adopt(value)
                if let adapter {
                    trainedAdapter = adapter
                    let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: adapter.appendingPathComponent("manifest.json"))) as? [String: Any]
                    trainedName = manifest?["name"] as? String ?? "Texture LoRA"
                }
                busy = false
                if rememberLocations { UserDefaults.standard.set(value.sourceFolder.path, forKey: "SourceImageFolder") }
                if value.requiresClassification { analyze() }
                else { status = "Images ready" }
            } catch {
                guard current == generation else { return }
                fail(error)
            }
        }
    }

    func analyze() {
        let current = generation
        let onProgress = eventHandler()
        perform("Categorizing your images") { [weak self] project in
            let value = try await project.plan(progress: onProgress)
            guard let self, self.generation == current else { return }
            self.adopt(value)
        }
    }

    func train() {
        guard canTrain else { return }
        var value = options
        value.datasetIDs = selectedDatasetID.isEmpty ? [] : [selectedDatasetID]
        let current = generation
        let onProgress = eventHandler()
        perform("Training your LoRA", clearEvents: true) { [weak self] project in
            try await project.updateOptions(value)
            let adapter = try await project.train(options: value, progress: onProgress)
            guard let self, self.generation == current else { return }
            self.trainedAdapter = adapter
            self.trainedName = value.name
            self.exportedAdapter = nil
            self.installed = false
        }
    }

    func export() {
        guard let trainedAdapter, !busy else { return }
        let current = generation
        perform("Exporting your LoRA") { [weak self] project in
            let package = try await project.export(adapter: trainedAdapter)
            guard let self, self.generation == current else { return }
            self.exportedAdapter = package
            NSWorkspace.shared.activateFileViewerSelecting([package])
        }
    }

    func install() {
        guard let adapter = exportedAdapter ?? trainedAdapter, !busy else { return }
        let current = generation
        perform("Installing into Texture Studio") { [weak self] project in
            try await project.install(adapter: adapter)
            guard let self, self.generation == current else { return }
            self.installed = true
        }
    }

    func cancel() {
        guard let project, busy else { return }
        operation?.cancel()
        status = "Stopping safely…"
        Task { await project.cancel() }
    }

    func pollFolder() async {
        guard let project, !busy, !polling else { return }
        polling = true
        defer { polling = false }
        let current = generation
        let imageSize = options.imageSize
        do {
            let value = try await project.snapshot(imageSize: imageSize)
            guard current == generation, !busy, options.imageSize == imageSize else { return }
            let recovered = folderIssue != nil
            folderIssue = nil
            adopt(value)
            if recovered { status = value.requiresClassification ? "Images need categorizing" : "Images ready" }
        } catch {
            guard current == generation, !busy else { return }
            plan = nil
            folderIssue = error.localizedDescription
            status = "Dataset needs attention"
        }
    }

    func scheduleOptionsSave() {
        guard let project, !busy else { return }
        saveOperation?.cancel()
        guard !options.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        var value = options
        value.datasetIDs = selectedDatasetID.isEmpty ? [] : [selectedDatasetID]
        let current = generation
        saveOperation = Task {
            do {
                try await Task.sleep(for: .milliseconds(300))
                try Task.checkCancellation()
                try await project.updateOptions(value)
                let updated = try await project.snapshot(imageSize: value.imageSize)
                guard current == generation, !busy, options.imageSize == value.imageSize else { return }
                adopt(updated)
            } catch is CancellationError {} catch {
                guard current == generation else { return }
                errorMessage = error.localizedDescription
            }
        }
    }

    func revealSource() { if let sourceFolder { NSWorkspace.shared.open(sourceFolder) } }

    func prepareForTermination() async -> Bool {
        guard !busy else { return false }
        saveOperation?.cancel()
        guard let project, !options.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return true }
        var value = options
        value.datasetIDs = selectedDatasetID.isEmpty ? [] : [selectedDatasetID]
        let current = generation
        do {
            try await project.updateOptions(value)
            return current == generation && !busy
        } catch {
            errorMessage = error.localizedDescription
            return false
        }
    }
    func openStudio() { if let studio { NSWorkspace.shared.open(studio.applicationURL) } }

    private func perform(_ title: String, clearEvents: Bool = false,
                         action: @escaping @MainActor (WorkshopProject) async throws -> Void) {
        guard let project, !busy else { return }
        saveOperation?.cancel()
        let current = generation
        busy = true
        status = title
        progress = nil
        errorMessage = nil
        if clearEvents { events = [] }
        operation = Task {
            do {
                try Task.checkCancellation()
                try await action(project)
                let value = try await project.snapshot(imageSize: options.imageSize)
                guard current == generation else { return }
                adopt(value)
                busy = false
                status = Task.isCancelled ? "Stopped" : "\(title) complete"
                progress = Task.isCancelled ? nil : 1
            } catch {
                guard current == generation else { return }
                if Task.isCancelled || error is CancellationError {
                    busy = false
                    status = "Stopped"
                    progress = nil
                    if let value = try? await project.snapshot(imageSize: options.imageSize) { adopt(value) }
                } else { fail(error) }
            }
        }
    }

    private func eventHandler() -> @Sendable (WorkshopEvent) -> Void {
        let current = generation
        return { [weak self] event in
            Task { @MainActor [weak self] in
                guard let self, self.generation == current else { return }
                self.events.append(event)
                if self.events.count > 200 { self.events.removeFirst(self.events.count - 200) }
                self.status = event.message
                if let progress = event.progress { self.progress = progress }
            }
        }
    }

    private func adopt(_ value: WorkshopPlan) {
        plan = value
        sourceFolder = value.sourceFolder
        if !selectedDatasetID.isEmpty, !value.datasets.contains(where: { $0.id == selectedDatasetID }) { selectedDatasetID = "" }
    }

    private func clearProject() {
        project = nil; sourceFolder = nil; folderIssue = nil; plan = nil
        selectedDatasetID = ""; options = WorkshopOptions()
        trainedAdapter = nil; trainedName = ""; exportedAdapter = nil; installed = false
        events = []; errorMessage = nil
    }

    private func fail(_ error: Error) {
        busy = false; progress = nil; status = "Action could not finish"
        errorMessage = error.localizedDescription
    }

    private func argument(_ name: String, in arguments: [String]) -> String? {
        guard let index = arguments.firstIndex(of: name), arguments.indices.contains(index + 1) else { return nil }
        return arguments[index + 1]
    }
}
