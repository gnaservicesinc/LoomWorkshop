import AppKit
import CryptoKit
import Darwin
import ImageIO
import SwiftUI
import Testing
import UniformTypeIdentifiers
@testable import LoomWorkshop
@testable import WorkshopCore

@Suite(.serialized)
struct WorkshopViewTests {
    @Test @MainActor
    func advancedControlsRenderAndRejectInvalidTrainingSettings() async throws {
        let fixture = try await WorkshopViewFixture()
        defer { fixture.remove() }
        let model = WorkshopModel(rememberLocations: false)
        try await open(model, fixture)
        let profile = try #require(model.studio?.categories.first?.modelProfiles?.first)
        #expect(profile.baseParameters == 42_553)
        model.options.optimizer = .lion
        model.options.beta2 = 0.99
        model.options.batchSize = 2
        model.options.gradientAccumulationSteps = 4
        model.options.cropRefresh = .run
        model.options.augmentationRefresh = .epoch
        #expect(model.options.effectiveBatchSize == 8)
        #expect(model.canTrain)
        for size in [NSSize(width: 1000, height: 720), NSSize(width: 1180, height: 940), NSSize(width: 1000, height: 2200)] {
            try await render(model, size: size, name: "loom-workshop-\(Int(size.width))x\(Int(size.height))-advanced.png", advanced: true)
        }
        model.options.gradientAccumulationSteps = 0
        #expect(!model.canTrain)
        model.options.gradientAccumulationSteps = 4
        model.options.beta2 = 1
        #expect(!model.canTrain)
        model.options.epochs = Int.min
        model.options.rank = Int.max
        #expect(!model.canTrain)
        try await render(model, size: NSSize(width: 1000, height: 720), name: "loom-workshop-invalid-options.png", advanced: true)
    }
    @Test @MainActor
    func nativeWorkflowRendersAtMinimumAndDefaultSizes() async throws {
        let fixture = try await WorkshopViewFixture()
        defer { fixture.remove() }
        let model = WorkshopModel(rememberLocations: false)
        try await open(model, fixture)
        #expect(model.plan?.imageCount == 1)
        #expect(model.availableDatasets.count == 1)
        #expect(model.canTrain)
        for size in [NSSize(width: 1000, height: 720), NSSize(width: 1180, height: 940)] {
            try await render(model, size: size, name: "loom-workshop-\(Int(size.width))x\(Int(size.height))-project.png")
        }
        let start = WorkshopModel(rememberLocations: false)
        try await render(start, size: NSSize(width: 1180, height: 940), name: "loom-workshop-start.png")
    }

    @Test @MainActor
    func anotherSourceFolderImmediatelyClearsSelectionAndRestoresItsOwnOptions() async throws {
        let fixture = try await WorkshopViewFixture()
        defer { fixture.remove() }
        let second = fixture.root.appendingPathComponent("Second Images", isDirectory: true)
        try FileManager.default.createDirectory(at: second, withIntermediateDirectories: true)
        let secondProject = try WorkshopProject(sourceFolder: second, textureStudio: fixture.application)
        var options = WorkshopOptions()
        options.name = "Another LoRA"
        options.imageSize = 1024
        options.shuffleEachEpoch = false
        try await secondProject.updateOptions(options)
        let model = WorkshopModel(rememberLocations: false)
        try await open(model, fixture)
        model.selectedDatasetID = try #require(model.availableDatasets.first?.id)
        model.openSourceFolder(second)
        #expect(model.plan == nil)
        #expect(model.selectedDatasetID.isEmpty)
        #expect(model.trainedAdapter == nil)
        #expect(!model.canTrain)
        try await finish(model)
        #expect(model.options.name == "Another LoRA")
        #expect(model.options.imageSize == 1024)
        #expect(!model.options.shuffleEachEpoch)
        #expect(model.availableDatasets.isEmpty)
        model.openSourceFolder(fixture.source)
        try await finish(model)
        #expect(model.options.name == "Brick Detail")
        #expect(model.options.imageSize == 256)
        #expect(model.canTrain)
    }

    @Test @MainActor
    func removingATargetRemovesItsTrainingPairAndChangingBaseClearsTheProject() async throws {
        let fixture = try await WorkshopViewFixture()
        defer { fixture.remove() }
        let model = WorkshopModel(rememberLocations: false)
        try await open(model, fixture)
        #expect(model.availableDatasets.first?.pairedCounts["depth"] == 1)
        try FileManager.default.removeItem(at: fixture.source.appendingPathComponent("brick_depth.png"))
        await model.pollFolder()
        #expect(model.availableDatasets.isEmpty)
        #expect(!model.canTrain)
        model.inspectStudio(fixture.root.appendingPathComponent("Missing Texture Studio.app"))
        #expect(model.plan == nil)
        #expect(model.sourceFolder == nil)
        #expect(model.studio == nil)
        try await finish(model, expectsProject: false)
        #expect(model.studioIssue != nil)
        #expect(!model.canTrain)
    }

    @Test @MainActor
    func reopeningAFolderRestoresItsCompatibleCompletedLoRA() async throws {
        let fixture = try await WorkshopViewFixture()
        defer { fixture.remove() }
        let adapter = try fixture.addAdapter(named: "Saved Brick Detail")
        let model = WorkshopModel(rememberLocations: false)
        try await open(model, fixture)
        #expect(model.trainedAdapter?.resolvingSymlinksInPath() == adapter.resolvingSymlinksInPath())
        #expect(model.trainedName == "Saved Brick Detail")
        #expect(model.options.name == "Brick Detail")
        model.options.name = "Next Training Run"
        #expect(model.trainedName == "Saved Brick Detail")
        let other = fixture.root.appendingPathComponent("Another Images Folder", isDirectory: true)
        try FileManager.default.createDirectory(at: other, withIntermediateDirectories: true)
        model.openSourceFolder(other)
        #expect(model.trainedAdapter == nil)
        #expect(model.trainedName.isEmpty)
        try await finish(model)
        #expect(model.trainedAdapter == nil)
    }

    @Test @MainActor
    func changingImageSizeCannotStartWithThePreviousPairCount() async throws {
        let fixture = try await WorkshopViewFixture()
        defer { fixture.remove() }
        let model = WorkshopModel(rememberLocations: false)
        try await open(model, fixture)
        #expect(model.canTrain)
        model.options.imageSize = 2048
        #expect(!model.canTrain)
        model.scheduleOptionsSave()
        for _ in 0..<100 {
            if model.plan?.options.imageSize == 2048 { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(model.errorMessage == nil)
        #expect(model.plan?.options.imageSize == 2048)
        #expect(model.availableDatasets.isEmpty)
        #expect(!model.canTrain)
    }

    @Test @MainActor
    func unavailableCurrentTagClearsTheTrainingPlanAndRecoversWithoutRepeatedAlerts() async throws {
        let fixture = try await WorkshopViewFixture()
        defer { fixture.remove() }
        let adapter = try fixture.addAdapter(named: "Saved Brick Detail")
        let model = WorkshopModel(rememberLocations: false)
        try await open(model, fixture)
        #expect(model.canTrain)
        #expect(model.availableDatasets.first?.pairedCounts["depth"] == 1)
        let input = fixture.source.appendingPathComponent("brick_diffuse.png")
        let key = "com.gnaservicesinc.loomfactory.image.v1"
        let data = try JSONSerialization.data(withJSONObject: [
            "id": UUID().uuidString, "pairingKey": "brick-family", "role": "diffuse",
            "category": "Absent Material", "verified": false
        ])
        let written = data.withUnsafeBytes { setxattr(input.path, key, $0.baseAddress, data.count, 0, 0) }
        #expect(written == 0)
        await model.pollFolder()
        #expect(model.plan == nil)
        #expect(model.availableDatasets.isEmpty)
        #expect(!model.canTrain)
        #expect(model.folderIssue?.contains("Absent Material") == true)
        #expect(model.errorMessage == nil)
        #expect(model.trainedAdapter?.resolvingSymlinksInPath() == adapter.resolvingSymlinksInPath())
        await model.pollFolder()
        #expect(model.errorMessage == nil)
        #expect(model.plan == nil)
        #expect(removexattr(input.path, key, 0) == 0)
        await model.pollFolder()
        #expect(model.folderIssue == nil)
        #expect(model.availableDatasets.first?.pairedCounts["depth"] == 1)
        #expect(model.canTrain)
        #expect(model.errorMessage == nil)
        #expect(model.trainedName == "Saved Brick Detail")
    }

    @MainActor private func open(_ model: WorkshopModel, _ fixture: WorkshopViewFixture) async throws {
        model.inspectStudio(fixture.application)
        try await finish(model, expectsProject: false)
        _ = try #require(model.studio)
        model.openSourceFolder(fixture.source)
        try await finish(model)
    }

    @MainActor private func finish(_ model: WorkshopModel, expectsProject: Bool = true) async throws {
        for _ in 0..<200 {
            if !model.busy { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(!model.busy)
        #expect(model.errorMessage == nil)
        if expectsProject { _ = try #require(model.plan) }
    }

    @MainActor private func render(_ model: WorkshopModel, size: NSSize, name: String, advanced: Bool = false) async throws {
        _ = NSApplication.shared
        let content = NSHostingView(rootView: WorkshopView(model: model, startsAutomatically: false, showsAdvancedOptionsInitially: advanced).tint(WorkshopStyle.accent))
        let window = NSWindow(contentRect: NSRect(origin: .zero, size: size), styleMask: [.titled], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = content
        content.frame = NSRect(origin: .zero, size: size)
        window.orderFrontRegardless()
        defer { window.close() }
        try await Task.sleep(for: .milliseconds(200))
        window.layoutIfNeeded()
        content.layoutSubtreeIfNeeded()
        #expect(content.frame.size == size)
        let bitmap = try #require(content.bitmapImageRepForCachingDisplay(in: content.bounds))
        content.cacheDisplay(in: content.bounds, to: bitmap)
        let png = try #require(bitmap.representation(using: .png, properties: [:]))
        #expect(png.count > 1000)
        if let destination = ProcessInfo.processInfo.environment["LOOM_WORKSHOP_UI_RENDER_OUTPUT"] {
            let directory = URL(fileURLWithPath: destination, isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try png.write(to: directory.appendingPathComponent(name))
        }
    }
}

private struct WorkshopViewFixture {
    let root: URL
    let source: URL
    let application: URL
    init() async throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("loom-workshop-view-\(UUID().uuidString)", isDirectory: true)
        source = root.appendingPathComponent("Brick Images", isDirectory: true)
        application = root.appendingPathComponent("Texture Studio.app", isDirectory: true)
        let package = application.appendingPathComponent("Contents/Resources/TextureLoom", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        let info = ["CFBundleIdentifier": "org.ipde.texture-studio"]
        try PropertyListSerialization.data(fromPropertyList: info, format: .xml, options: 0)
            .write(to: application.appendingPathComponent("Contents/Info.plist"))
        // These small fixture artifacts exercise folder/catalog state only. They are never used for inference or training.
        let artifactPaths = ["classifier/model.mlpackage/data", "classifier/weights.safetensors", "bricks/depth/model.mlpackage/data", "bricks/depth/weights.safetensors", "bricks/depth/model.json"]
        var files: [[String: String]] = []
        for path in artifactPaths {
            let file = package.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            let data = path.hasSuffix("model.json") ?
                try JSONSerialization.data(withJSONObject: ["architecture_config": ["id": "loom-factory-material-residual-v1", "width": 24, "blocks": 4]]) : Data("Catalog fixture: \(path)".utf8)
            try data.write(to: file)
            files.append(["path": path, "sha256": SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()])
        }
        let modelID = UUID().uuidString
        let datasetID = "dataset-bricks"
        let manifest: [String: Any] = [
            "schema": "texture-loom.package.v1", "id": modelID, "name": "Texture Loom", "version": "1.0.0",
            "classifier": ["path": "classifier/model.mlpackage", "weightsPath": "classifier/weights.safetensors",
                           "labels": [["id": datasetID, "name": "Bricks"]]],
            "categories": [["id": datasetID, "name": "Bricks", "version": "1.0.0", "models": [[
                "map": "depth", "path": "bricks/depth/model.mlpackage", "weightsPath": "bricks/depth/weights.safetensors",
                "architecture": "loom-factory-material-residual-v1"
            ]]]], "files": files
        ]
        try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
            .write(to: package.appendingPathComponent("manifest.json"))
        for role in ["diffuse", "depth"] {
            let url = source.appendingPathComponent("brick_\(role).png")
            var pixels = [UInt8](repeating: 255, count: 256 * 256 * 4)
            for y in 0..<256 {
                for x in 0..<256 {
                    let i = (y * 256 + x) * 4
                    let joint = y % 32 < 2 || (x + (y / 32 % 2) * 32) % 64 < 2
                    pixels[i] = joint ? 45 : 180
                    pixels[i + 1] = joint ? 45 : (role == "diffuse" ? 92 : 180)
                    pixels[i + 2] = joint ? 45 : (role == "diffuse" ? 64 : 180)
                }
            }
            let provider = try #require(CGDataProvider(data: Data(pixels) as CFData))
            let image = try #require(CGImage(width: 256, height: 256, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 1024,
                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
            let destination = try #require(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
            CGImageDestinationAddImage(destination, image, nil)
            #expect(CGImageDestinationFinalize(destination))
            let stamp = try identity(url)
            let tag = ImageTag(id: UUID().uuidString, pairID: "brick-family", role: role,
                categoryID: datasetID, baseID: modelID, baseVersion: "1.0.0", classifiedBytes: stamp.0, classifiedModified: stamp.1)
            try writeTag(tag, to: url)
        }
        let project = try WorkshopProject(sourceFolder: source, textureStudio: application)
        var options = WorkshopOptions()
        options.name = "Brick Detail"; options.imageSize = 256; options.flipCrops = false
        try await project.updateOptions(options)
    }
    func addAdapter(named name: String) throws -> URL {
        let base = application.appendingPathComponent("Contents/Resources/TextureLoom")
        let manifest = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: base.appendingPathComponent("manifest.json"))) as? [String: Any])
        let modelID = try #require(manifest["id"] as? String)
        let directory = source.appendingPathComponent(".loom-workshop/runs/completed/adapter", isDirectory: true)
        let paths = ["bricks-depth/model.mlpackage/data", "bricks-depth/weights.safetensors"]
        var files: [[String: String]] = []
        for path in paths {
            let file = directory.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            let data = Data("Completed catalog fixture: \(path)".utf8)
            try data.write(to: file)
            files.append(["path": path, "sha256": SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()])
        }
        let weights = try Data(contentsOf: base.appendingPathComponent("bricks/depth/weights.safetensors"))
        let hash = SHA256.hash(data: weights).map { String(format: "%02x", $0) }.joined()
        let adapter: [String: Any] = [
            "schema": "texture-loom.adapter.v1", "id": UUID().uuidString, "name": name, "version": "1.0.0",
            "baseModel": ["id": modelID, "version": "1.0.0"],
            "categories": [["id": "dataset-bricks", "version": "1.0.0", "adapters": [[
                "map": "depth", "format": "coreml", "path": "bricks-depth/model.mlpackage", "weightsPath": "bricks-depth/weights.safetensors", "baseSHA256": hash
            ]]]], "files": files
        ]
        try JSONSerialization.data(withJSONObject: adapter, options: [.sortedKeys])
            .write(to: directory.appendingPathComponent("manifest.json"))
        return directory
    }
    func remove() { try? FileManager.default.removeItem(at: root) }
}
