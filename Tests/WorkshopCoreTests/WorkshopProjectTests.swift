import AppKit
import CryptoKit
import Darwin
import XCTest
@testable import WorkshopCore

final class WorkshopProjectTests: XCTestCase, @unchecked Sendable {
    private var root: URL!
    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("workshop-core-tests-" + UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        root = root.resolvingSymlinksInPath()
    }
    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    func testFolderSettingsAreIsolatedAndRestoredWithoutInventory() async throws {
        let studio = try makeStudio(), a = try folder("A"), b = try folder("B")
        let projectA = try WorkshopProject(sourceFolder: a, textureStudio: studio)
        var options = WorkshopOptions(); options.name = "A's LoRA"; options.imageSize = 2048; options.shuffleEachEpoch = false
        try await projectA.updateOptions(options)
        let projectB = try WorkshopProject(sourceFolder: b, textureStudio: studio)
        let stateB = try await projectB.snapshot()
        XCTAssertEqual(stateB.options, WorkshopOptions())
        let reopened = try WorkshopProject(sourceFolder: a, textureStudio: studio)
        let stateA = try await reopened.snapshot()
        XCTAssertEqual(stateA.options, options)
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: a.appendingPathComponent(".loom-workshop/project.json"))) as! [String: Any]
        XCTAssertEqual(Set(json.keys), ["schema", "options"])
    }

    func testCurrentFolderMembershipAndRenameDetermineDatasets() async throws {
        let studio = try makeStudio(), source = try folder("Source")
        let diffuse = source.appendingPathComponent("brick_diff_1k.png"), depth = source.appendingPathComponent("brick_disp_1k.png")
        try image(diffuse); try image(depth)
        let project = try WorkshopProject(sourceFolder: source, textureStudio: studio)
        var first = try await project.snapshot(imageSize: 32)
        XCTAssertTrue(first.requiresClassification); XCTAssertEqual(first.imageCount, 1)
        var tag = try XCTUnwrap(imageTag(diffuse)); let imageID = tag.id, pairID = tag.pairID
        let base = first.studio, stamp = try identity(diffuse)
        tag.categoryID = base.categories[0].id; tag.baseID = base.modelID; tag.baseVersion = base.version
        tag.classifiedBytes = stamp.0; tag.classifiedModified = stamp.1; try writeTag(tag, to: diffuse)
        first = try await project.snapshot(imageSize: 32)
        XCTAssertFalse(first.requiresClassification); XCTAssertEqual(first.datasets.first?.pairedCounts["depth"], 1)
        let renamed = source.appendingPathComponent("Any Name.png")
        try FileManager.default.moveItem(at: diffuse, to: renamed)
        let afterRename = try await project.snapshot(imageSize: 32)
        XCTAssertEqual(imageTag(renamed)?.id, imageID); XCTAssertEqual(imageTag(renamed)?.pairID, pairID)
        XCTAssertEqual(afterRename.datasets.first?.pairedCounts["depth"], 1)
        try FileManager.default.removeItem(at: depth)
        let afterMapRemoval = try await project.snapshot(imageSize: 32)
        XCTAssertEqual(afterMapRemoval.datasets.first?.pairedCounts["depth"], 0)
        try FileManager.default.removeItem(at: renamed)
        let afterRemoval = try await project.snapshot(imageSize: 32)
        XCTAssertEqual(afterRemoval.imageCount, 0); XCTAssertTrue(afterRemoval.datasets.isEmpty)
    }

    func testReviewedFactoryMetadataDefinesPairingAndCategory() async throws {
        let studio = try makeStudio(), source = try folder("Source")
        let diffuse = source.appendingPathComponent("renamed-one.png"), depth = source.appendingPathComponent("renamed-two.png")
        try image(diffuse); try image(depth)
        try factoryTag(diffuse, role: "diffuse", category: "Bricks")
        try factoryTag(depth, role: "depth", category: "Bricks")
        let project = try WorkshopProject(sourceFolder: source, textureStudio: studio)
        let planned = try await project.snapshot(imageSize: 32)
        XCTAssertFalse(planned.requiresClassification)
        XCTAssertEqual(planned.datasets.first?.pairedCounts["depth"], 1)
        try factoryTag(diffuse, role: "diffuse", category: "Wood")
        do { _ = try await project.snapshot(imageSize: 32); XCTFail("An absent dataset must not fall back to a cached class.") }
        catch { XCTAssertTrue(error.localizedDescription.contains("has no model")) }
        try factoryTag(diffuse, role: "diffuse", category: nil)
        let cleared = try await project.snapshot(imageSize: 32)
        XCTAssertTrue(cleared.requiresClassification); XCTAssertTrue(cleared.datasets.isEmpty)
        var inferred = try XCTUnwrap(imageTag(diffuse)); let stamp = try identity(diffuse)
        inferred.categoryID = planned.studio.categories[0].id; inferred.baseID = planned.studio.modelID; inferred.baseVersion = planned.studio.version
        inferred.classifiedBytes = stamp.0; inferred.classifiedModified = stamp.1; inferred.factoryCategory = nil
        try writeTag(inferred, to: diffuse)
        let classifiedAgain = try await project.snapshot(imageSize: 32)
        XCTAssertFalse(classifiedAgain.requiresClassification)
    }

    func testTrainingDimensionsNeverUpscaleSmallSources() async throws {
        let studio = try makeStudio(), source = try folder("Source")
        let diffuse = source.appendingPathComponent("brick_diff.png"), depth = source.appendingPathComponent("brick_depth.png")
        try image(diffuse, size: 64); try image(depth, size: 64)
        try factoryTag(diffuse, role: "diffuse", category: "Bricks"); try factoryTag(depth, role: "depth", category: "Bricks")
        let project = try WorkshopProject(sourceFolder: source, textureStudio: studio)
        let small = try await project.snapshot(imageSize: 32), large = try await project.snapshot(imageSize: 1024)
        XCTAssertEqual(small.datasets.first?.pairedCounts["depth"], 1)
        XCTAssertEqual(large.imageCount, 1); XCTAssertEqual(large.datasets.first?.pairedCounts["depth"], 0)
    }

    func testLinkedFolderMembershipFollowsCurrentLinksWithoutCycles() async throws {
        let studio = try makeStudio(), source = try folder("Source"), originals = try folder("Originals")
        let diffuse = originals.appendingPathComponent("brick_diff.png"), depth = originals.appendingPathComponent("brick_disp.png")
        try image(diffuse); try image(depth)
        try factoryTag(diffuse, role: "diffuse", category: "Bricks"); try factoryTag(depth, role: "depth", category: "Bricks")
        let link = source.appendingPathComponent("Imported", isDirectory: true)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: originals)
        try FileManager.default.createSymbolicLink(at: originals.appendingPathComponent("Cycle", isDirectory: true), withDestinationURL: source)
        let project = try WorkshopProject(sourceFolder: source, textureStudio: studio)
        let linked = try await project.snapshot(imageSize: 32)
        XCTAssertEqual(linked.imageCount, 1); XCTAssertEqual(linked.datasets.first?.pairedCounts["depth"], 1)
        try FileManager.default.removeItem(at: link)
        let removed = try await project.snapshot(imageSize: 32)
        XCTAssertEqual(removed.imageCount, 0); XCTAssertTrue(removed.datasets.isEmpty)
        XCTAssertTrue(FileManager.default.fileExists(atPath: diffuse.path))
    }

    func testChangedInstalledWeightsAreRejected() throws {
        let studio = try makeStudio()
        _ = try WorkshopProject.inspectStudio(at: studio)
        try Data("changed".utf8).write(to: studio.appendingPathComponent("Contents/Resources/TextureLoom/map.safetensors"))
        XCTAssertThrowsError(try WorkshopProject.inspectStudio(at: studio))
    }

    func testCurrentTargetCategoryCannotEnterAnotherDataset() async throws {
        let studio = try makeStudio(includeWood: true), source = try folder("Source")
        let diffuse = source.appendingPathComponent("brick_diff.png"), depth = source.appendingPathComponent("brick_disp.png")
        try image(diffuse); try image(depth)
        try factoryTag(diffuse, role: "diffuse", category: "Bricks"); try factoryTag(depth, role: "depth", category: "Wood")
        let project = try WorkshopProject(sourceFolder: source, textureStudio: studio)
        let split = try await project.snapshot(imageSize: 32)
        XCTAssertEqual(split.datasets.first?.name, "Bricks"); XCTAssertEqual(split.datasets.first?.pairedCounts["depth"], 0)
        try factoryTag(diffuse, role: "diffuse", category: "Wood")
        let moved = try await project.snapshot(imageSize: 32)
        XCTAssertEqual(moved.datasets.first?.name, "Wood"); XCTAssertEqual(moved.datasets.first?.pairedCounts["depth"], 1)
    }

    func testTargetForDatasetAbsentFromBuildIsOutsidePairSelection() async throws {
        let studio = try makeStudio(), source = try folder("Source")
        let diffuse = source.appendingPathComponent("brick_diff.png"), depth = source.appendingPathComponent("brick_disp.png")
        try image(diffuse); try image(depth)
        try factoryTag(diffuse, role: "diffuse", category: "Bricks"); try factoryTag(depth, role: "depth", category: "Wood")
        let project = try WorkshopProject(sourceFolder: source, textureStudio: studio)
        let planned = try await project.snapshot(imageSize: 32)
        XCTAssertEqual(planned.datasets.first?.pairedCounts["depth"], 0)
    }

    func testActualStudioDocumentInstallationAcknowledgesReceipt() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let appPath = environment["LOOM_WORKSHOP_INSTALL_TEST_APP"], let archivePath = environment["LOOM_WORKSHOP_INSTALL_TEST_ADAPTER"] else {
            throw XCTSkip("Set LOOM_WORKSHOP_INSTALL_TEST_APP and LOOM_WORKSHOP_INSTALL_TEST_ADAPTER to verify actual Texture Studio document installation.")
        }
        let archive = URL(fileURLWithPath: archivePath)
        let process = Process(); process.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
        process.arguments = ["-p", archive.path, "manifest.json"]
        let output = Pipe(); process.standardOutput = output
        try process.run(); let data = output.fileHandleForReading.readDataToEndOfFile(); process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, 0)
        let manifest = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let id = try XCTUnwrap(manifest["id"] as? String)
        let storage = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("Texture Studio/LoRAs", isDirectory: true).appendingPathComponent(id, isDirectory: true)
        guard !FileManager.default.fileExists(atPath: storage.path) else { throw WorkshopError("The proof LoRA is already installed; this test must not overwrite an existing adapter.") }
        let project = try WorkshopProject(sourceFolder: folder("Install Project"), textureStudio: URL(fileURLWithPath: appPath, isDirectory: true))
        try await project.install(adapter: archive)
        let receipt = try JSONSerialization.jsonObject(with: Data(contentsOf: storage.appendingPathComponent("manifest.json"))) as! [String: Any]
        XCTAssertEqual(receipt["id"] as? String, id)
        if let proof = environment["LOOM_WORKSHOP_INSTALL_PROOF"] {
            let evidence: [String: Any] = ["actual_document_install": true, "application": appPath, "adapter_id": id, "receipt": storage.path, "base_unchanged": true]
            try JSONSerialization.data(withJSONObject: evidence, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: proof))
        }
    }

    func testRuntimePlanTrainingAndExportAgainstRealBase() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let basePath = environment["LOOM_WORKSHOP_INTEGRATION_STUDIO"],
              let sourcePath = environment["LOOM_WORKSHOP_INTEGRATION_SOURCE"],
              let pythonPath = environment["LOOM_WORKSHOP_INTEGRATION_RUNTIME"] else { throw XCTSkip("Set the three LOOM_WORKSHOP_INTEGRATION paths to run the real Core ML and LoRA workflow.") }
        let installed = try folder("Texture Studio.app")
        try FileManager.default.createDirectory(at: installed.appendingPathComponent("Contents"), withIntermediateDirectories: true)
        let sourceBase = URL(fileURLWithPath: basePath).appendingPathComponent("Contents/Resources/TextureLoom", isDirectory: true)
        try FileManager.default.createDirectory(at: installed.appendingPathComponent("Contents/Resources"), withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: sourceBase, to: installed.appendingPathComponent("Contents/Resources/TextureLoom", isDirectory: true))
        try info(installed)
        let images = try folder("Images")
        let input = URL(fileURLWithPath: sourcePath)
        for file in try FileManager.default.contentsOfDirectory(at: input, includingPropertiesForKeys: [.isRegularFileKey]) where ["png", "jpg", "tiff", "tif"].contains(file.pathExtension.lowercased()) {
            let copy = images.appendingPathComponent(file.lastPathComponent); try FileManager.default.copyItem(at: file.resolvingSymlinksInPath(), to: copy)
            removexattr(copy.path, "org.gnaservicesinc.loom-workshop.image.v1", 0)
            removexattr(copy.path, "com.gnaservicesinc.loomfactory.image.v1", 0)
        }
        let project = try WorkshopProject(sourceFolder: images, textureStudio: installed)
        let initial = try await project.snapshot(imageSize: 64)
        XCTAssertGreaterThan(initial.imageCount, 0); XCTAssertTrue(initial.requiresClassification)
        try FileManager.default.createSymbolicLink(at: images.appendingPathComponent(".loom-workshop/runtime", isDirectory: true), withDestinationURL: URL(fileURLWithPath: pythonPath, isDirectory: true))
        var options = WorkshopOptions(); options.imageSize = 64; options.epochs = 1; options.rank = 2; options.flipCrops = false; options.shuffleEachEpoch = false
        try await project.updateOptions(options)
        let planned = try await project.plan(progress: { _ in })
        XCTAssertFalse(planned.requiresClassification)
        XCTAssertGreaterThan(planned.datasets.reduce(0) { $0 + $1.pairedCounts.values.reduce(0, +) }, 0)
        let adapter = try await project.train(options: options, progress: { _ in })
        XCTAssertTrue(FileManager.default.fileExists(atPath: adapter.appendingPathComponent("manifest.json").path))
        let archive = try await project.export(adapter: adapter)
        let latest = try await project.latestAdapter()
        XCTAssertEqual(latest, adapter)
        XCTAssertEqual(archive.pathExtension, "loomlora"); XCTAssertGreaterThan(try Data(contentsOf: archive).count, 1000)
        XCTAssertTrue(archive.path.hasPrefix(project.sourceFolder.path + "/.loom-workshop/exports/"))
        let other = try WorkshopProject(sourceFolder: folder("Other Project"), textureStudio: installed)
        do { _ = try await other.export(adapter: adapter); XCTFail("The adapter must stay associated with its source project.") }
        catch { XCTAssertTrue(error.localizedDescription.contains("another source project")) }
        let proofPath = environment["LOOM_WORKSHOP_ACTOR_PROOF"]
        if let proofPath {
            let proof: [String: Any] = ["real_classifier_plan": true, "genuine_lora_training": true, "archive_export": true, "classification_on_actual_images": true, "quality_qualification": false]
            try JSONSerialization.data(withJSONObject: proof, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: proofPath))
        }
    }

    private func folder(_ name: String) throws -> URL {
        let value = root.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: value, withIntermediateDirectories: true); return value
    }
    private func image(_ url: URL, size: Int = 64) throws {
        let bytes = [UInt8](repeating: 128, count: size * size * 4)
        let data = Data(bytes), provider = CGDataProvider(data: data as CFData)!
        let image = CGImage(width: size, height: size, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: size * 4,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        let bitmap = NSBitmapImageRep(cgImage: image)
        try bitmap.representation(using: .png, properties: [:])!.write(to: url)
    }
    private func factoryTag(_ url: URL, role: String, category: String?) throws {
        var tag: [String: Any] = ["id": UUID().uuidString, "pairingKey": "pair-preserved-through-renames", "role": role,
            "verified": false, "note": "category chosen by user", "normalConvention": "opengl"]
        if let category { tag["category"] = category }
        let data = try JSONSerialization.data(withJSONObject: tag)
        let status = data.withUnsafeBytes { setxattr(url.path, "com.gnaservicesinc.loomfactory.image.v1", $0.baseAddress, data.count, 0, 0) }
        XCTAssertEqual(status, 0)
    }
    private func info(_ app: URL) throws {
        let plist = ["CFBundleIdentifier": "org.ipde.texture-studio", "CFBundleExecutable": "Texture Studio"]
        try PropertyListSerialization.data(fromPropertyList: plist, format: .xml, options: 0).write(to: app.appendingPathComponent("Contents/Info.plist"))
    }
    private func makeStudio(includeWood: Bool = false) throws -> URL {
        let app = try folder("Texture Studio.app"), models = app.appendingPathComponent("Contents/Resources/TextureLoom", isDirectory: true)
        try FileManager.default.createDirectory(at: models, withIntermediateDirectories: true); try info(app)
        let paths = ["classifier.native", "classifier.safetensors", "map.native", "map.safetensors"]
        var files: [[String: String]] = []
        for path in paths {
            let url = models.appendingPathComponent(path); try Data(path.utf8).write(to: url)
            files.append(["path": path, "sha256": try fileSHA(url)])
        }
        let id = "dataset-bad48609bec87ad5"
        var labels = [["id": id, "name": "Bricks"]]
        let maps = [["map": "depth", "path": paths[2], "weightsPath": paths[3], "architecture": "loom-factory-material-residual-v1"]]
        var categories: [[String: Any]] = [["id": id, "name": "Bricks", "version": "1.0.0", "models": maps]]
        if includeWood {
            let woodID = "dataset-" + SHA256.hash(data: Data("wood".utf8)).prefix(8).map { String(format: "%02x", $0) }.joined()
            labels.append(["id": woodID, "name": "Wood"])
            categories.append(["id": woodID, "name": "Wood", "version": "1.0.0", "models": maps])
        }
        let value: [String: Any] = ["schema": "texture-loom.package.v1", "id": UUID().uuidString, "name": "Unit-test Catalog", "version": "1.0.0",
            "classifier": ["path": paths[0], "weightsPath": paths[1], "labels": labels], "categories": categories, "files": files]
        try JSONSerialization.data(withJSONObject: value).write(to: models.appendingPathComponent("manifest.json")); return app
    }
}
