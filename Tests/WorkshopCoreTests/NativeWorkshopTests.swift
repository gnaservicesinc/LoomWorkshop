import CoreGraphics
import CryptoKit
import Darwin
import Foundation
import ImageIO
import LoomNative
import XCTest
@testable import WorkshopCore

final class NativeWorkshopTests: XCTestCase, @unchecked Sendable {
    func testNativePlanOneUpdatePreservesBaseAndExportsCompatibleLoRA() async throws {
        try await checkNativePlanOneUpdatePreservesBaseAndExportsCompatibleLoRA(architectureID: NativeArchitecture.identifier)
    }

    func testPyramidPlanOneUpdatePreservesBaseAndExportsCompatibleLoRA() async throws {
        try await checkNativePlanOneUpdatePreservesBaseAndExportsCompatibleLoRA(architectureID: NativeArchitecture.pyramidIdentifier)
    }

    private func checkNativePlanOneUpdatePreservesBaseAndExportsCompatibleLoRA(architectureID: String) async throws {
        let fixture = try NativeWorkshopFixture(architectureID: architectureID)
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let original = try Data(contentsOf: fixture.baseWeights)
        let project = try WorkshopProject(sourceFolder: fixture.images, textureStudio: fixture.application)
        let (base, _) = try readStudio(fixture.application)
        let profile = try XCTUnwrap(base.categories.first?.modelProfiles?.first)
        XCTAssertEqual(profile.architecture, architectureID)
        XCTAssertEqual(profile.baseParameters, try NativeWeights.read(from: fixture.baseWeights).values.reduce(0) { $0 + $1.values.count })
        var options = WorkshopOptions(); options.imageSize = 32; options.epochs = 1; options.rank = 2; options.alpha = 2
        options.flipCrops = false; options.shuffleEachEpoch = false; options.learningRate = 0.001
        try await project.updateOptions(options)
        let plan = try await project.plan(progress: { _ in })
        XCTAssertFalse(plan.requiresClassification)
        XCTAssertEqual(plan.datasets.first?.pairedCounts["depth"], 1)
        let adapter = try await project.train(options: options, progress: { _ in })
        let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: adapter.appendingPathComponent("manifest.json"))) as! [String: Any]
        let category = (manifest["categories"] as! [[String: Any]])[0]
        let model = (category["adapters"] as! [[String: Any]])[0]
        let weights = try NativeWeights.read(from: adapter.appendingPathComponent(model["weightsPath"] as! String))
        XCTAssertEqual(profile.trainableParameters(rank: options.rank), weights.values.reduce(0) { $0 + $1.values.count })
        XCTAssertTrue(weights.keys.allSatisfy { $0.hasSuffix(".down.weight") || $0.hasSuffix(".up.weight") })
        XCTAssertTrue(weights.filter { $0.key.hasSuffix(".up.weight") }.values.contains { $0.values.contains { $0 != 0 } })
        let metadataURL = adapter.appendingPathComponent(model["weightsPath"] as! String).deletingLastPathComponent().appendingPathComponent("adapter.json")
        let metadata = try JSONSerialization.jsonObject(with: Data(contentsOf: metadataURL)) as! [String: Any]
        XCTAssertEqual(metadata["updates"] as? Int, 1)
        XCTAssertEqual(metadata["base_weights_frozen"] as? Bool, true)
        XCTAssertLessThan(try XCTUnwrap(metadata["coreml_maximum_absolute_difference"] as? Double), 0.0006)
        XCTAssertEqual(try Data(contentsOf: fixture.baseWeights), original)
        XCTAssertEqual(model["baseSHA256"] as? String, try fileSHA(fixture.baseWeights))
        XCTAssertFalse(FileManager.default.fileExists(atPath: project.projectFolder.appendingPathComponent("runtime").path))
        let archive = try await project.export(adapter: adapter)
        XCTAssertEqual(archive.pathExtension, "loomlora")
        XCTAssertGreaterThan(try Data(contentsOf: archive).count, 1000)
        let run = adapter.deletingLastPathComponent()
        let checkpoints = try FileManager.default.subpathsOfDirectory(atPath: run.appendingPathComponent("checkpoints").path)
        XCTAssertTrue(checkpoints.contains { $0.hasSuffix(".optimizer.safetensors") })
        XCTAssertEqual(try Data(contentsOf: fixture.diffuse), fixture.originalDiffuse)
        XCTAssertEqual(try Data(contentsOf: fixture.target), fixture.originalTarget)
    }

    func testProjectActorCanStopNativeTrainingAndRetainsLastCompletedUpdate() async throws {
        let fixture = try NativeWorkshopFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let project = try WorkshopProject(sourceFolder: fixture.images, textureStudio: fixture.application)
        var options = WorkshopOptions(); options.imageSize = 32; options.epochs = 100_000; options.rank = 2; options.alpha = 2
        options.flipCrops = false; options.checkpointEveryEpochs = 100_000
        let signal = AsyncStream<Void>.makeStream()
        let running = Task { try await project.train(options: options) { event in
            if event.update == 1 { signal.continuation.yield(()) }
        } }
        for await _ in signal.stream { break }
        let started = ContinuousClock.now
        await project.cancel()
        do { _ = try await running.value; XCTFail("Stop must cancel the native operation.") }
        catch is CancellationError {} catch { XCTFail("Unexpected error: \(error)") }
        XCTAssertLessThan(ContinuousClock.now - started, .seconds(5))
        let runs = try FileManager.default.contentsOfDirectory(at: project.projectFolder.appendingPathComponent("runs"), includingPropertiesForKeys: nil)
        let run = try XCTUnwrap(runs.first)
        let result = try JSONSerialization.jsonObject(with: Data(contentsOf: run.appendingPathComponent("result.json"))) as! [String: Any]
        XCTAssertEqual(result["status"] as? String, "cancelled")
        XCTAssertFalse(FileManager.default.fileExists(atPath: run.appendingPathComponent("adapter/manifest.json").path))
        let files = try FileManager.default.subpathsOfDirectory(atPath: run.appendingPathComponent("checkpoints").path)
        XCTAssertTrue(files.contains { $0.contains("stopped-update-") && $0.hasSuffix(".optimizer.safetensors") })
        let reopened = try await project.snapshot(imageSize: 32)
        XCTAssertFalse(reopened.requiresClassification)
    }

    func testStopAfterCoreMLExportCannotPublishACompletionManifest() async throws {
        let fixture = try NativeWorkshopFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let project = try WorkshopProject(sourceFolder: fixture.images, textureStudio: fixture.application)
        let run = project.projectFolder.appendingPathComponent("runs/final-boundary", isDirectory: true)
        try FileManager.default.createDirectory(at: run, withIntermediateDirectories: true)
        var options = WorkshopOptions(); options.imageSize = 32; options.epochs = 1; options.rank = 2; options.alpha = 2
        let cancellation = NativeCancellation()
        let engine = NativeWorkshopEngine(source: fixture.images, project: project.projectFolder, application: fixture.application,
            options: options, cancellation: cancellation, progress: { event in
                if event.message.contains("LoRA exported and checked") { cancellation.cancel() }
            })
        do { _ = try await Task.detached { try engine.train(run: run) }.value; XCTFail("Stop before final binding must cancel publication.") }
        catch is CancellationError {} catch { XCTFail("Unexpected error: \(error)") }
        let result = try JSONSerialization.jsonObject(with: Data(contentsOf: run.appendingPathComponent("result.json"))) as! [String: Any]
        XCTAssertEqual(result["status"] as? String, "cancelled")
        XCTAssertFalse(FileManager.default.fileExists(atPath: run.appendingPathComponent("adapter/manifest.json").path))
        let exported = try FileManager.default.subpathsOfDirectory(atPath: run.appendingPathComponent("adapter").path)
        XCTAssertTrue(exported.contains { $0.hasSuffix("adapter.mlmodel") })
        let latest = try await project.latestAdapter()
        XCTAssertNil(latest)
    }

    func testSchedulesAndPartialAccumulationRetainTrainingOptions() {
        var options = WorkshopOptions(); options.batchSize = 2; options.gradientAccumulationSteps = 2
        XCTAssertEqual(NativeWorkshopEngine.stepsPerEpoch(samples: 5, options: options), 2)
        options.epochs = 3; options.warmupEpochs = 1; options.learningRate = 0.001
        options.learningRateSchedule = .cosine; options.minimumLearningRateRatio = 0.1
        XCTAssertEqual(NativeWorkshopEngine.learningRate(step: 0, total: 6, stepsPerEpoch: 2, options: options), 0.0005, accuracy: 1e-12)
        XCTAssertEqual(NativeWorkshopEngine.learningRate(step: 1, total: 6, stepsPerEpoch: 2, options: options), 0.001, accuracy: 1e-12)
        XCTAssertEqual(NativeWorkshopEngine.learningRate(step: 2, total: 6, stepsPerEpoch: 2, options: options), 0.001, accuracy: 1e-12)
        XCTAssertEqual(NativeWorkshopEngine.learningRate(step: 5, total: 6, stepsPerEpoch: 2, options: options), 0.0001, accuracy: 1e-12)
    }

    func testManualCheckpointThenSaveAndStopResumesExactPartialEpoch() async throws {
        let fixture = try NativeWorkshopFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        for index in 1...2 {
            for (original, role) in [(fixture.diffuse, "diffuse"), (fixture.target, "depth")] {
                let copied = fixture.images.appendingPathComponent("brick\(index)_\(role).png")
                try FileManager.default.copyItem(at: original, to: copied)
                let tag = try JSONSerialization.data(withJSONObject: ["id": UUID().uuidString,
                    "pairingKey": "pair-\(index)", "role": role, "category": "Bricks", "ignored": false])
                let status = tag.withUnsafeBytes { setxattr(copied.path, "com.gnaservicesinc.loomfactory.image.v1", $0.baseAddress, tag.count, 0, 0) }
                XCTAssertEqual(status, 0)
            }
        }
        let project = try WorkshopProject(sourceFolder: fixture.images, textureStudio: fixture.application)
        var options = WorkshopOptions(); options.imageSize = 32; options.epochs = 3; options.rank = 2; options.alpha = 2
        options.batchSize = 1; options.gradientAccumulationSteps = 2; options.flipCrops = true
        options.rotateCrops = true; options.shuffleEachEpoch = true; options.checkpointEveryEpochs = 100_000
        options.learningRateSchedule = .cosine; options.warmupEpochs = 1
        let reference = try await project.train(options: options, progress: { _ in })
        let run = project.projectFolder.appendingPathComponent("runs/manual-stop", isDirectory: true)
        try FileManager.default.createDirectory(at: run, withIntermediateDirectories: true)
        let control = WorkshopTrainingControl()
        let receiptIDs = WorkshopReceiptIDs()
        let engine = NativeWorkshopEngine(source: fixture.images, project: project.projectFolder, application: fixture.application,
            options: options, cancellation: NativeCancellation(), progress: { event in
                if event.update == 1 && event.loss != nil { receiptIDs.add(control.requestCheckpoint(stopAfter: false)) }
                if event.update == 3 && event.loss != nil { receiptIDs.add(control.requestCheckpoint(stopAfter: true)) }
            }, control: control)
        do { _ = try await Task.detached { try engine.train(run: run) }.value; XCTFail("Save & Stop must end the run after saving.") }
        catch is WorkshopSavedStop {} catch { XCTFail("Unexpected stop error: \(error)") }
        for id in receiptIDs.values { _ = try await control.wait(for: id) }
        XCTAssertEqual(receiptIDs.values.count, 2)
        let saved = try await project.resumableRuns()
        let selected = try XCTUnwrap(saved.first { $0.id == run.lastPathComponent })
        XCTAssertEqual(selected.updates, 3)
        let result = try JSONSerialization.jsonObject(with: Data(contentsOf: run.appendingPathComponent("result.json"))) as! [String: Any]
        XCTAssertEqual(result["status"] as? String, "saved-stop")
        let checkpointFiles = try FileManager.default.subpathsOfDirectory(atPath: run.appendingPathComponent("checkpoints").path)
        let partialPath = try XCTUnwrap(checkpointFiles.first { $0.hasSuffix("stopped-update-000000003.json") })
        let checkpoint = try JSONDecoder().decode(WorkshopCheckpointState.self, from: Data(contentsOf: run.appendingPathComponent("checkpoints/" + partialPath)))
        XCTAssertFalse(checkpoint.epochCompleted)
        XCTAssertEqual(checkpoint.epoch, 2)
        XCTAssertEqual(checkpoint.nextBatch, 2)
        let resumed: URL
        do { resumed = try await project.train(options: options, resumeRun: selected.directory, progress: { _ in }) }
        catch {
            XCTFail("Resume failed: \(error.localizedDescription). Run files: \(try FileManager.default.subpathsOfDirectory(atPath: run.path)). Result: \(String(data: try Data(contentsOf: run.appendingPathComponent("result.json")), encoding: .utf8) ?? "")")
            throw error
        }
        func adapterWeights(_ package: URL) throws -> [String: NativeTensor] {
            let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: package.appendingPathComponent("manifest.json"))) as! [String: Any]
            let category = (manifest["categories"] as! [[String: Any]])[0]
            let model = (category["adapters"] as! [[String: Any]])[0]
            return try NativeWeights.read(from: package.appendingPathComponent(model["weightsPath"] as! String))
        }
        let expected = try adapterWeights(reference), actual = try adapterWeights(resumed)
        XCTAssertEqual(Set(expected.keys), Set(actual.keys))
        for name in expected.keys {
            let difference = zip(expected[name]!.values, actual[name]!.values).map { abs($0 - $1) }.max() ?? 0
            XCTAssertLessThanOrEqual(difference, 0.0000001, name)
        }
        let remaining = try await project.resumableRuns()
        XCTAssertFalse(remaining.contains { $0.id == run.lastPathComponent })
        XCTAssertEqual(try Data(contentsOf: fixture.diffuse), fixture.originalDiffuse)
        XCTAssertEqual(try Data(contentsOf: fixture.target), fixture.originalTarget)
    }
}

private final class WorkshopReceiptIDs: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: [UUID] = []
    func add(_ id: UUID) { lock.withLock { stored.append(id) } }
    var values: [UUID] { lock.withLock { stored } }
}

private struct NativeWorkshopFixture {
    let root: URL
    let application: URL
    let images: URL
    let baseWeights: URL
    let diffuse: URL
    let target: URL
    let originalDiffuse: Data
    let originalTarget: Data
    init(architectureID: String = NativeArchitecture.identifier) throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("native-workshop-" + UUID().uuidString, isDirectory: true).resolvingSymlinksInPath()
        application = root.appendingPathComponent("Texture Studio.app", isDirectory: true)
        images = root.appendingPathComponent("Images", isDirectory: true)
        let models = application.appendingPathComponent("Contents/Resources/TextureLoom", isDirectory: true)
        try FileManager.default.createDirectory(at: models, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: images, withIntermediateDirectories: true)
        try PropertyListSerialization.data(fromPropertyList: ["CFBundleIdentifier": "org.ipde.texture-studio"], format: .xml, options: 0)
            .write(to: application.appendingPathComponent("Contents/Info.plist"))
        let architecture = try NativeArchitecture(mapKind: "depth", width: 4, blocks: 1, identifier: architectureID)
        let initial = try NativeTrainingSession(architecture: architecture, options: NativeTrainingOptions(), seed: 42)
        baseWeights = models.appendingPathComponent("weights.safetensors")
        try NativeWeights.write(initial.weights(), to: baseWeights)
        try NativeWorkshopEngine.json(["architecture_config": architecture.descriptor], to: models.appendingPathComponent("model.json"))
        _ = try NativeCoreML.export(weights: initial.weights(), architecture: architecture, imageSize: 32, to: models.appendingPathComponent("map.mlmodel"))
        // Factory category metadata is authoritative, so this fixture never invokes a classifier.
        try Data("unused classifier fixture".utf8).write(to: models.appendingPathComponent("classifier.native"))
        let category = "dataset-bad48609bec87ad5"
        let paths = ["weights.safetensors", "model.json", "map.mlmodel", "classifier.native"]
        try NativeWorkshopEngine.json(["schema": "texture-loom.package.v1", "id": UUID().uuidString,
            "name": "Native LoRA fixture", "version": "1.0.0",
            "classifier": ["path": "classifier.native", "labels": [["id": category, "name": "Bricks"]]],
            "categories": [["id": category, "name": "Bricks", "version": "1.0.0", "models": [["map": "depth", "architecture": architecture.identifier,
                "path": "map.mlmodel", "weightsPath": "weights.safetensors"]]]],
            "files": try paths.map { ["path": $0, "sha256": try fileSHA(models.appendingPathComponent($0))] }],
            to: models.appendingPathComponent("manifest.json"))
        diffuse = images.appendingPathComponent("brick_diff.png"); target = images.appendingPathComponent("brick_depth.png")
        for (path, role) in [(diffuse, "diffuse"), (target, "depth")] {
            let values = (0..<(32 * 32)).map { UInt8(($0 % 32) * 7 + 12) }
            let provider = CGDataProvider(data: Data(values) as CFData)!
            let image = CGImage(width: 32, height: 32, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: 32,
                space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGBitmapInfo(rawValue: 0), provider: provider,
                decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
            let destination = CGImageDestinationCreateWithURL(path as CFURL, "public.png" as CFString, 1, nil)!
            CGImageDestinationAddImage(destination, image, nil); XCTAssertTrue(CGImageDestinationFinalize(destination))
            let tag = try JSONSerialization.data(withJSONObject: ["id": UUID().uuidString, "pairingKey": "native-pair", "role": role, "category": "Bricks", "ignored": false])
            let status = tag.withUnsafeBytes { setxattr(path.path, "com.gnaservicesinc.loomfactory.image.v1", $0.baseAddress, tag.count, 0, 0) }
            guard status == 0 else { throw WorkshopError("Could not create the image-owned fixture metadata.") }
        }
        originalDiffuse = try Data(contentsOf: diffuse); originalTarget = try Data(contentsOf: target)
    }
}
