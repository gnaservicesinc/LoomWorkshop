import CoreML
import Foundation
import XCTest
@testable import WorkshopCore

final class CreateMLClassifierContractTests: XCTestCase {
    func testFactoryImportedTensorExportHasNoClassifierWeightsAndReadsItsNativeCatalog() throws {
        let app = try fixture(); defer { try? FileManager.default.removeItem(at: app) }
        let (_, manifest) = try readStudio(app)
        XCTAssertNil(manifest.classifier.weightsPath)
        XCTAssertEqual(manifest.classifier.inputType, "multiArray")
        XCTAssertEqual(manifest.classifier.inputShape?.prefix(2), [1, 3])
        XCTAssertEqual(Set(manifest.classifier.labels.map(\.id)), Set(manifest.categories.map(\.id)))
    }

    func testNativeImageClassifierMustMatchEveryFixedSourceLabel() throws {
        let app = try fixture(); defer { try? FileManager.default.removeItem(at: app) }
        let root = app.appendingPathComponent("Contents/Resources/TextureLoom")
        let asset = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("LoomFactory/Resources/DTTC3.mlmodel")
        guard FileManager.default.fileExists(atPath: asset.path) else { throw XCTSkip("Requires the trained Create ML fixture.") }
        let native = root.appendingPathComponent("native.mlmodelc")
        try FileManager.default.copyItem(at: MLModel.compileModel(at: asset), to: native)
        let configuration = MLModelConfiguration(); configuration.computeUnits = .cpuOnly
        let description = try MLModel(contentsOf: native, configuration: configuration).modelDescription
        let labels = try XCTUnwrap(description.classLabels as? [String]).enumerated().map {
            ["id": "native-dataset-\($0.offset)", "name": $0.element, "sourceLabel": $0.element]
        }
        let input = try XCTUnwrap(description.inputDescriptionsByName.first { $0.value.type == .image })
        let constraint = try XCTUnwrap(input.value.imageConstraint)
        let manifestURL = root.appendingPathComponent("manifest.json")
        var document = try JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL)) as! [String: Any]
        let original = (document["categories"] as! [[String: Any]])[0]
        document["categories"] = labels.map { label in
            var category = original; category["id"] = label["id"]; category["name"] = label["name"]; return category
        }
        document["classifier"] = ["architecture": "create-ml-image-classifier-v1", "format": "coreml", "path": native.lastPathComponent,
            "inputName": input.key, "outputName": try XCTUnwrap(description.predictedFeatureName), "inputType": "image",
            "inputWidth": constraint.pixelsWide, "inputHeight": constraint.pixelsHigh, "inputColorSpace": "RGB", "labels": labels]
        var files: [[String: String]] = []
        let entries = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isRegularFileKey])!
        for case let file as URL in entries where try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true && file.lastPathComponent != "manifest.json" {
            files.append(["path": String(file.standardizedFileURL.resolvingSymlinksInPath().path.dropFirst(root.path.count + 1)), "sha256": try fileSHA(file)])
        }
        document["files"] = files
        try JSONSerialization.data(withJSONObject: document).write(to: manifestURL)
        let (base, manifest) = try readStudio(app)
        XCTAssertEqual(base.categories.count, labels.count)
        XCTAssertNil(manifest.classifier.weightsPath)
        var classifier = document["classifier"] as! [String: Any]
        var mismatched = labels; mismatched[0]["sourceLabel"] = "Absent classifier label"
        classifier["labels"] = mismatched; document["classifier"] = classifier
        try JSONSerialization.data(withJSONObject: document).write(to: manifestURL)
        XCTAssertThrowsError(try readStudio(app))
    }

    private func fixture() throws -> URL {
        let archive = URL(fileURLWithPath: "/Users/andrewsmith/Library/Application Support/Loom Factory/Validation/classifier-folder-fix/IntegrationTextureLoom.textureloom")
        guard FileManager.default.fileExists(atPath: archive.path) else { throw XCTSkip("Requires a genuine imported-classifier Loom Factory export.") }
        let app = FileManager.default.temporaryDirectory.appendingPathComponent("workshop-native-\(UUID().uuidString).app")
            .standardizedFileURL.resolvingSymlinksInPath()
        let root = app.appendingPathComponent("Contents/Resources/TextureLoom")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try PropertyListSerialization.data(fromPropertyList: ["CFBundleIdentifier": "org.ipde.texture-studio"], format: .xml, options: 0)
            .write(to: app.appendingPathComponent("Contents/Info.plist"))
        let process = Process(); process.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        process.arguments = ["-x", "-k", archive.path, root.path]
        try process.run(); process.waitUntilExit(); XCTAssertEqual(process.terminationStatus, 0)
        return app
    }
}
