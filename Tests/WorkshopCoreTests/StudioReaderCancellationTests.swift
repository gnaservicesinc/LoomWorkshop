import CryptoKit
import Foundation
import LoomNative
import XCTest
@testable import WorkshopCore

final class StudioReaderCancellationTests: XCTestCase {
    func testCancelledStudioInspectionAndFileVerificationStopBeforeReading() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("studio-reader-cancellation-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let application = root.appendingPathComponent("Texture Studio.app", isDirectory: true)
        let package = application.appendingPathComponent("Contents/Resources/TextureLoom", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        let info: [String: Any] = ["CFBundleIdentifier": "org.ipde.texture-studio"]
        try PropertyListSerialization.data(fromPropertyList: info, format: .xml, options: 0)
            .write(to: application.appendingPathComponent("Contents/Info.plist"))
        let artifacts = ["classifier.native": Data("classifier fixture".utf8), "map.native": Data("map fixture".utf8),
                         "weights.safetensors": Data("weights fixture".utf8)]
        for (path, bytes) in artifacts { try bytes.write(to: package.appendingPathComponent(path)) }
        let files = artifacts.map { path, bytes in
            ["path": path, "sha256": SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()]
        }
        let id = UUID().uuidString
        let manifest: [String: Any] = ["schema": "texture-loom.package.v1", "id": id, "name": "Cancellation fixture", "version": "1",
            "classifier": ["path": "classifier.native", "labels": [["id": "bricks", "name": "Bricks"]]],
            "categories": [["id": "bricks", "name": "Bricks", "version": "1",
                "models": [["map": "depth", "path": "map.native", "weightsPath": "weights.safetensors", "architecture": "fixture"]]]],
            "files": files]
        try JSONSerialization.data(withJSONObject: manifest).write(to: package.appendingPathComponent("manifest.json"))
        let inspected = try readStudio(application)
        XCTAssertEqual(inspected.0.modelID, id)
        try verifyFiles(inspected.1.files, root: package)

        let cancellation = NativeCancellation(); cancellation.cancel()
        XCTAssertThrowsError(try readStudio(application, cancellation: cancellation)) { XCTAssertTrue($0 is CancellationError) }
        XCTAssertThrowsError(try verifyFiles(inspected.1.files, root: package, cancellation: cancellation)) {
            XCTAssertTrue($0 is CancellationError)
        }
        XCTAssertThrowsError(try fileSHA(package.appendingPathComponent("weights.safetensors"), cancellation: cancellation)) {
            XCTAssertTrue($0 is CancellationError)
        }
    }
}
