import CryptoKit
import Foundation
import CoreML

struct BaseManifest: Decodable, Sendable {
    struct File: Decodable, Sendable { let path: String; let sha256: String }
    struct Classifier: Decodable, Sendable {
        let path: String; let weightsPath: String?; let architecture: String?
        let inputName: String?; let outputName: String?; let labels: [Label]
        let inputWidth: Int?; let inputHeight: Int?; let inputColorSpace: String?
        let inputType: String?; let inputShape: [Int]?
    }
    struct Label: Decodable, Sendable {
        let id: String; let name: String; let sourceLabel: String?
        var classifierValue: String { sourceLabel ?? id }
    }
    struct Model: Decodable, Sendable { let map: String; let path: String; let weightsPath: String; let architecture: String }
    struct Category: Decodable, Sendable { let id: String; let name: String; let version: String; let models: [Model] }
    let schema: String; let id: String; let name: String; let version: String
    let classifier: Classifier; let categories: [Category]; let files: [File]
}
func modelPath(_ relative: String, in root: URL) throws -> URL {
    let parts = relative.split(separator: "/", omittingEmptySubsequences: false)
    guard !relative.isEmpty, !relative.hasPrefix("/"), !relative.contains("\\"), parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
        throw WorkshopError("A model artifact points outside its own package.")
    }
    var file = root.standardizedFileURL.resolvingSymlinksInPath()
    for part in parts {
        file.appendPathComponent(String(part))
        guard (try? FileManager.default.destinationOfSymbolicLink(atPath: file.path)) == nil else { throw WorkshopError("Model packages cannot contain links.") }
    }
    return file
}
func fileSHA(_ file: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: file); defer { try? handle.close() }
    var digest = SHA256()
    while let bytes = try handle.read(upToCount: 1_048_576), !bytes.isEmpty { digest.update(data: bytes) }
    return digest.finalize().map { String(format: "%02x", $0) }.joined()
}
func verifyFiles(_ files: [BaseManifest.File], root: URL) throws {
    guard !files.isEmpty, Set(files.map(\.path)).count == files.count else { throw WorkshopError("The model file inventory is empty or repeated.") }
    for entry in files {
        guard try fileSHA(modelPath(entry.path, in: root)) == entry.sha256.lowercased() else { throw WorkshopError("A model file is missing or changed: \(entry.path).") }
    }
}
func readStudio(_ application: URL) throws -> (StudioBase, BaseManifest) {
    let app = application.standardizedFileURL.resolvingSymlinksInPath()
    guard let info = try? PropertyListSerialization.propertyList(from: Data(contentsOf: app.appendingPathComponent("Contents/Info.plist")), format: nil) as? [String: Any],
          info["CFBundleIdentifier"] as? String == "org.ipde.texture-studio" else { throw WorkshopError("Choose the installed Texture Studio.app application.") }
    let root = app.appendingPathComponent("Contents/Resources/TextureLoom", isDirectory: true)
    guard FileManager.default.fileExists(atPath: root.appendingPathComponent("manifest.json").path) else {
        throw WorkshopError("This Texture Studio build has no Texture Loom base. Train and export a model in Loom Factory, then build Texture Studio with that package before using Loom Workshop.")
    }
    let manifest = try JSONDecoder().decode(BaseManifest.self, from: Data(contentsOf: root.appendingPathComponent("manifest.json")))
    guard manifest.schema == "texture-loom.package.v1", UUID(uuidString: manifest.id) != nil, !manifest.version.isEmpty,
          !manifest.categories.isEmpty, Set(manifest.categories.map(\.id)).count == manifest.categories.count,
          Set(manifest.classifier.labels.map(\.id)).count == manifest.classifier.labels.count,
          Set(manifest.classifier.labels.map(\.classifierValue)).count == manifest.classifier.labels.count,
          Set(manifest.classifier.labels.map(\.id)) == Set(manifest.categories.map(\.id)),
          manifest.classifier.labels.allSatisfy({ label in !label.classifierValue.isEmpty && manifest.categories.contains { $0.id == label.id && $0.name == label.name } }) else { throw WorkshopError("Texture Studio's fixed model catalog is incomplete.") }
    try verifyFiles(manifest.files, root: root)
    let inventory = Set(manifest.files.map(\.path))
    let paths = [manifest.classifier.path] + (manifest.classifier.weightsPath.map { [$0] } ?? []) +
        manifest.categories.flatMap { $0.models.flatMap { [$0.path, $0.weightsPath] } }
    for path in paths {
        let file = try modelPath(path, in: root)
        guard FileManager.default.fileExists(atPath: file.path), inventory.contains(where: { $0 == path || $0.hasPrefix(path + "/") }) else { throw WorkshopError("The installed model is missing a registered native artifact: \(path).") }
        if (try file.resourceValues(forKeys: [.isDirectoryKey])).isDirectory == true {
            guard let entries = FileManager.default.enumerator(at: file, includingPropertiesForKeys: [.isRegularFileKey]) else { throw WorkshopError("The installed model folder could not be inspected.") }
            for case let item as URL in entries where try item.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true && item.lastPathComponent != ".DS_Store" {
                let relative = String(item.standardizedFileURL.resolvingSymlinksInPath().path.dropFirst(root.standardizedFileURL.resolvingSymlinksInPath().path.count + 1))
                _ = try modelPath(relative, in: root)
                guard inventory.contains(relative) else { throw WorkshopError("The installed model has an unregistered artifact: \(relative).") }
            }
        }
    }
    if manifest.classifier.architecture == "create-ml-image-classifier-v1" {
        let source = try modelPath(manifest.classifier.path, in: root)
        let compiled = source.pathExtension == "mlmodelc" ? source : try MLModel.compileModel(at: source)
        let configuration = MLModelConfiguration(); configuration.computeUnits = .cpuOnly
        let description = try MLModel(contentsOf: compiled, configuration: configuration).modelDescription
        guard let input = manifest.classifier.inputName, let output = manifest.classifier.outputName,
              description.inputDescriptionsByName.count == 1,
              let feature = description.inputDescriptionsByName[input],
              description.outputDescriptionsByName[output]?.type == .string,
              let labels = description.classLabels as? [String], !labels.isEmpty,
              Set(labels).count == labels.count,
              Set(labels) == Set(manifest.classifier.labels.map(\.classifierValue)) else {
            throw WorkshopError("Texture Studio's fixed classifier does not match its dataset catalog. Rebuild Texture Studio with a complete Loom Factory export.")
        }
        if let constraint = feature.imageConstraint {
            guard constraint.pixelsWide > 0, constraint.pixelsHigh > 0,
                  constraint.pixelsWide <= 8192, constraint.pixelsHigh <= 8192,
                  manifest.classifier.inputWidth.map({ $0 == constraint.pixelsWide }) ?? true,
                  manifest.classifier.inputHeight.map({ $0 == constraint.pixelsHigh }) ?? true,
                  manifest.classifier.inputType.map({ $0 == "image" }) ?? true,
                  manifest.classifier.inputColorSpace.map({ $0 == "RGB" }) ?? true else {
                throw WorkshopError("Texture Studio's classifier image dimensions do not match its native model.")
            }
        } else if let constraint = feature.multiArrayConstraint {
            let shape = constraint.shape.map(\.intValue)
            guard constraint.dataType == .float32, shape.count == 4, shape[0] == 1, shape[1] == 3,
                  shape[2] > 0, shape[3] > 0, shape[2] <= 8192, shape[3] <= 8192,
                  manifest.classifier.inputShape.map({ $0 == shape }) ?? true,
                  manifest.classifier.inputType.map({ $0 == "multiArray" }) ?? true else {
                throw WorkshopError("Texture Studio's classifier requires a supported RGB tensor input.")
            }
        } else { throw WorkshopError("Texture Studio's classifier requires an image or RGB tensor input.") }
    }
    let datasets = manifest.categories.map { BaseDataset(id: $0.id, name: $0.name, version: $0.version, mapKinds: $0.models.map(\.map)) }
    return (StudioBase(applicationURL: app, modelID: manifest.id, modelName: manifest.name, version: manifest.version, categories: datasets, packageURL: root), manifest)
}
