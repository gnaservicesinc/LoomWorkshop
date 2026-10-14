import CryptoKit
import Darwin
import Foundation
import ImageIO

struct ImageTag: Codable, Sendable {
    var id: String
    var pairID: String
    var role: String
    var categoryID: String?
    var baseID: String?
    var baseVersion: String?
    var classifiedBytes: UInt64?
    var classifiedModified: TimeInterval?
    var normalConvention: String?
    var factoryCategory: String?
}
struct WorkshopSample: Sendable {
    let id: String
    let pairID: String
    let input: URL
    let targets: [String: URL]
    let categoryID: String?
    let normalConvention: String
}
struct ActiveImageSelection {
    let id: String
    let pairID: String
    let role: String
    let categoryID: String?
    let normalConvention: String
    let ignored: Bool
}
private struct FactoryTag: Decodable {
    let id: String
    let pairingKey: String
    let role: String
    let category: String?
    let normalConvention: String?
    let ignored: Bool?
}
private let tagKey = "org.gnaservicesinc.loom-workshop.image.v1"
func activeImageSelection(_ file: URL, studio: StudioBase) throws -> ActiveImageSelection {
    guard let tag = imageTag(file) else { throw WorkshopError("The image lost its identity metadata: \(file.lastPathComponent).") }
    let state = try identity(file)
    var category = tag.baseID == studio.modelID && tag.baseVersion == studio.version &&
        tag.classifiedBytes == state.0 && tag.classifiedModified == state.1 ? tag.categoryID : nil
    if let data = metadataBytes(file, key: "com.gnaservicesinc.loomfactory.image.v1") {
        let factory = try JSONDecoder().decode(FactoryTag.self, from: data)
        let name = factory.category?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let name, !name.isEmpty { category = "dataset-" + digest(name.lowercased()) }
        else if tag.factoryCategory != nil { category = nil }
        return ActiveImageSelection(id: tag.id, pairID: factory.pairingKey, role: factory.role, categoryID: category,
            normalConvention: factory.normalConvention ?? tag.normalConvention ?? "opengl", ignored: factory.ignored == true)
    }
    if tag.factoryCategory != nil { category = nil }
    return ActiveImageSelection(id: tag.id, pairID: tag.pairID, role: tag.role, categoryID: category,
        normalConvention: tag.normalConvention ?? "opengl", ignored: false)
}
private func metadataBytes(_ file: URL, key: String) -> Data? {
    let size = getxattr(file.path, key, nil, 0, 0, 0)
    guard size > 0, size <= 65536 else { return nil }
    var bytes = Data(count: size)
    let count = bytes.withUnsafeMutableBytes { getxattr(file.path, key, $0.baseAddress, size, 0, 0) }
    return count == size ? bytes : nil
}
func imageTag(_ file: URL) -> ImageTag? {
    metadataBytes(file, key: tagKey).flatMap { try? JSONDecoder().decode(ImageTag.self, from: $0) }
}
func writeTag(_ tag: ImageTag, to file: URL) throws {
    let data = try JSONEncoder().encode(tag)
    let result = data.withUnsafeBytes { setxattr(file.path, tagKey, $0.baseAddress, data.count, 0, 0) }
    guard result == 0 else { throw WorkshopError("Image metadata could not be saved for \(file.lastPathComponent). Check that the source folder is writable.") }
}
/// Review flags and notes remain informational; only these fields select training data.
func membershipSignature(_ file: URL) throws -> Data {
    let specifications: [(String, [String])] = [
        ("com.gnaservicesinc.loomfactory.image.v1", ["id", "pairingKey", "role", "category", "normalConvention", "ignored"]),
        (tagKey, ["id", "pairID", "role", "categoryID", "baseID", "baseVersion", "classifiedBytes", "classifiedModified", "factoryCategory", "normalConvention"])
    ]
    var result: [Any] = []
    for (key, fields) in specifications {
        guard let data = metadataBytes(file, key: key) else { result.append(NSNull()); continue }
        guard let tag = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw WorkshopError("Image metadata needs repair: \(file.lastPathComponent).")
        }
        result.append(Dictionary(uniqueKeysWithValues: fields.map { ($0, tag[$0] ?? NSNull()) }))
    }
    return try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
}
func imageDimensions(_ file: URL) -> (Int, Int)? {
    guard let source = CGImageSourceCreateWithURL(file as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary),
          let values = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [String: Any],
          let width = (values[kCGImagePropertyPixelWidth as String] as? NSNumber)?.intValue,
          let height = (values[kCGImagePropertyPixelHeight as String] as? NSNumber)?.intValue else { return nil }
    return (width, height)
}
func identity(_ file: URL) throws -> (UInt64, TimeInterval) {
    let attributes = try FileManager.default.attributesOfItem(atPath: file.path)
    return ((attributes[.size] as? NSNumber)?.uint64Value ?? 0, (attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0)
}
private func filenameRole(_ file: URL, legacy: Bool = false) -> (String, String, String) {
    var stem = file.deletingPathExtension().lastPathComponent.lowercased()
        .replacingOccurrences(of: "-", with: "_").replacingOccurrences(of: " ", with: "_")
    let diffusePattern = legacy ? "(?:^|_)(?:diff|diffuse|albedo|basecolor|color|col)(?=_|$)"
        : "(?:^|_)(?:diff|diffuse|albedo|base_?(?:color|colour|col|colo)|color|colour|col_?1|coll_?1|col|colo)(?=_|$)"
    let normalPattern = legacy ? "(?:^|_)(?:nor|normal)(?=_|$)" : "(?:^|_)(?:nor|normal|normalgl)(?=_|$)"
    var replacements: [(String, String, String)] = [
        ("(?:^|_)(?:nor|normal)_(?:dx|directx)(?=_|$)", "normal", "directx"),
        ("(?:^|_)(?:nor|normal)_(?:gl|opengl)(?=_|$)", "normal", "opengl"),
        (diffusePattern, "diffuse", "opengl"),
        ("(?:^|_)(?:disp|displacement|depth|height)(?=_|$)", "depth", "opengl"),
        ("(?:^|_)(?:rough|roughness)(?=_|$)", "roughness", "opengl"),
        (normalPattern, "normal", "opengl"),
        ("(?:^|_)(?:ao|ambientocclusion|metal|metalness|metallic|spec|specular|opacity|alpha)(?=_|$)", "unknown", "opengl")]
    if !legacy {
        replacements.append(("(?:^|_)(?:bump|bumpmap)(?=_|$)", "bump", "opengl"))
        replacements.append(("(?:^|_)(?:normaldx)(?=_|$)", "normal", "directx"))
    }
    if legacy {
        for (pattern, role, convention) in replacements where stem.range(of: pattern, options: .regularExpression) != nil {
            let family = stem.replacingOccurrences(of: pattern, with: "_", options: .regularExpression)
                .trimmingCharacters(in: CharacterSet(charactersIn: "_"))
            return (role, family, convention)
        }
        return ("diffuse", stem, "opengl")
    }
    // Role words in a material's name must not override its map suffix.
    let matches = replacements.compactMap { pattern, role, convention -> (Range<String.Index>, String, String)? in
        guard let expression = try? NSRegularExpression(pattern: pattern),
              let match = expression.matches(in: stem, range: NSRange(stem.startIndex..., in: stem)).last,
              let range = Range(match.range, in: stem) else { return nil }
        return (range, role, convention)
    }
    if let (range, role, convention) = matches.max(by: { $0.0.upperBound < $1.0.upperBound }) {
        stem.replaceSubrange(range, with: "_")
        stem = stem.trimmingCharacters(in: CharacterSet(charactersIn: "_"))
        return (role, stem, convention)
    }
    return ("unknown", stem, "opengl")
}
private func digest(_ text: String) -> String { SHA256.hash(data: Data(text.utf8)).prefix(8).map { String(format: "%02x", $0) }.joined() }
private func physicalPath(_ file: URL) -> String {
    guard let resolved = realpath(file.path, nil) else { return file.standardizedFileURL.path }
    defer { free(resolved) }
    return String(cString: resolved)
}
private func sourceFiles(_ source: URL) throws -> [URL] {
    var folders = [source], visitedFolders = Set<String>(), visitedFiles = Set<String>(), files: [URL] = []
    while let folder = folders.popLast() {
        let actualFolder = physicalPath(folder)
        guard visitedFolders.insert(actualFolder).inserted else { continue }
        // Read the actual directory, then reconstruct its in-project link path.
        // URL resource caches describe the link itself, so enumerating that URL
        // directly can report "not a directory" for a perfectly valid link.
        let entries = try FileManager.default.contentsOfDirectory(atPath: actualFolder)
            .filter { !$0.hasPrefix(".") }.sorted().map { folder.appendingPathComponent($0) }
        for entry in entries {
            var directory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: entry.path, isDirectory: &directory) else { continue }
            if directory.boolValue { folders.append(entry) }
            else {
                let actualFile = physicalPath(entry)
                if visitedFiles.insert(actualFile).inserted { files.append(entry) }
            }
        }
    }
    return files
}
func currentSamples(source: URL, studio: StudioBase, imageSize: Int) throws -> [WorkshopSample] {
    let entries = try sourceFiles(source)
    let extensions = Set(["png", "jpg", "jpeg", "tif", "tiff", "heic", "heif", "webp", "bmp", "exr"])
    let allowedCategories = Set(studio.categories.map(\.id))
    var groups: [String: [(URL, ImageTag, String)]] = [:]
    var seenIDs = Set<String>()
    for file in entries {
        guard extensions.contains(file.pathExtension.lowercased()), FileManager.default.fileExists(atPath: file.path),
              (try? file.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory != true else { continue }
        let inferred = filenameRole(file)
        var tag = imageTag(file) ?? ImageTag(id: UUID().uuidString, pairID: "", role: inferred.0, normalConvention: inferred.2)
        // Repair exact earlier filename inferences only. Renames, custom pairs
        // and Factory's deliberate map-role decisions remain authoritative.
        if metadataBytes(file, key: "com.gnaservicesinc.loomfactory.image.v1") == nil {
            let folder = String(file.deletingLastPathComponent().standardizedFileURL.path.dropFirst(source.standardizedFileURL.path.count))
            let previous = filenameRole(file, legacy: true)
            let currentPair = "pair-" + digest(folder + "/" + inferred.1)
            if tag.role == previous.0, tag.pairID == "pair-" + digest(folder + "/" + previous.1),
               tag.role != inferred.0 || tag.pairID != currentPair {
                tag.role = inferred.0; tag.pairID = currentPair; tag.normalConvention = inferred.2
                tag.categoryID = nil; tag.baseID = nil; tag.baseVersion = nil
                tag.classifiedBytes = nil; tag.classifiedModified = nil; tag.factoryCategory = nil
            }
        }
        if tag.pairID.isEmpty {
            let folder = String(file.deletingLastPathComponent().standardizedFileURL.path.dropFirst(source.standardizedFileURL.path.count))
            tag.pairID = "pair-" + digest(folder + "/" + inferred.1)
        }
        if !seenIDs.insert(tag.id).inserted { tag.id = UUID().uuidString; seenIDs.insert(tag.id) }
        let current = try identity(file)
        if tag.baseID != studio.modelID || tag.baseVersion != studio.version || tag.classifiedBytes != current.0 || tag.classifiedModified != current.1 { tag.categoryID = nil }
        var currentFactoryCategory: String?
        if let bytes = metadataBytes(file, key: "com.gnaservicesinc.loomfactory.image.v1") {
            guard let factory = try? JSONDecoder().decode(FactoryTag.self, from: bytes) else {
                throw WorkshopError("Loom Factory metadata could not be read for \(file.lastPathComponent). Check this file's metadata in Loom Factory.")
            }
            if factory.ignored == true { continue }
            tag.pairID = factory.pairingKey
            tag.role = factory.role
            tag.normalConvention = factory.normalConvention ?? inferred.2
            currentFactoryCategory = factory.category?.trimmingCharacters(in: .whitespacesAndNewlines)
            if currentFactoryCategory?.isEmpty == true { currentFactoryCategory = nil }
            if let category = currentFactoryCategory {
                let categoryID = "dataset-" + digest(category.trimmingCharacters(in: .whitespacesAndNewlines).lowercased())
                if allowedCategories.contains(categoryID) {
                    tag.categoryID = categoryID; tag.baseID = studio.modelID; tag.baseVersion = studio.version
                    tag.classifiedBytes = current.0; tag.classifiedModified = current.1
                    tag.factoryCategory = category
                } else if tag.role == "diffuse" {
                    throw WorkshopError("The \(category) dataset has no model in this Texture Studio build.")
                } else {
                    // A map for another dataset is outside this pair's training
                    // selection, even if this build has no model for that dataset.
                    tag.categoryID = categoryID; tag.factoryCategory = category
                }
            }
        }
        if currentFactoryCategory == nil, tag.factoryCategory != nil { tag.categoryID = nil; tag.factoryCategory = nil }
        if tag.role == "diffuse", let categoryID = tag.categoryID, !allowedCategories.contains(categoryID) { tag.categoryID = nil }
        try writeTag(tag, to: file)
        groups[tag.pairID, default: []].append((file, tag, tag.normalConvention ?? inferred.2))
    }
    var samples: [WorkshopSample] = []
    for members in groups.values {
        let inputs = members.filter { $0.1.role == "diffuse" }
        guard inputs.count <= 1 else { throw WorkshopError("Two diffuse images belong to the same pair. Keep one diffuse image per material and resolution.") }
        guard let (input, tag, _) = inputs.first else { continue }
        var targets: [String: URL] = [:], convention = "opengl"
        if let dimensions = imageDimensions(input), dimensions.0 >= imageSize, dimensions.1 >= imageSize {
            for role in ["depth", "roughness", "normal"] {
                let candidates = members.filter { candidate in
                    guard candidate.1.role == role, role != "normal" || candidate.2.lowercased() != "directx",
                          let targetSize = imageDimensions(candidate.0) else { return false }
                    if let inputCategory = tag.categoryID, let targetCategory = candidate.1.categoryID, inputCategory != targetCategory { return false }
                    return targetSize == dimensions
                }.sorted { $0.2 == "opengl" && $1.2 != "opengl" }
                let sameConvention = candidates.filter { $0.2 == (candidates.first?.2 ?? "opengl") }
                guard sameConvention.count <= 1 else { throw WorkshopError("More than one \(role) map belongs to \(input.lastPathComponent). Keep one matching map for each target.") }
                if let (target, _, normal) = candidates.first {
                    targets[role] = target
                    if role == "normal" { convention = normal }
                }
            }
        }
        samples.append(WorkshopSample(id: tag.id, pairID: tag.pairID, input: input, targets: targets,
                                      categoryID: tag.categoryID, normalConvention: convention))
    }
    return samples.sorted { $0.input.path < $1.input.path }
}
