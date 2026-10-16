import Darwin
import Foundation

/// Idle source-folder polling needs the catalog, but must not repeatedly hash
/// every base tensor or compile and load its classifier. Training and adapter
/// publication still perform their own complete readStudio verification.
struct StudioInspectionCache {
    private struct Stamp: Equatable {
        let device: dev_t
        let inode: ino_t
        let mode: mode_t
        let bytes: off_t
        let modifiedSeconds: Int
        let modifiedNanoseconds: Int
        let changedSeconds: Int
        let changedNanoseconds: Int
        var isSymbolicLink: Bool { mode & mode_t(S_IFMT) == mode_t(S_IFLNK) }

        init(_ value: stat) {
            device = value.st_dev; inode = value.st_ino; mode = value.st_mode; bytes = value.st_size
            modifiedSeconds = value.st_mtimespec.tv_sec; modifiedNanoseconds = value.st_mtimespec.tv_nsec
            changedSeconds = value.st_ctimespec.tv_sec; changedNanoseconds = value.st_ctimespec.tv_nsec
        }
    }
    private var cached: (inventory: [String: Stamp], studio: StudioBase)?

    mutating func read(_ application: URL, inspect: (URL) throws -> StudioBase = { try readStudio($0).0 }) throws -> StudioBase {
        let before = try inventory(application)
        // Enumeration does not descend directory links. Leave any linked
        // package to the complete reader rather than cache an unseen target.
        let cacheable = !before.values.contains(where: \.isSymbolicLink)
        if cacheable, let cached, cached.inventory == before { return cached.studio }
        // Never retain the previous success if reinspection fails.
        cached = nil
        let studio = try inspect(application)
        if cacheable, before == (try inventory(application)) { cached = (before, studio) }
        return studio
    }

    private func inventory(_ application: URL) throws -> [String: Stamp] {
        let app = application.standardizedFileURL.resolvingSymlinksInPath()
        let package = app.appendingPathComponent("Contents/Resources/TextureLoom", isDirectory: true).resolvingSymlinksInPath()
        var result: [String: Stamp] = [:]
        func record(_ file: URL) throws {
            var value = stat()
            guard lstat(file.path, &value) == 0 else {
                throw WorkshopError("Texture Studio's model package could not be inspected. Check that the application is still available.")
            }
            result[file.path] = Stamp(value)
        }
        let info = app.appendingPathComponent("Contents/Info.plist")
        try record(info)
        try record(info.resolvingSymlinksInPath())
        try record(package)
        var readError: Error?
        guard let files = FileManager.default.enumerator(at: package, includingPropertiesForKeys: nil,
            errorHandler: { _, error in readError = error; return false }) else {
            throw WorkshopError("Texture Studio's model package could not be inspected.")
        }
        for case let file as URL in files { try record(file) }
        if let readError { throw readError }
        return result
    }
}
