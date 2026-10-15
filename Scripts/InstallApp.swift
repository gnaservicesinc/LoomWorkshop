import Foundation

func run(_ executable: String, _ arguments: [String], capture: Bool = false) throws -> String {
    let process = Process(); process.executableURL = URL(fileURLWithPath: executable); process.arguments = arguments
    let output = Pipe()
    if capture { process.standardOutput = output }
    try process.run()
    let data = capture ? output.fileHandleForReading.readDataToEndOfFile() : Data()
    process.waitUntilExit()
    guard process.terminationStatus == 0 else { throw NSError(domain: "WorkshopInstaller", code: Int(process.terminationStatus)) }
    return String(decoding: data, as: UTF8.self)
}
func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw NSError(domain: "WorkshopInstaller", code: 1, userInfo: [NSLocalizedDescriptionKey: message]) }
}
let files = FileManager.default
let source = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
let installationRoot = URL(fileURLWithPath: NSString(string: CommandLine.arguments[2]).expandingTildeInPath, isDirectory: true)
    .standardizedFileURL.resolvingSymlinksInPath()
let destination = installationRoot.appendingPathComponent("Applications/Loom Workshop.app", isDirectory: true)
do {
    try files.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
    let metadata = try PropertyListSerialization.propertyList(from: Data(contentsOf: source.appendingPathComponent("Contents/Info.plist")), format: nil) as! [String: Any]
    try require(metadata["CFBundleIdentifier"] as? String == "com.gnaservicesinc.LoomWorkshop", "The built bundle is not Loom Workshop.")
    try require(metadata["LoomBuildConfiguration"] as? String == "Release", "Install requires an optimized Release build of Loom Workshop.")
    let running = try run("/bin/ps", ["-axo", "command="], capture: true)
    try require(!running.contains(destination.appendingPathComponent("Contents/MacOS/LoomWorkshop").path), "Quit Loom Workshop before installing this build.")
    let stage = destination.deletingLastPathComponent().appendingPathComponent(".Loom Workshop.staging-" + UUID().uuidString + ".app", isDirectory: true)
    try run("/usr/bin/ditto", [source.path, stage.path])
    try run("/usr/bin/codesign", ["--verify", "--strict", "--verbose=2", stage.path])
    var backup: URL?
    if files.fileExists(atPath: destination.path) {
        // Keep a rollback copy only while the replacement is being verified.
        // Successful installation leaves one current app, without old bundles.
        backup = destination.deletingLastPathComponent().appendingPathComponent(
            ".Loom Workshop.rollback-" + UUID().uuidString + ".app", isDirectory: true)
        try files.moveItem(at: destination, to: backup!)
    }
    do {
        try files.moveItem(at: stage, to: destination)
        try run("/usr/bin/codesign", ["--verify", "--strict", "--verbose=2", destination.path])
    } catch {
        if files.fileExists(atPath: destination.path) { try? files.moveItem(at: destination, to: stage) }
        if let backup { try? files.moveItem(at: backup, to: destination) }
        throw error
    }
    if let backup { try files.removeItem(at: backup) }
    print(destination.path)
} catch {
    FileHandle.standardError.write(Data((error.localizedDescription + "\n").utf8))
    exit(1)
}
