// swift-tools-version: 6.2
import PackageDescription
import Foundation

let nativeDependency: Package.Dependency
if let path = ProcessInfo.processInfo.environment["LOOM_NATIVE_DEVELOPMENT_PATH"] {
    nativeDependency = .package(path: path)
} else {
    nativeDependency = .package(url: "https://github.com/gnaservicesinc/LoomFactory.git",
        revision: "84e55713712ba457e89942d5d671cf12811a1695")
}

let package = Package(
    name: "LoomWorkshop",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "WorkshopCore", targets: ["WorkshopCore"]),
        .executable(name: "LoomWorkshop", targets: ["LoomWorkshop"])
    ],
    dependencies: [nativeDependency],
    targets: [
        .target(name: "WorkshopCore", dependencies: [.product(name: "LoomNative", package: "LoomFactory")]),
        .executableTarget(name: "LoomWorkshop", dependencies: ["WorkshopCore"]),
        .testTarget(name: "WorkshopCoreTests", dependencies: ["WorkshopCore"], path: "Tests/WorkshopCoreTests"),
        .testTarget(name: "LoomWorkshopUITests", dependencies: ["WorkshopCore", "LoomWorkshop"], path: "Tests/LoomWorkshopUITests")
    ]
)
