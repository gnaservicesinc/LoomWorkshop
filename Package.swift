// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "LoomWorkshop",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "WorkshopCore", targets: ["WorkshopCore"]),
        .executable(name: "LoomWorkshop", targets: ["LoomWorkshop"])
    ],
    targets: [
        .target(name: "WorkshopCore"),
        .executableTarget(name: "LoomWorkshop", dependencies: ["WorkshopCore"]),
        .testTarget(name: "WorkshopCoreTests", dependencies: ["WorkshopCore"], path: "Tests/WorkshopCoreTests"),
        .testTarget(name: "LoomWorkshopUITests", dependencies: ["WorkshopCore", "LoomWorkshop"], path: "Tests/LoomWorkshopUITests")
    ]
)
