import AppKit
import Foundation

let resources = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
let icons = resources.appendingPathComponent("LoomWorkshop.iconset", isDirectory: true)
try FileManager.default.createDirectory(at: icons, withIntermediateDirectories: true)
let variants = [(16, "16x16"), (32, "16x16@2x"), (32, "32x32"), (64, "32x32@2x"),
                (128, "128x128"), (256, "128x128@2x"), (256, "256x256"), (512, "256x256@2x"),
                (512, "512x512"), (1024, "512x512@2x")]
for (pixels, name) in variants {
    let scale = CGFloat(pixels) / 1024
    let image = NSImage(size: NSSize(width: pixels, height: pixels))
    image.lockFocus()
    let outline = NSBezierPath(roundedRect: NSRect(x: 40 * scale, y: 40 * scale, width: 944 * scale, height: 944 * scale), xRadius: 210 * scale, yRadius: 210 * scale)
    NSGradient(starting: NSColor(srgbRed: 0.08, green: 0.38, blue: 0.39, alpha: 1), ending: NSColor(srgbRed: 0.21, green: 0.73, blue: 0.62, alpha: 1))!.draw(in: outline, angle: 70)
    for row in 0..<3 {
        for column in 0..<3 {
            NSColor.white.withAlphaComponent(row == column ? 1 : 0.45).setFill()
            NSBezierPath(roundedRect: NSRect(x: CGFloat(276 + column * 174) * scale, y: CGFloat(276 + row * 174) * scale, width: 126 * scale, height: 126 * scale), xRadius: 24 * scale, yRadius: 24 * scale).fill()
        }
    }
    image.unlockFocus()
    guard let tiff = image.tiffRepresentation, let bitmap = NSBitmapImageRep(data: tiff), let png = bitmap.representation(using: .png, properties: [:]) else { fatalError("Could not render the app icon") }
    try png.write(to: icons.appendingPathComponent("icon_\(name).png"))
}
let compiler = Process()
compiler.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
compiler.arguments = ["-c", "icns", icons.path, "-o", resources.appendingPathComponent("LoomWorkshop.icns").path]
try compiler.run(); compiler.waitUntilExit()
guard compiler.terminationStatus == 0 else { fatalError("Could not compile the app icon") }
try FileManager.default.removeItem(at: icons)
