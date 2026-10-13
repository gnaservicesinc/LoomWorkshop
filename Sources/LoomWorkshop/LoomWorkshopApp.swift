import AppKit
import SwiftUI

@main
struct LoomWorkshopApp: App {
    @NSApplicationDelegateAdaptor(WorkshopApplicationDelegate.self) private var applicationDelegate
    @State private var model = WorkshopModel()
    var body: some Scene {
        WindowGroup("Loom Workshop") {
            WorkshopView(model: model).frame(minWidth: 1000, minHeight: 720).tint(WorkshopStyle.accent)
                .onAppear { applicationDelegate.bind(to: model) }
        }
        .defaultSize(width: 1180, height: 940)
        .windowToolbarStyle(.unified)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Choose Image Folder…") { model.chooseSourceFolder() }
                    .keyboardShortcut("o").disabled(model.studio == nil || model.busy)
            }
        }
    }
}

@MainActor
final class WorkshopApplicationDelegate: NSObject, NSApplicationDelegate {
    var operationIsActive: () -> Bool = { false }
    var prepareForTermination: (() async -> Bool)?
    var replyToTermination: (NSApplication, Bool) -> Void = { application, allowed in
        application.reply(toApplicationShouldTerminate: allowed)
    }
    private var terminationPending = false

    func bind(to model: WorkshopModel) {
        operationIsActive = { [weak model] in model?.busy ?? false }
        prepareForTermination = { [weak model] in
            guard let model else { return true }
            return await model.prepareForTermination()
        }
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard !operationIsActive() else {
            NSLog("Loom Workshop is busy. Stop the current operation before quitting.")
            return .terminateCancel
        }
        guard let prepareForTermination else { return .terminateNow }
        guard !terminationPending else { return .terminateLater }
        terminationPending = true
        Task { @MainActor [weak self] in
            let prepared = await prepareForTermination()
            guard let self else { return }
            self.terminationPending = false
            self.replyToTermination(sender, prepared && !self.operationIsActive())
        }
        return .terminateLater
    }
}
