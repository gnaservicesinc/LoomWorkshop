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
            CommandGroup(replacing: .saveItem) {
                Button("Save Training Checkpoint") { model.saveCheckpoint() }
                    .keyboardShortcut("s").disabled(!model.trainingActive || model.checkpointPending)
                Button("Save & Stop Training") { model.saveAndStop() }
                    .disabled(!model.trainingActive || model.stoppingAfterSave)
                Button("Stop Current Operation") { model.cancel() }
                    .disabled(!model.busy || model.stoppingAfterSave)
            }
        }
    }
}

@MainActor
final class WorkshopApplicationDelegate: NSObject, NSApplicationDelegate {
    var operationIsActive: () -> Bool = { false }
    var trainingIsActive: () -> Bool = { false }
    var finishActiveOperation: (() async -> Bool)?
    var quitDecision: (Bool) -> Bool = { training in
        let alert = NSAlert()
        alert.messageText = training ? "Save your training before quitting?" : "Stop the current operation and quit?"
        alert.informativeText = training ? "Save & Quit writes the latest completed update and waits for the checkpoint to finish. You can resume this run later." : "Loom Workshop will wait for the operation to stop before quitting."
        alert.addButton(withTitle: training ? "Keep Training" : "Keep Working")
        alert.addButton(withTitle: training ? "Save & Quit" : "Stop & Quit")
        return alert.runModal() == .alertSecondButtonReturn
    }
    var prepareForTermination: (() async -> Bool)?
    var replyToTermination: (NSApplication, Bool) -> Void = { application, allowed in
        application.reply(toApplicationShouldTerminate: allowed)
    }
    private var terminationPending = false

    func bind(to model: WorkshopModel) {
        operationIsActive = { [weak model] in model?.busy ?? false }
        trainingIsActive = { [weak model] in model?.trainingActive ?? false }
        finishActiveOperation = { [weak model] in
            guard let model else { return true }
            return model.trainingActive ? await model.saveAndStopAndWait() : await model.cancelAndWait()
        }
        prepareForTermination = { [weak model] in
            guard let model else { return true }
            return await model.prepareForTermination()
        }
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard !terminationPending else { return .terminateLater }
        if operationIsActive() {
            guard quitDecision(trainingIsActive()), let finishActiveOperation else { return .terminateCancel }
            terminationPending = true
            Task { @MainActor [weak self] in
                let finished = await finishActiveOperation()
                guard let self else { return }
                let prepared = finished && !self.operationIsActive() ? await self.prepareForTermination?() ?? true : false
                self.terminationPending = false
                self.replyToTermination(sender, prepared && !self.operationIsActive())
            }
            return .terminateLater
        }
        guard let prepareForTermination else { return .terminateNow }
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
