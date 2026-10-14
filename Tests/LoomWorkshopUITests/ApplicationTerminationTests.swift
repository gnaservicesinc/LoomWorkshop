import AppKit
import Testing
@testable import LoomWorkshop

@Suite(.serialized)
struct ApplicationTerminationTests {
    @Test @MainActor
    func keepTrainingRefusesQuitWithoutStopping() {
        let delegate = WorkshopApplicationDelegate()
        delegate.operationIsActive = { true }
        delegate.trainingIsActive = { true }
        var askedAboutTraining = false
        var stopped = false
        delegate.quitDecision = { training in askedAboutTraining = training; return false }
        delegate.finishActiveOperation = { stopped = true; return true }
        #expect(delegate.applicationShouldTerminate(NSApplication.shared) == .terminateCancel)
        #expect(askedAboutTraining)
        #expect(!stopped)
    }

    @Test @MainActor
    func saveAndQuitWaitsForSaveThenFlushesSettings() async throws {
        let delegate = WorkshopApplicationDelegate()
        var active = true
        var saved = false
        var prepared = false
        var reply: Bool?
        delegate.operationIsActive = { active }
        delegate.trainingIsActive = { true }
        delegate.quitDecision = { _ in true }
        delegate.finishActiveOperation = {
            try? await Task.sleep(for: .milliseconds(25))
            saved = true; active = false
            return true
        }
        delegate.prepareForTermination = { prepared = saved; return true }
        delegate.replyToTermination = { _, allowed in reply = allowed }
        #expect(delegate.applicationShouldTerminate(NSApplication.shared) == .terminateLater)
        #expect(reply == nil)
        #expect(delegate.applicationShouldTerminate(NSApplication.shared) == .terminateLater)
        for _ in 0..<100 where reply == nil { try await Task.sleep(for: .milliseconds(5)) }
        #expect(reply == true)
        #expect(saved && prepared)
    }

    @Test @MainActor
    func failedCheckpointRefusesQuitAndKeepsTraining() async throws {
        let delegate = WorkshopApplicationDelegate()
        var reply: Bool?
        var flushed = false
        delegate.operationIsActive = { true }
        delegate.trainingIsActive = { true }
        delegate.quitDecision = { _ in true }
        delegate.finishActiveOperation = { false }
        delegate.prepareForTermination = { flushed = true; return true }
        delegate.replyToTermination = { _, allowed in reply = allowed }
        #expect(delegate.applicationShouldTerminate(NSApplication.shared) == .terminateLater)
        for _ in 0..<100 where reply == nil { try await Task.sleep(for: .milliseconds(5)) }
        #expect(reply == false)
        #expect(!flushed)
    }
}

@Suite(.serialized)
struct WorkshopApplicationTerminationTests {
    @Test @MainActor
    func terminationReadsCurrentOperationStateWithoutStoppingTheOperation() async throws {
        let model = WorkshopModel(rememberLocations: false)
        let delegate = WorkshopApplicationDelegate()
        delegate.quitDecision = { _ in false }
        delegate.bind(to: model)
        var replies: [Bool] = []
        delegate.replyToTermination = { _, allowed in replies.append(allowed) }

        let missing = FileManager.default.temporaryDirectory.appendingPathComponent("missing-studio-\(UUID().uuidString).app", isDirectory: true)
        model.inspectStudio(missing)
        #expect(model.busy)
        #expect(delegate.applicationShouldTerminate(NSApplication.shared) == .terminateCancel)
        #expect(model.busy)
        for _ in 0..<100 {
            if !model.busy { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(!model.busy)
        #expect(model.studioIssue != nil)
        #expect(delegate.applicationShouldTerminate(NSApplication.shared) == .terminateLater)
        for _ in 0..<100 {
            if !replies.isEmpty { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(replies == [true])
    }

    @Test @MainActor
    func terminationDelegateDoesNotKeepTheWorkspaceAlive() async {
        let delegate = WorkshopApplicationDelegate()
        delegate.quitDecision = { _ in false }
        var model: WorkshopModel? = WorkshopModel(rememberLocations: false)
        weak var workspace = model
        delegate.bind(to: model!)
        model = nil
        #expect(workspace == nil)
        #expect(!delegate.operationIsActive())
        #expect(await delegate.prepareForTermination?() == true)
    }

    @Test @MainActor
    func idleTerminationWaitsForOneSaveAndRechecksForNewWork() async throws {
        let delegate = WorkshopApplicationDelegate()
        delegate.quitDecision = { _ in false }
        #expect(delegate.applicationShouldTerminate(NSApplication.shared) == .terminateNow)
        var active = false
        var preparations = 0
        var replies: [Bool] = []
        delegate.operationIsActive = { active }
        delegate.replyToTermination = { _, allowed in replies.append(allowed) }
        delegate.prepareForTermination = {
            preparations += 1
            try? await Task.sleep(for: .milliseconds(40))
            active = true
            return true
        }
        #expect(delegate.applicationShouldTerminate(NSApplication.shared) == .terminateLater)
        #expect(delegate.applicationShouldTerminate(NSApplication.shared) == .terminateLater)
        for _ in 0..<100 {
            if replies.count == 1 { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(preparations == 1)
        #expect(replies == [false])
        #expect(delegate.applicationShouldTerminate(NSApplication.shared) == .terminateCancel)
        active = false
        delegate.prepareForTermination = { false }
        #expect(delegate.applicationShouldTerminate(NSApplication.shared) == .terminateLater)
        for _ in 0..<100 {
            if replies.count == 2 { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(replies == [false, false])
    }
}
