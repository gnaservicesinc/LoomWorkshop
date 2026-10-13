import AppKit
import Testing
@testable import LoomWorkshop

@Suite(.serialized)
struct WorkshopApplicationTerminationTests {
    @Test @MainActor
    func terminationReadsCurrentOperationStateWithoutStoppingTheOperation() async throws {
        let model = WorkshopModel(rememberLocations: false)
        let delegate = WorkshopApplicationDelegate()
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
