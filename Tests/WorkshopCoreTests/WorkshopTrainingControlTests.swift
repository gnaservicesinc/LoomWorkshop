import Foundation
import XCTest
@testable import WorkshopCore

final class WorkshopTrainingControlTests: XCTestCase, @unchecked Sendable {
    func testCoalescedSaveAndStopReceiptsResumeAllWaitersAfterDurableSave() async throws {
        let control = WorkshopTrainingControl()
        let first = control.requestCheckpoint(stopAfter: false)
        let second = control.requestCheckpoint(stopAfter: true)
        let waiting = Task { try await control.wait(for: first) }
        let request = try XCTUnwrap(control.takeRequest())
        XCTAssertTrue(request.stop)
        XCTAssertEqual(Set(request.ids), [first, second])
        let url = URL(fileURLWithPath: "/tmp/saved-checkpoint.json")
        control.complete(request.ids, result: .success(url))
        let firstResult = try await waiting.value
        XCTAssertEqual(firstResult, url)
        let secondResult = try await control.wait(for: second)
        XCTAssertEqual(secondResult, url)
        XCTAssertNil(control.takeRequest())
    }

    func testFailedSaveReceiptAllowsRetryWithoutEndingTraining() async throws {
        let control = WorkshopTrainingControl()
        let first = control.requestCheckpoint(stopAfter: true)
        let request = try XCTUnwrap(control.takeRequest())
        control.complete(request.ids, result: .failure(WorkshopError("Disk full")))
        do { _ = try await control.wait(for: first); XCTFail("A failed save must fail its receipt.") }
        catch { XCTAssertEqual(error.localizedDescription, "Disk full") }
        let retry = control.requestCheckpoint(stopAfter: false)
        let retried = try XCTUnwrap(control.takeRequest())
        XCTAssertFalse(retried.stop)
        let url = URL(fileURLWithPath: "/tmp/retried-checkpoint.json")
        control.complete(retried.ids, result: .success(url))
        let saved = try await control.wait(for: retry)
        XCTAssertEqual(saved, url)
    }

    func testTrainingEndFailsPendingAndLateSaveReceiptsPromptly() async throws {
        let control = WorkshopTrainingControl()
        let pending = control.requestCheckpoint(stopAfter: true)
        control.finish()
        let late = control.requestCheckpoint(stopAfter: false)
        for id in [pending, late] {
            do { _ = try await control.wait(for: id); XCTFail("An ended run cannot fulfill a pending save.") }
            catch { XCTAssertTrue(error.localizedDescription.contains("Training finished")) }
        }
    }
}
