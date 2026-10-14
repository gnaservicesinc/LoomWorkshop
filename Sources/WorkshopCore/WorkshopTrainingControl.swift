import Foundation
import LoomNative

/// Requests are handled on the GPU worker after a complete optimizer update.
/// Setting a request never cancels an in-progress update or interrupts a save.
final class WorkshopTrainingControl: @unchecked Sendable {
    private let lock = NSLock()
    private var queued: [(UUID, Bool)] = []
    private var waiters: [UUID: CheckedContinuation<URL, Error>] = [:]
    private var results: [UUID: Result<URL, Error>] = [:]
    private var ended: Error?

    func requestCheckpoint(stopAfter: Bool) -> UUID {
        lock.withLock {
            let id = UUID()
            if let ended { results[id] = .failure(ended) }
            else { queued.append((id, stopAfter)) }
            return id
        }
    }

    func wait(for id: UUID) async throws -> URL {
        try await withCheckedThrowingContinuation { continuation in
            let result: Result<URL, Error>? = lock.withLock {
                if let result = results.removeValue(forKey: id) { return result }
                waiters[id] = continuation
                return nil
            }
            if let result { continuation.resume(with: result) }
        }
    }

    func takeRequest() -> (ids: [UUID], stop: Bool)? {
        lock.withLock {
            guard !queued.isEmpty else { return nil }
            let result = (queued.map(\.0), queued.contains { $0.1 })
            queued = []
            return result
        }
    }

    func complete(_ ids: [UUID], result: Result<URL, Error>) {
        let pending = lock.withLock {
            ids.compactMap { id -> CheckedContinuation<URL, Error>? in
                if let waiter = waiters.removeValue(forKey: id) { return waiter }
                results[id] = result
                return nil
            }
        }
        for waiter in pending { waiter.resume(with: result) }
    }

    func finish() {
        let error = WorkshopError("Training finished before another checkpoint could be saved. Open the run folder to see its saved results.")
        let ids = lock.withLock { ended = error; return queued.map(\.0) }
        complete(ids, result: .failure(error))
    }
}

public struct WorkshopSavedStop: LocalizedError, Sendable {
    public let checkpoint: URL
    public var errorDescription: String? { "Training stopped; the latest completed update was saved." }
}

public struct WorkshopSavedRun: Sendable, Identifiable {
    public var id: String { directory.lastPathComponent }
    public let directory: URL
    public let name: String
    public let options: WorkshopOptions
    public let updates: Int
    public let modified: Date
}

struct WorkshopRunRequest: Codable {
    var schema = "loom-workshop-run-v1"
    let options: WorkshopOptions
    let baseID: String
    let baseVersion: String
    let sourceSignature: String
}

struct WorkshopCheckpointState: Decodable {
    let schema: String
    let epoch: Int
    let updates: Int
    let epochCompleted: Bool
    let nextBatch: Int
    let epochOrder: [Int]
    let baseID: String
    let baseVersion: String
    let categoryID: String
    let categoryVersion: String
    let map: String
    let baseSHA256: String
    let finalLoss: Double
    let options: WorkshopOptions
    let weightsPath: String
    let optimizerPath: String
    let weightsSHA256: String
    let optimizerSHA256: String
    let samplingState: LoomNative.NativeAugmentationPlan
    let orderState: String

    enum CodingKeys: String, CodingKey {
        case schema, epoch, updates, map, options
        case epochCompleted = "epoch_completed", nextBatch = "next_batch", epochOrder = "epoch_order"
        case baseID = "base_id", baseVersion = "base_version", categoryID = "category_id", categoryVersion = "category_version"
        case baseSHA256 = "base_sha256", finalLoss = "final_loss", weightsPath = "weights_path", optimizerPath = "optimizer_path"
        case weightsSHA256 = "weights_sha256", optimizerSHA256 = "optimizer_sha256", samplingState = "sampling_state", orderState = "order_state"
    }
}
