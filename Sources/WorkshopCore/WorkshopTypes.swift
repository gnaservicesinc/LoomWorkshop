import Foundation

public struct BaseDataset: Codable, Sendable, Identifiable {
    public let id: String
    public let name: String
    public let version: String
    public let mapKinds: [String]
}
public struct StudioBase: Codable, Sendable {
    public let applicationURL: URL
    public let modelID: String
    public let modelName: String
    public let version: String
    public let categories: [BaseDataset]
    public let packageURL: URL
}
public struct WorkshopDataset: Codable, Sendable, Identifiable {
    public let id: String
    public let name: String
    public let imageCount: Int
    public let pairedCounts: [String: Int]
}
public struct WorkshopOptions: Codable, Sendable, Equatable {
    public var name = "My Texture LoRA"
    public var datasetIDs: [String] = []
    public var epochs = 5
    public var imageSize = 1024
    public var learningRate = 0.0001
    public var rank = 8
    public var shuffleEachEpoch = true
    public var flipCrops = true
    public var exaggerateFeatures = false
    public init() {}
}
public struct WorkshopPlan: Codable, Sendable {
    public let sourceFolder: URL
    public let studio: StudioBase
    public let datasets: [WorkshopDataset]
    public let requiresClassification: Bool
    public let options: WorkshopOptions
    public let imageCount: Int
}
public struct WorkshopEvent: Sendable {
    public let message: String
    public let progress: Double?
    public let epoch: Int?
    public let loss: Double?
    public init(message: String, progress: Double? = nil, epoch: Int? = nil, loss: Double? = nil) {
        self.message = message; self.progress = progress; self.epoch = epoch; self.loss = loss
    }
}
public struct WorkshopError: LocalizedError, Sendable {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var errorDescription: String? { message }
}
