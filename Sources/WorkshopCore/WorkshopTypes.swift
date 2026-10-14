import Foundation
import LoomNative

public struct BaseDataset: Codable, Sendable, Identifiable {
    public let id: String
    public let name: String
    public let version: String
    public let mapKinds: [String]
    public var modelProfiles: [BaseModelProfile]? = nil
}
public struct BaseModelProfile: Codable, Sendable, Identifiable {
    public let map: String
    public let width: Int
    public let blocks: Int
    public var architecture: String? = nil
    public var id: String { map }
    public var baseParameters: Int {
        guard (4...512).contains(width), (1...64).contains(blocks) else { return 0 }
        return (try? NativeArchitecture(mapKind: map, width: width, blocks: blocks,
            identifier: architecture ?? NativeArchitecture.identifier).parameterCount) ?? 0
    }
    public func trainableParameters(rank: Int) -> Int {
        guard (1...128).contains(rank), (4...512).contains(width), (1...64).contains(blocks) else { return 0 }
        guard let network = try? NativeArchitecture(mapKind: map, width: width, blocks: blocks,
            identifier: architecture ?? NativeArchitecture.identifier) else { return 0 }
        return network.convolutionNames.reduce(0) { count, name in
            let shape = network.parameterShapes[name + ".weight"]!
            return count + rank * (shape[1] * shape[2] * shape[3] + shape[0])
        }
    }
}
public enum WorkshopOptimizer: String, Codable, Sendable, CaseIterable, Identifiable {
    case adamW = "adamw", adam, sgd, lion
    public var id: String { rawValue }
    public var title: String {
        switch self { case .adamW: "AdamW"; case .adam: "Adam"; case .sgd: "SGD"; case .lion: "Lion" }
    }
}
public enum WorkshopRefresh: String, Codable, Sendable, CaseIterable, Identifiable {
    case sample = "per_sample", epoch = "per_epoch", run = "per_run"
    public var id: String { rawValue }
    public var title: String {
        switch self { case .sample: "Every presentation"; case .epoch: "Once per epoch"; case .run: "Once per run" }
    }
}
public enum WorkshopSchedule: String, Codable, Sendable, CaseIterable, Identifiable {
    case constant, cosine
    public var id: String { rawValue }
    public var title: String { self == .constant ? "Constant" : "Cosine decay" }
}
public enum WorkshopDevice: String, Codable, Sendable, CaseIterable, Identifiable {
    case automatic = "auto", gpu = "mps", cpu
    public var id: String { rawValue }
    public var title: String {
        switch self { case .automatic: "Automatic (Apple GPU)"; case .gpu: "Apple GPU"; case .cpu: "CPU (unavailable)" }
    }
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
    public var alpha = 8.0
    public var shuffleEachEpoch = true
    public var flipCrops = true
    public var rotateCrops = false
    public var exaggerateFeatures = false
    public var batchSize = 1
    public var gradientAccumulationSteps = 1
    public var optimizer = WorkshopOptimizer.adamW
    public var weightDecay = 0.01
    public var beta1 = 0.9
    public var beta2 = 0.999
    public var momentum = 0.9
    public var maxGradientNorm = 1.0
    public var cropRefresh = WorkshopRefresh.sample
    public var augmentationRefresh = WorkshopRefresh.sample
    public var learningRateSchedule = WorkshopSchedule.constant
    public var warmupEpochs = 0
    public var minimumLearningRateRatio = 0.1
    public var seed = 42
    public var checkpointEveryEpochs = 1
    public var device = WorkshopDevice.automatic
    public var effectiveBatchSize: Int {
        let product = batchSize.multipliedReportingOverflow(by: gradientAccumulationSteps)
        return product.overflow ? Int.max : product.partialValue
    }
    public init() {}
    public init(from decoder: Decoder) throws {
        self.init()
        let values = try decoder.container(keyedBy: CodingKeys.self)
        name = try values.decodeIfPresent(String.self, forKey: .name) ?? name
        datasetIDs = try values.decodeIfPresent([String].self, forKey: .datasetIDs) ?? datasetIDs
        epochs = try values.decodeIfPresent(Int.self, forKey: .epochs) ?? epochs
        imageSize = try values.decodeIfPresent(Int.self, forKey: .imageSize) ?? imageSize
        learningRate = try values.decodeIfPresent(Double.self, forKey: .learningRate) ?? learningRate
        rank = try values.decodeIfPresent(Int.self, forKey: .rank) ?? rank
        alpha = try values.decodeIfPresent(Double.self, forKey: .alpha) ?? Double(rank)
        shuffleEachEpoch = try values.decodeIfPresent(Bool.self, forKey: .shuffleEachEpoch) ?? shuffleEachEpoch
        flipCrops = try values.decodeIfPresent(Bool.self, forKey: .flipCrops) ?? flipCrops
        rotateCrops = try values.decodeIfPresent(Bool.self, forKey: .rotateCrops) ?? rotateCrops
        exaggerateFeatures = try values.decodeIfPresent(Bool.self, forKey: .exaggerateFeatures) ?? exaggerateFeatures
        batchSize = try values.decodeIfPresent(Int.self, forKey: .batchSize) ?? batchSize
        gradientAccumulationSteps = try values.decodeIfPresent(Int.self, forKey: .gradientAccumulationSteps) ?? gradientAccumulationSteps
        optimizer = try values.decodeIfPresent(WorkshopOptimizer.self, forKey: .optimizer) ?? optimizer
        weightDecay = try values.decodeIfPresent(Double.self, forKey: .weightDecay) ?? weightDecay
        beta1 = try values.decodeIfPresent(Double.self, forKey: .beta1) ?? beta1
        beta2 = try values.decodeIfPresent(Double.self, forKey: .beta2) ?? (optimizer == .lion ? 0.99 : beta2)
        momentum = try values.decodeIfPresent(Double.self, forKey: .momentum) ?? momentum
        maxGradientNorm = try values.decodeIfPresent(Double.self, forKey: .maxGradientNorm) ?? maxGradientNorm
        cropRefresh = try values.decodeIfPresent(WorkshopRefresh.self, forKey: .cropRefresh) ?? cropRefresh
        augmentationRefresh = try values.decodeIfPresent(WorkshopRefresh.self, forKey: .augmentationRefresh) ?? augmentationRefresh
        learningRateSchedule = try values.decodeIfPresent(WorkshopSchedule.self, forKey: .learningRateSchedule) ?? learningRateSchedule
        warmupEpochs = try values.decodeIfPresent(Int.self, forKey: .warmupEpochs) ?? warmupEpochs
        minimumLearningRateRatio = try values.decodeIfPresent(Double.self, forKey: .minimumLearningRateRatio) ?? minimumLearningRateRatio
        seed = try values.decodeIfPresent(Int.self, forKey: .seed) ?? seed
        checkpointEveryEpochs = try values.decodeIfPresent(Int.self, forKey: .checkpointEveryEpochs) ?? checkpointEveryEpochs
        device = try values.decodeIfPresent(WorkshopDevice.self, forKey: .device) ?? device
    }
    public func validate() throws {
        let optimizerCoefficientsValid = optimizer == .sgd ? (0..<1).contains(momentum) :
            ((0..<1).contains(beta1) && (0..<1).contains(beta2))
        guard !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              (32...8192).contains(imageSize), (1...100_000).contains(epochs), (1...128).contains(rank),
              learningRate.isFinite, learningRate > 0, learningRate <= 1, Set(datasetIDs).count == datasetIDs.count,
              alpha.isFinite, alpha > 0,
              (1...256).contains(batchSize), (1...4096).contains(gradientAccumulationSteps),
              weightDecay.isFinite, (0...10).contains(weightDecay), beta1.isFinite,
              beta2.isFinite, momentum.isFinite, optimizerCoefficientsValid,
              maxGradientNorm.isFinite, (0...1_000_000).contains(maxGradientNorm), warmupEpochs >= 0, warmupEpochs < epochs,
              (1...100_000).contains(checkpointEveryEpochs),
              minimumLearningRateRatio.isFinite, (0...1).contains(minimumLearningRateRatio),
              (0...Int(Int32.max)).contains(seed) else {
            throw WorkshopError("Check the training settings. Batch size, accumulation and rank must be positive; learning rate must be positive; decay and gradient clipping cannot be negative; optimizer coefficients must be below 1.")
        }
    }
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
    public let update: Int?
    public let totalUpdates: Int?
    public let learningRate: Double?
    public init(message: String, progress: Double? = nil, epoch: Int? = nil, loss: Double? = nil,
                update: Int? = nil, totalUpdates: Int? = nil, learningRate: Double? = nil) {
        self.message = message; self.progress = progress; self.epoch = epoch; self.loss = loss
        self.update = update; self.totalUpdates = totalUpdates; self.learningRate = learningRate
    }
}
public struct WorkshopError: LocalizedError, Sendable {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var errorDescription: String? { message }
}
