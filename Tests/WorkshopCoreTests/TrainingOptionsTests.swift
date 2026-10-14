import Foundation
import XCTest
@testable import WorkshopCore

final class TrainingOptionsTests: XCTestCase {
    func testLegacyOptionsRestoreNewDefaultsAndPreserveRankScaling() throws {
        let data = Data(#"{"name":"Old LoRA","rank":4,"flipCrops":false,"imageSize":512}"#.utf8)
        let old = try JSONDecoder().decode(WorkshopOptions.self, from: data)
        XCTAssertEqual(old.name, "Old LoRA")
        XCTAssertEqual(old.rank, 4); XCTAssertEqual(old.alpha, 4)
        XCTAssertFalse(old.flipCrops); XCTAssertEqual(old.imageSize, 512)
        XCTAssertEqual(old.optimizer, .adamW); XCTAssertEqual(old.batchSize, 1)
        XCTAssertEqual(old.gradientAccumulationSteps, 1); XCTAssertEqual(old.cropRefresh, .sample)
        XCTAssertEqual(old.augmentationRefresh, .sample); XCTAssertEqual(old.device, .automatic)
        try old.validate()
        let lion = try JSONDecoder().decode(WorkshopOptions.self, from: Data(#"{"optimizer":"lion"}"#.utf8))
        XCTAssertEqual(lion.beta2, 0.99)
        let explicitLion = try JSONDecoder().decode(WorkshopOptions.self, from: Data(#"{"optimizer":"lion","beta2":0.995}"#.utf8))
        XCTAssertEqual(explicitLion.beta2, 0.995)
    }

    func testAllTrainingSettingsRoundTripAndInvalidValuesAreRejected() throws {
        var options = WorkshopOptions()
        options.optimizer = .lion; options.batchSize = 2; options.gradientAccumulationSteps = 8
        options.weightDecay = 0.05; options.beta1 = 0.85; options.beta2 = 0.99; options.momentum = 0.8
        options.maxGradientNorm = 0; options.cropRefresh = .run; options.augmentationRefresh = .epoch
        options.learningRateSchedule = .cosine; options.warmupEpochs = 2
        options.minimumLearningRateRatio = 0.01; options.rotateCrops = true
        options.seed = 9876; options.alpha = 16; options.device = .cpu; options.checkpointEveryEpochs = 3
        try options.validate()
        XCTAssertEqual(options.effectiveBatchSize, 16)
        XCTAssertEqual(try JSONDecoder().decode(WorkshopOptions.self, from: JSONEncoder().encode(options)), options)
        options.gradientAccumulationSteps = 0
        XCTAssertThrowsError(try options.validate())
        options.gradientAccumulationSteps = 8; options.beta2 = 1
        XCTAssertThrowsError(try options.validate())
        options.beta2 = 0.99; options.warmupEpochs = options.epochs + 1
        XCTAssertThrowsError(try options.validate())
        options.warmupEpochs = 0; options.learningRate = .nan
        XCTAssertThrowsError(try options.validate())
    }

    func testRuntimeAlignedBoundaryValuesCannotReachSurpriseErrors() throws {
        var options = WorkshopOptions()
        options.learningRate = 1; options.gradientAccumulationSteps = 4096
        options.maxGradientNorm = 1_000_000; options.seed = Int(Int32.max)
        options.warmupEpochs = options.epochs - 1; options.weightDecay = 10
        try options.validate()
        options.warmupEpochs = options.epochs
        XCTAssertThrowsError(try options.validate())
        options.warmupEpochs = 0; options.seed = Int(Int32.max) + 1
        XCTAssertThrowsError(try options.validate())
        options.seed = 42; options.maxGradientNorm = 1_000_001
        XCTAssertThrowsError(try options.validate())
        options.maxGradientNorm = 1; options.learningRate = 1.00001
        XCTAssertThrowsError(try options.validate())
        options.learningRate = 0.0001; options.gradientAccumulationSteps = 4097
        XCTAssertThrowsError(try options.validate())
        options.gradientAccumulationSteps = 1; options.epochs = 1; options.warmupEpochs = 0
        try options.validate()
        options.epochs = Int.min
        XCTAssertThrowsError(try options.validate())
    }

    func testParameterPreviewCountsAllFrozenConvolutionsAndTrainableDeltas() {
        let depth = BaseModelProfile(map: "depth", width: 24, blocks: 4)
        let normal = BaseModelProfile(map: "normal", width: 24, blocks: 4)
        XCTAssertEqual(depth.baseParameters, 42_553)
        XCTAssertEqual(depth.trainableParameters(rank: 8), 17_504)
        XCTAssertEqual(normal.baseParameters, 42_987)
        XCTAssertEqual(normal.trainableParameters(rank: 8), 17_520)
        XCTAssertEqual(depth.trainableParameters(rank: 16), 2 * depth.trainableParameters(rank: 8))
        XCTAssertEqual(depth.trainableParameters(rank: Int.max), 0)
        XCTAssertEqual(depth.trainableParameters(rank: Int.min), 0)
        let malformed = BaseModelProfile(map: "depth", width: Int.max, blocks: Int.max)
        XCTAssertEqual(malformed.baseParameters, 0)
        XCTAssertEqual(malformed.trainableParameters(rank: 8), 0)
    }
}
