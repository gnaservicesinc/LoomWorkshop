import CoreML
import CryptoKit
import Foundation
import ImageIO
import LoomNative

private struct NativeWorkshopPair: Sendable {
    let sample: WorkshopSample
    let category: BaseManifest.Category
    let artifact: BaseManifest.Model
    let target: URL
    let width: Int
    let height: Int
}
private struct ImageFingerprint: Sendable {
    let bytes: UInt64
    let modified: TimeInterval
    let sha256: String
    let membership: Data
}
/// Native operations run in a detached task; the project actor remains available to cancel them.
struct NativeWorkshopEngine: Sendable {
    let source: URL
    let project: URL
    let application: URL
    let options: WorkshopOptions
    let cancellation: NativeCancellation
    let progress: @Sendable (WorkshopEvent) -> Void

    func plan() throws {
        try cancellation.check()
        let (studio, manifest) = try readStudio(application, cancellation: cancellation)
        let samples = try currentSamples(source: source, studio: studio, imageSize: options.imageSize)
        _ = try classify(samples, studio: studio, manifest: manifest)
    }

    func train(run: URL) throws -> URL {
        defer { NativeImages.clearDecodedPixelCache() }
        var updates = 0
        var saveStoppedUpdate: (() throws -> Void)?
        do {
            try options.validate()
            try cancellation.check()
            let (studio, manifest) = try readStudio(application, cancellation: cancellation)
            let originalBaseManifest = try Data(contentsOf: studio.packageURL.appendingPathComponent("manifest.json"))
            let samples = try currentSamples(source: source, studio: studio, imageSize: options.imageSize)
            guard !samples.isEmpty else { throw WorkshopError("This folder has no diffuse images. Add diffuse images with matching material maps.") }
            guard Set(options.datasetIDs).isSubset(of: Set(manifest.categories.map(\.id))) else {
                throw WorkshopError("A selected dataset is absent from this Texture Studio build.")
            }
            let groups = try classify(samples, studio: studio, manifest: manifest)
            guard !groups.isEmpty else {
                throw WorkshopError("These images have no matching paired target maps for the installed model. Add matching depth, roughness or normal maps; photos alone cannot train a supervised material LoRA.")
            }
            let paths = Set(groups.values.flatMap { $0.flatMap { [$0.sample.input, $0.target] } })
            var fingerprints: [URL: ImageFingerprint] = [:]
            for path in paths {
                try cancellation.check()
                let state = try identity(path)
                fingerprints[path] = ImageFingerprint(bytes: state.0, modified: state.1, sha256: try fileSHA(path, cancellation: cancellation), membership: try membershipSignature(path))
            }
            let package = run.appendingPathComponent("adapter", isDirectory: true)
            try FileManager.default.createDirectory(at: package, withIntermediateDirectories: false)
            var adapters: [String: [[String: String]]] = [:]
            var sampler = NativeAugmentationPlan(seed: options.seed, cropRefresh: options.cropRefresh.rawValue,
                                                  augmentationRefresh: options.augmentationRefresh.rawValue)
            var orderGenerator = WorkshopOrderGenerator(seed: UInt64(options.seed))
            let totalUpdates = options.epochs * groups.values.reduce(0) { $0 + Self.stepsPerEpoch(samples: $1.count, options: options) }
            emit(.init(message: "Native Apple GPU LoRA training started", update: 0, totalUpdates: totalUpdates), run: run)
            for key in groups.keys.sorted() {
                try cancellation.check()
                let members = groups[key]!
                let first = members[0], category = first.category, artifact = first.artifact
                guard [NativeArchitecture.identifier, NativeArchitecture.pyramidIdentifier].contains(artifact.architecture) else {
                    throw WorkshopError("This custom base architecture needs a compatible Loom Workshop version.")
                }
                let baseWeightsURL = try modelPath(artifact.weightsPath, in: studio.packageURL)
                let baseHash = try fileSHA(baseWeightsURL, cancellation: cancellation)
                let descriptorURL = baseWeightsURL.deletingLastPathComponent().appendingPathComponent("model.json")
                guard let descriptor = try JSONSerialization.jsonObject(with: Data(contentsOf: descriptorURL)) as? [String: Any],
                      (descriptor["architecture"] as? String).map({ $0 == artifact.architecture }) ?? true,
                      let config = (descriptor["architecture_config"] ?? descriptor["architecture"]) as? [String: Any],
                      config["id"] as? String == artifact.architecture,
                      let width = config["width"] as? Int, let blocks = config["blocks"] as? Int else {
                    throw WorkshopError("The installed base is missing its LoRA architecture configuration.")
                }
                let architecture = try NativeArchitecture(mapKind: artifact.map, width: width, blocks: blocks, identifier: artifact.architecture)
                let frozen = try NativeWeights.read(from: baseWeightsURL)
                let nativeOptions = try NativeTrainingOptions(optimizer: options.optimizer.rawValue, weightDecay: options.weightDecay,
                    beta1: options.optimizer == .sgd ? 0.9 : options.beta1,
                    beta2: options.optimizer == .sgd ? 0.999 : options.beta2,
                    momentum: options.optimizer == .sgd ? options.momentum : 0.9, gradientClipNorm: options.maxGradientNorm)
                let session = try NativeTrainingSession(architecture: architecture, options: nativeOptions, seed: UInt64(options.seed),
                    weights: frozen, adapter: NativeAdapterConfiguration(rank: options.rank, alpha: options.alpha),
                    device: options.device.rawValue, cancellation: cancellation, channelsLast: !architecture.isPyramid)
                let stepsPerEpoch = Self.stepsPerEpoch(samples: members.count, options: options)
                let groupTotal = stepsPerEpoch * options.epochs
                let baseParameters = frozen.values.reduce(0) { $0 + $1.values.count }
                let trainableParameters = try session.adapterWeights().values.reduce(0) { $0 + $1.values.count }
                emit(.init(message: "\(category.name) \(artifact.map): \(trainableParameters.formatted()) trainable LoRA parameters"), run: run)
                var finalLoss = 0.0
                for epoch in 0..<options.epochs {
                    var order = Array(members.indices)
                    if options.shuffleEachEpoch { order.shuffle(using: &orderGenerator) }
                    let batches = stride(from: 0, to: order.count, by: options.batchSize).map { Array(order[$0..<min(order.count, $0 + options.batchSize)]) }
                    for start in stride(from: 0, to: batches.count, by: options.gradientAccumulationSteps) {
                        let window = batches[start..<min(batches.count, start + options.gradientAccumulationSteps)]
                        let windowSamples = window.reduce(0) { $0 + $1.count }
                        var loss = 0.0
                        for batch in window {
                            try cancellation.check()
                            var input: [Float] = [], target: [Float] = []
                            for index in batch {
                                let pair = members[index]
                                try verify([pair.sample.input, pair.target], fingerprints: fingerprints)
                                let transform = try sampler.transform(sampleID: pair.sample.id, epoch: epoch,
                                    width: pair.width, height: pair.height, size: options.imageSize,
                                    flip: options.flipCrops, rotate: options.rotateCrops)
                                let loaded = try NativeImages.pair(inputURL: pair.sample.input, targetURL: pair.target, transform: transform,
                                    mapKind: artifact.map, normalConvention: pair.sample.normalConvention,
                                    exaggerate: artifact.map == "depth" && options.exaggerateFeatures, cancellation: cancellation)
                                input.append(contentsOf: loaded.input.values); target.append(contentsOf: loaded.target.values)
                            }
                            let weight = Double(batch.count) / Double(windowSamples)
                            loss += try session.train(input: input, target: target, batchSize: batch.count,
                                height: options.imageSize, width: options.imageSize, weight: Float(weight)) * weight
                        }
                        try cancellation.check()
                        let rate = Self.learningRate(step: session.updateCount, total: groupTotal, stepsPerEpoch: stepsPerEpoch, options: options)
                        _ = try session.applyGradients(learningRate: rate)
                        guard loss.isFinite else { throw WorkshopError("LoRA training produced a non-finite loss; no adapter was published.") }
                        finalLoss = loss; updates += 1
                        let checkpointSampler = sampler, checkpointOrder = orderGenerator.state
                        saveStoppedUpdate = {
                            try checkpoint(session: session, run: run, category: category, map: artifact.map, epoch: epoch + 1,
                                baseHash: baseHash, loss: loss, sampler: checkpointSampler, orderState: checkpointOrder,
                                studio: studio, stopped: true)
                        }
                        emit(.init(message: "\(category.name) · \(artifact.map) · epoch \(epoch + 1) of \(options.epochs) · update \(session.updateCount)",
                            progress: Double(updates) / Double(totalUpdates), epoch: epoch + 1, loss: loss,
                            update: updates, totalUpdates: totalUpdates, learningRate: rate), run: run)
                    }
                    if (epoch + 1) % options.checkpointEveryEpochs == 0 || epoch + 1 == options.epochs {
                        try checkpoint(session: session, run: run, category: category, map: artifact.map, epoch: epoch + 1,
                            baseHash: baseHash, loss: finalLoss, sampler: sampler, orderState: orderGenerator.state, studio: studio)
                        emit(.init(message: "Saved \(category.name) \(artifact.map) LoRA checkpoint for epoch \(epoch + 1)", epoch: epoch + 1), run: run)
                    }
                }
                try cancellation.check()
                guard try session.weights() == frozen, try fileSHA(baseWeightsURL, cancellation: cancellation) == baseHash else {
                    throw WorkshopError("Frozen base weights changed; adapter export was stopped.")
                }
                let folder = package.appendingPathComponent("models/" + Self.folderID(category.id) + "/" + artifact.map, isDirectory: true)
                try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
                try NativeWeights.write(session.adapterWeights(), to: folder.appendingPathComponent("adapter.safetensors"),
                    metadata: ["schema": "texture-loom.lora-weights.v1", "rank": String(options.rank), "alpha": String(options.alpha)])
                emit(.init(message: "Exporting \(category.name) \(artifact.map) to native Core ML"), run: run)
                let compiled = try NativeCoreML.export(weights: session.fusedWeights(), architecture: architecture,
                    imageSize: options.imageSize, to: folder.appendingPathComponent("adapter.mlmodel"))
                defer { try? FileManager.default.removeItem(at: compiled) }
                try cancellation.check()
                let difference = try compatibilityProof(session: session, compiled: compiled, map: artifact.map)
                let relative = "models/" + Self.folderID(category.id) + "/" + artifact.map
                adapters[category.id, default: []].append(["map": artifact.map, "format": "coreml", "path": relative + "/adapter.mlmodel",
                    "weightsPath": relative + "/adapter.safetensors", "baseSHA256": baseHash])
                try Self.json(["schema": "loom-workshop-lora-v1", "backend": "native-mpsgraph", "base_id": studio.modelID,
                    "base_version": studio.version, "category_id": category.id, "category_version": category.version,
                    "map": artifact.map, "base_sha256": baseHash, "rank": options.rank, "alpha": options.alpha,
                    "targets": frozen.keys.filter { $0.hasSuffix(".weight") }.map { String($0.dropLast(7)) }.sorted(),
                    "architecture_config": config, "epochs": options.epochs, "updates": session.updateCount, "final_loss": finalLoss,
                    "base_weights_frozen": true, "coreml_maximum_absolute_difference": difference,
                    "training_options": try Self.dictionary(options), "base_parameters": baseParameters,
                    "trainable_parameters": trainableParameters,
                    "quality_note": "Adapter completion does not measure prediction quality on unseen materials."],
                    to: folder.appendingPathComponent("adapter.json"))
                emit(.init(message: "\(category.name) \(artifact.map) LoRA exported and checked"), run: run)
                saveStoppedUpdate = nil
            }
            try cancellation.check()
            try verify(Array(paths), fingerprints: fingerprints, checkBytes: true)
            _ = try readStudio(application, cancellation: cancellation)
            guard try Data(contentsOf: studio.packageURL.appendingPathComponent("manifest.json")) == originalBaseManifest else {
                throw WorkshopError("Texture Studio's built-in package changed during training. Reload the application before training.")
            }
            var files: [[String: String]] = []
            let entries = FileManager.default.enumerator(at: package, includingPropertiesForKeys: [.isRegularFileKey])!
            for case let file as URL in entries where try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true {
                try cancellation.check()
                files.append(["path": String(file.path.dropFirst(package.path.count + 1)), "sha256": try fileSHA(file, cancellation: cancellation)])
            }
            let assembled: [String: Any] = ["schema": "texture-loom.adapter.v1", "id": UUID().uuidString.lowercased(), "name": options.name,
                "version": "1.0.0", "baseModel": ["id": studio.modelID, "version": studio.version],
                "categories": manifest.categories.filter { adapters[$0.id] != nil }.map { ["id": $0.id, "version": $0.version, "adapters": adapters[$0.id]!] as [String: Any] },
                "files": files.sorted { $0["path"]! < $1["path"]! }, "createdAt": ISO8601DateFormatter().string(from: Date())]
            try cancellation.check()
            try Self.json(assembled, to: package.appendingPathComponent("manifest.json"))
            try cancellation.check()
            try Self.json(["status": "completed", "backend": "native-mpsgraph", "updates": updates,
                "base_weights_frozen": true, "originals_preserved": true], to: run.appendingPathComponent("result.json"))
            try cancellation.check()
            return package
        } catch {
            if error is CancellationError { try? saveStoppedUpdate?() }
            try? FileManager.default.removeItem(at: run.appendingPathComponent("adapter/manifest.json"))
            try? Self.json(["status": error is CancellationError ? "cancelled" : "failed", "backend": "native-mpsgraph",
                "updates": updates, "message": error.localizedDescription], to: run.appendingPathComponent("result.json"))
            throw error
        }
    }

    private func classify(_ samples: [WorkshopSample], studio: StudioBase, manifest: BaseManifest) throws -> [String: [NativeWorkshopPair]] {
        var classifier: MLModel?
        var groups: [String: [NativeWorkshopPair]] = [:]
        let targets = Set(samples.flatMap { $0.targets.values })
        let targetMembership = try Dictionary(uniqueKeysWithValues: targets.map { ($0, try membershipSignature($0)) })
        let inputMembership = try Dictionary(uniqueKeysWithValues: samples.map { ($0.input, try membershipSignature($0.input)) })
        let descriptor = manifest.classifier
        for (index, sample) in samples.enumerated() {
            try cancellation.check()
            let before = try membershipSignature(sample.input)
            guard before == inputMembership[sample.input] else {
                throw WorkshopError("An image's dataset or pair metadata changed while images were being classified. Refresh the source folder.")
            }
            let selection = try activeImageSelection(sample.input, studio: studio)
            guard selection.id == sample.id, selection.pairID == sample.pairID, selection.role == "diffuse", !selection.ignored,
                  sample.categoryID == nil || sample.categoryID == selection.categoryID else {
                throw WorkshopError("An image's dataset, pair, role or ignore metadata changed. Refresh the source folder.")
            }
            let originalIdentity = try identity(sample.input)
            var categoryID = selection.categoryID
            if categoryID == nil {
                if classifier == nil {
                    let source = try modelPath(descriptor.path, in: studio.packageURL)
                    let compiled = source.pathExtension == "mlmodelc" ? source : try MLModel.compileModel(at: source)
                    let config = MLModelConfiguration(); config.computeUnits = .cpuOnly
                    classifier = try MLModel(contentsOf: compiled, configuration: config)
                }
                let model = classifier!
                let inputName = descriptor.inputName ?? "rgb", outputName = descriptor.outputName ?? "dataset"
                guard let feature = model.modelDescription.inputDescriptionsByName[inputName] else {
                    throw WorkshopError("The installed classifier is missing its declared input.")
                }
                let value: MLFeatureValue
                if let constraint = feature.imageConstraint {
                    guard let imageSource = CGImageSourceCreateWithURL(sample.input as CFURL, nil),
                          let image = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) else {
                        throw WorkshopError("The classifier could not read \(sample.input.lastPathComponent).")
                    }
                    value = try MLFeatureValue(cgImage: image, pixelsWide: constraint.pixelsWide,
                        pixelsHigh: constraint.pixelsHigh, pixelFormatType: constraint.pixelFormatType, options: nil)
                } else if let constraint = feature.multiArrayConstraint {
                    let shape = constraint.shape.map(\.intValue)
                    guard shape.count == 4, shape[0] == 1, shape[1] == 3 else {
                        throw WorkshopError("The installed classifier requires an unsupported tensor input.")
                    }
                    value = MLFeatureValue(multiArray: try Self.classifierArray(try NativeImages.readRGB(from: sample.input, cancellation: cancellation),
                        width: shape[3], height: shape[2], center: descriptor.architecture != "create-ml-image-classifier-v1"))
                } else { throw WorkshopError("The installed classifier requires an image or RGB tensor input.") }
                let result = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: [inputName: value]))
                let label = result.featureValue(for: outputName)?.stringValue
                categoryID = descriptor.labels.first { $0.classifierValue == label }?.id
            }
            guard try membershipSignature(sample.input) == before else {
                throw WorkshopError("An image's metadata changed while it was being classified. Refresh the source folder.")
            }
            let currentIdentity = try identity(sample.input)
            guard originalIdentity.0 == currentIdentity.0, originalIdentity.1 == currentIdentity.1 else {
                throw WorkshopError("An image changed while it was being classified. Refresh the source folder.")
            }
            guard let categoryID, let category = manifest.categories.first(where: { $0.id == categoryID }) else {
                throw WorkshopError("The installed classifier returned a dataset missing from its base model.")
            }
            guard var tag = imageTag(sample.input), tag.id == sample.id else {
                throw WorkshopError("The source image lost its identity metadata during classification.")
            }
            let state = try identity(sample.input)
            tag.categoryID = categoryID; tag.baseID = studio.modelID; tag.baseVersion = studio.version
            tag.classifiedBytes = state.0; tag.classifiedModified = state.1
            try writeTag(tag, to: sample.input)
            if options.datasetIDs.isEmpty || options.datasetIDs.contains(categoryID) {
                for artifact in category.models {
                    if let target = sample.targets[artifact.map] {
                        let teaching = try activeImageSelection(target, studio: studio)
                        guard teaching.pairID == sample.pairID, teaching.role == artifact.map, !teaching.ignored,
                              artifact.map != "normal" || teaching.normalConvention.lowercased() == sample.normalConvention.lowercased() else {
                            throw WorkshopError("A target's pair, role, convention or ignore metadata changed. Refresh the source folder.")
                        }
                        if let targetCategory = teaching.categoryID, targetCategory != categoryID { continue }
                        guard let dimensions = imageDimensions(sample.input) else { throw WorkshopError("The source image dimensions could not be read.") }
                        let refreshed = WorkshopSample(id: sample.id, pairID: sample.pairID, input: sample.input, targets: sample.targets,
                            categoryID: categoryID, normalConvention: sample.normalConvention)
                        groups[categoryID + "|" + artifact.map, default: []].append(.init(sample: refreshed,
                            category: category, artifact: artifact, target: target, width: dimensions.0, height: dimensions.1))
                    }
                }
            }
            progress(.init(message: "Sorted image \(index + 1) of \(samples.count)", progress: Double(index + 1) / Double(samples.count)))
        }
        for target in targets {
            try cancellation.check()
            guard try membershipSignature(target) == targetMembership[target] else {
                throw WorkshopError("A target's dataset or pair metadata changed while images were being classified. Refresh the source folder.")
            }
        }
        for pair in groups.values.flatMap({ $0 }) {
            let current = try activeImageSelection(pair.sample.input, studio: studio)
            guard current.id == pair.sample.id, current.pairID == pair.sample.pairID,
                  current.role == "diffuse", !current.ignored, current.categoryID == pair.category.id else {
                throw WorkshopError("An image's selection changed while images were being classified. Refresh the source folder.")
            }
        }
        return groups
    }

    private func verify(_ paths: [URL], fingerprints: [URL: ImageFingerprint], checkBytes: Bool = false) throws {
        for path in paths {
            try cancellation.check()
            guard let expected = fingerprints[path], FileManager.default.fileExists(atPath: path.path) else {
                throw WorkshopError("A training image was moved or deleted: \(path.lastPathComponent).")
            }
            let current = try identity(path)
            guard current.0 == expected.bytes, current.1 == expected.modified,
                  try membershipSignature(path) == expected.membership,
                  try (!checkBytes || fileSHA(path, cancellation: cancellation) == expected.sha256) else {
                throw WorkshopError("A training image or its dataset, pair, role or ignore metadata changed. Refresh the folder and start another run.")
            }
        }
    }

    private func checkpoint(session: NativeTrainingSession, run: URL, category: BaseManifest.Category, map: String,
                            epoch: Int, baseHash: String, loss: Double, sampler: NativeAugmentationPlan,
                            orderState: UInt64, studio: StudioBase, stopped: Bool = false) throws {
        if !stopped { try cancellation.check() }
        let directory = run.appendingPathComponent("checkpoints/" + Self.folderID(category.id) + "/" + map, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let prefix = stopped ? String(format: "stopped-update-%09d", session.updateCount) : String(format: "epoch-%06d", epoch)
        try NativeWeights.write(session.adapterWeights(), to: directory.appendingPathComponent(prefix + ".safetensors"),
            metadata: ["rank": String(options.rank), "alpha": String(options.alpha)])
        try NativeWeights.write(session.optimizerState(), to: directory.appendingPathComponent(prefix + ".optimizer.safetensors"))
        try Self.json(["schema": "loom-workshop-checkpoint-v2", "backend": "native-mpsgraph", "epoch": epoch,
            "updates": session.updateCount, "epoch_completed": !stopped, "base_id": studio.modelID, "base_version": studio.version,
            "category_id": category.id, "category_version": category.version, "map": map, "base_sha256": baseHash,
            "final_loss": loss, "rank": options.rank, "alpha": options.alpha, "options": try Self.dictionary(options),
            "weights_path": prefix + ".safetensors", "optimizer_path": prefix + ".optimizer.safetensors",
            "sampling_state": try Self.dictionary(sampler), "order_state": String(orderState)],
            to: directory.appendingPathComponent(prefix + ".json"))
    }

    private func compatibilityProof(session: NativeTrainingSession, compiled: URL, map: String) throws -> Double {
        let configuration = MLModelConfiguration(); configuration.computeUnits = .cpuOnly
        let model = try MLModel(contentsOf: compiled, configuration: configuration)
        var generator = WorkshopOrderGenerator(seed: UInt64(options.seed))
        let input = (0..<(3 * 32 * 32)).map { _ in Float(generator.next() >> 40) / Float(1 << 24) }
        let expected = try session.predict(input: input, batchSize: 1, height: 32, width: 32)
        let array = try MLMultiArray(shape: [1, 3, 32, 32], dataType: .float32)
        let buffer = array.dataPointer.assumingMemoryBound(to: Float.self)
        for index in input.indices { buffer[index] = input[index] }
        let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["rgb": MLFeatureValue(multiArray: array)]))
        guard let actual = output.featureValue(for: "material_map")?.multiArrayValue, actual.count == expected.count else {
            throw WorkshopError("The exported Core ML LoRA has an incompatible output.")
        }
        var difference = 0.0
        for index in expected.indices {
            let value = actual[index].doubleValue, reference = Double(expected[index])
            guard value.isFinite, abs(value - reference) <= 0.0003 + 0.0003 * abs(reference) else {
                throw WorkshopError("The native Core ML export differs from the learned LoRA. No adapter was published.")
            }
            difference = max(difference, abs(value - reference))
        }
        return difference
    }

    private func emit(_ event: WorkshopEvent, run: URL) {
        progress(event)
        var value: [String: Any] = ["message": event.message, "backend": "native-mpsgraph"]
        if let progress = event.progress { value["progress"] = progress }
        if let epoch = event.epoch { value["epoch"] = epoch }
        if let loss = event.loss { value["loss"] = loss }
        if let update = event.update { value["update"] = update }
        if let total = event.totalUpdates { value["total_updates"] = total }
        if let rate = event.learningRate { value["learning_rate"] = rate }
        if var line = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]) {
            line.append(10)
            let log = run.appendingPathComponent("train.log")
            if !FileManager.default.fileExists(atPath: log.path) { _ = FileManager.default.createFile(atPath: log.path, contents: nil) }
            if let handle = try? FileHandle(forWritingTo: log) { _ = try? handle.seekToEnd(); try? handle.write(contentsOf: line); try? handle.close() }
        }
    }
    static func learningRate(step: Int, total: Int, stepsPerEpoch: Int, options: WorkshopOptions) -> Double {
        let warmup = min(options.warmupEpochs * stepsPerEpoch, total)
        if warmup > 0, step < warmup { return options.learningRate * Double(step + 1) / Double(warmup) }
        guard options.learningRateSchedule == .cosine else { return options.learningRate }
        let span = total - warmup
        let fraction = span <= 1 ? 0 : Double(step - warmup) / Double(span - 1)
        let factor = options.minimumLearningRateRatio + (1 - options.minimumLearningRateRatio) * (1 + cos(.pi * fraction)) / 2
        return options.learningRate * factor
    }
    static func stepsPerEpoch(samples: Int, options: WorkshopOptions) -> Int {
        let batches = (samples + options.batchSize - 1) / options.batchSize
        return (batches + options.gradientAccumulationSteps - 1) / options.gradientAccumulationSteps
    }
    private static func folderID(_ category: String) -> String {
        "dataset-" + SHA256.hash(data: Data(category.utf8)).prefix(8).map { String(format: "%02x", $0) }.joined()
    }
    private static func dictionary<T: Encodable>(_ value: T) throws -> [String: Any] {
        try JSONSerialization.jsonObject(with: JSONEncoder().encode(value)) as! [String: Any]
    }
    static func json(_ object: [String: Any], to path: URL) throws {
        try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys]).write(to: path, options: .atomic)
    }
    private static func classifierArray(_ image: NativeImage, width: Int, height: Int, center: Bool) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)], dataType: .float32)
        let destination = array.dataPointer.assumingMemoryBound(to: Float.self)
        let sourceWidth = center ? min(image.width, image.height) : image.width
        let sourceHeight = center ? min(image.width, image.height) : image.height
        let left = center ? (image.width - sourceWidth) / 2 : 0, top = center ? (image.height - sourceHeight) / 2 : 0
        for y in 0..<height { for x in 0..<width {
            let px = max(0, min(Double(sourceWidth - 1), (Double(x) + 0.5) * Double(sourceWidth) / Double(width) - 0.5))
            let py = max(0, min(Double(sourceHeight - 1), (Double(y) + 0.5) * Double(sourceHeight) / Double(height) - 0.5))
            let x0 = Int(px), y0 = Int(py), x1 = min(sourceWidth - 1, x0 + 1), y1 = min(sourceHeight - 1, y0 + 1)
            let fx = Float(px - Double(x0)), fy = Float(py - Double(y0))
            for channel in 0..<3 {
                func pixel(_ sx: Int, _ sy: Int) -> Float { image.values[((sy + top) * image.width + sx + left) * 3 + channel] }
                let a = pixel(x0, y0) * (1 - fx) + pixel(x1, y0) * fx
                let b = pixel(x0, y1) * (1 - fx) + pixel(x1, y1) * fx
                destination[(channel * height + y) * width + x] = a * (1 - fy) + b * fy
            }
        } }
        return array
    }
}

struct WorkshopOrderGenerator: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var result = state
        result = (result ^ (result >> 30)) &* 0xBF58476D1CE4E5B9
        result = (result ^ (result >> 27)) &* 0x94D049BB133111EB
        return result ^ (result >> 31)
    }
}
