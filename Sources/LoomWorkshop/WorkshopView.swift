import SwiftUI
import WorkshopCore

struct WorkshopView: View {
    @Bindable var model: WorkshopModel
    var startsAutomatically = true
    var showsAdvancedOptionsInitially = false
    @State private var advancedOptions = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                HStack(alignment: .top, spacing: 22) {
                    VStack(spacing: 14) {
                        studioCard
                        imagesCard
                        trainingCard
                    }
                    .frame(maxWidth: .infinity)
                    VStack(spacing: 18) {
                        summaryCard
                        outputCard
                    }
                    .frame(width: 290)
                }
                .padding(22)
            }
            .background(Color(nsColor: .windowBackgroundColor))
            footer
        }
        .task {
            guard startsAutomatically else { return }
            model.start()
            while !Task.isCancelled {
                do { try await Task.sleep(for: .seconds(3)) }
                catch { return }
                await model.pollFolder()
            }
        }
        .onAppear { if showsAdvancedOptionsInitially { advancedOptions = true } }
        .onChange(of: model.options) { _, _ in model.scheduleOptionsSave() }
        .onChange(of: model.selectedDatasetID) { _, _ in model.scheduleOptionsSave() }
        .alert("Action could not finish", isPresented: Binding(
            get: { model.errorMessage != nil },
            set: { if !$0 { model.errorMessage = nil } }
        )) {
            Button("OK") { model.errorMessage = nil }
        } message: {
            Text(model.errorMessage ?? "")
        }
    }

    private var header: some View {
        HStack(spacing: 15) {
            WorkshopMark().frame(width: 52, height: 52)
            VStack(alignment: .leading, spacing: 4) {
                Text("Loom Workshop").font(.system(size: 25, weight: .semibold))
                Text("Train your materials into a LoRA for Texture Studio.")
                    .font(.callout).foregroundStyle(.secondary)
            }
            Spacer()
            Label("Built for your installed model", systemImage: "link")
                .font(.callout).foregroundStyle(.secondary)
        }
        .padding(.horizontal, 28).padding(.vertical, 22)
        .background(.background)
    }

    private var studioCard: some View {
        WorkshopCard(1, "Texture Studio", subtitle: "Your LoRA uses the Texture Loom model inside this app.") {
            HStack(alignment: .top, spacing: 14) {
                Image(systemName: model.studio == nil ? "shippingbox" : "checkmark.seal.fill")
                    .font(.title2)
                    .foregroundStyle(model.studio == nil ? Color.secondary : WorkshopStyle.accent)
                    .frame(width: 30)
                VStack(alignment: .leading, spacing: 5) {
                    if let studio = model.studio {
                        Text(studio.modelName).font(.body.weight(.semibold))
                        Text("Version \(studio.version) · \(studio.categories.count) \(studio.categories.count == 1 ? "dataset" : "datasets")")
                            .font(.callout).foregroundStyle(.secondary)
                    } else {
                        Text(model.busy && model.sourceFolder == nil ? "Checking installed app…" : "Choose Texture Studio")
                            .font(.body.weight(.semibold))
                    }
                    WorkshopPath(path: model.studioURL, symbol: "app")
                }
                Spacer(minLength: 8)
                Button(model.studio == nil ? "Choose App…" : "Change App…") { model.chooseStudio() }
                    .disabled(model.busy)
            }
            if let issue = model.studioIssue {
                Label {
                    Text(issue).font(.callout).fixedSize(horizontal: false, vertical: true)
                } icon: { Image(systemName: "info.circle") }
                .foregroundStyle(.secondary)
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 9))
            }
        }
    }

    private var imagesCard: some View {
        WorkshopCard(2, "Your material images", subtitle: "Diffuse images paired with depth, roughness, or normal maps.") {
            if let folder = model.sourceFolder {
                HStack(alignment: .top, spacing: 14) {
                    Image(systemName: "folder.fill").font(.title2).foregroundStyle(WorkshopStyle.accent)
                        .frame(width: 30)
                    VStack(alignment: .leading, spacing: 5) {
                        Text(folder.lastPathComponent).font(.body.weight(.semibold))
                        WorkshopPath(path: folder, symbol: "folder")
                    }
                    Spacer(minLength: 8)
                    Button("Change Folder…") { model.chooseSourceFolder() }.disabled(model.busy)
                }
                HStack {
                    if let plan = model.plan {
                        Text("\(plan.imageCount) diffuse \(plan.imageCount == 1 ? "image" : "images") · \(model.availableDatasets.count) \(model.availableDatasets.count == 1 ? "dataset" : "datasets") with matching maps")
                            .font(.callout).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("Open Folder") { model.revealSource() }
                    if model.plan?.requiresClassification == true {
                        Button("Update Datasets") { model.analyze() }.disabled(model.busy)
                    }
                }
                if let issue = model.folderIssue {
                    Label(issue, systemImage: "info.circle")
                        .font(.callout).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            } else {
                HStack(spacing: 14) {
                    Image(systemName: "photo.stack").font(.title2).foregroundStyle(.secondary).frame(width: 30)
                    Text("The classifier sorts your images into datasets automatically.")
                        .font(.callout).foregroundStyle(.secondary)
                    Spacer(minLength: 8)
                    Button("Choose Image Folder…") { model.chooseSourceFolder() }
                        .buttonStyle(.borderedProminent).disabled(model.studio == nil || model.busy)
                }
            }
        }
    }

    private var trainingCard: some View {
        WorkshopCard(3, "Train your LoRA", subtitle: "Train one dataset or every dataset found in your folder.") {
            Grid(alignment: .leading, horizontalSpacing: 20, verticalSpacing: 16) {
                GridRow {
                    Text("LoRA name").foregroundStyle(.secondary)
                    TextField("My Texture LoRA", text: $model.options.name).textFieldStyle(.roundedBorder)
                        .accessibilityIdentifier("adapter-name")
                }
                GridRow {
                    Text("Dataset").foregroundStyle(.secondary)
                    Picker("Dataset", selection: $model.selectedDatasetID) {
                        Text("All available datasets").tag("")
                        ForEach(model.availableDatasets) { dataset in Text(dataset.name).tag(dataset.id) }
                    }.labelsHidden().accessibilityIdentifier("dataset-picker")
                }
                GridRow {
                    Text("Image size").foregroundStyle(.secondary)
                    Picker("Image size", selection: $model.options.imageSize) {
                        ForEach([256, 512, 1024, 2048, 4096], id: \.self) { size in
                            Text("\(size.formatted()) × \(size.formatted())").tag(size)
                        }
                    }.labelsHidden()
                }
                GridRow {
                    Text("Epochs").foregroundStyle(.secondary)
                    HStack {
                        Stepper(value: $model.options.epochs, in: 1...10_000) {
                            Text(model.options.epochs, format: .number).monospacedDigit().frame(width: 44, alignment: .leading)
                        }.fixedSize()
                        Text("Complete passes through your images").font(.callout).foregroundStyle(.secondary)
                    }
                }
            }
            Divider()
            VStack(alignment: .leading, spacing: 12) {
                Toggle("Randomize image order each epoch", isOn: $model.options.shuffleEachEpoch)
                Toggle("Randomly flip crops", isOn: $model.options.flipCrops)
                Toggle("Randomly rotate crops by quarter turns", isOn: $model.options.rotateCrops)
                HStack {
                    Text("Choose crop positions").foregroundStyle(.secondary)
                    Spacer()
                    Picker("Choose crop positions", selection: $model.options.cropRefresh) {
                        ForEach(WorkshopRefresh.allCases) { Text($0.title).tag($0) }
                    }.labelsHidden().frame(width: 200)
                }
                HStack {
                    Text("Choose flips and rotations").foregroundStyle(.secondary)
                    Spacer()
                    Picker("Choose flips and rotations", selection: $model.options.augmentationRefresh) {
                        ForEach(WorkshopRefresh.allCases) { Text($0.title).tag($0) }
                    }.labelsHidden().frame(width: 200)
                }.disabled(!model.options.flipCrops && !model.options.rotateCrops)
                Toggle("Exaggerate features", isOn: $model.options.exaggerateFeatures)
                    .disabled(!hasDepthPairs)
                Text("Flips use independent 50% chances horizontally and vertically. Crop positions vary wherever the image has room. Exaggerate features strengthens depth relief.")
                    .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }.toggleStyle(.checkbox)
            DisclosureGroup("Advanced options", isExpanded: $advancedOptions) {
                Grid(alignment: .leading, horizontalSpacing: 20, verticalSpacing: 12) {
                    GridRow {
                        Text("Optimizer").foregroundStyle(.secondary)
                        Picker("Optimizer", selection: $model.options.optimizer) {
                            ForEach(WorkshopOptimizer.allCases) { Text($0.title).tag($0) }
                        }.labelsHidden().frame(maxWidth: 180).accessibilityIdentifier("optimizer-picker")
                    }
                    GridRow {
                        Text("Learning rate").foregroundStyle(.secondary)
                        TextField("Learning rate", value: $model.options.learningRate,
                                  format: .number.precision(.fractionLength(1...7)))
                            .textFieldStyle(.roundedBorder).frame(maxWidth: 180)
                    }
                    GridRow {
                        Text("LoRA rank").foregroundStyle(.secondary)
                        Picker("LoRA rank", selection: $model.options.rank) {
                            ForEach([1, 2, 4, 8, 16, 32, 64, 128], id: \.self) { rank in Text(rank, format: .number).tag(rank) }
                        }.labelsHidden().frame(maxWidth: 180)
                    }
                    GridRow {
                        Text("LoRA alpha").foregroundStyle(.secondary)
                        HStack { decimalField($model.options.alpha); Text("Scale = alpha ÷ rank").font(.caption).foregroundStyle(.secondary) }
                    }
                    GridRow {
                        Text("Batch size").foregroundStyle(.secondary)
                        Stepper(value: $model.options.batchSize, in: 1...256) {
                            Text("\(model.options.batchSize) \(model.options.batchSize == 1 ? "pair" : "pairs") at once")
                        }.accessibilityIdentifier("batch-size")
                    }
                    GridRow {
                        Text("Gradient accumulation").foregroundStyle(.secondary)
                        Stepper(value: $model.options.gradientAccumulationSteps, in: 1...4096) {
                            Text("\(model.options.gradientAccumulationSteps) batches per update")
                        }.accessibilityIdentifier("gradient-accumulation")
                    }
                    GridRow {
                        Text("Effective batch").foregroundStyle(.secondary)
                        Text("Up to \(model.options.effectiveBatchSize) pairs per update").font(.callout)
                    }
                    GridRow {
                        Text("Weight decay").foregroundStyle(.secondary)
                        decimalField($model.options.weightDecay)
                    }
                    if model.options.optimizer == .sgd {
                        GridRow {
                            Text("Momentum").foregroundStyle(.secondary)
                            decimalField($model.options.momentum)
                        }
                    } else {
                        GridRow {
                            Text("Beta 1 / Beta 2").foregroundStyle(.secondary)
                            HStack { decimalField($model.options.beta1); decimalField($model.options.beta2) }
                        }
                    }
                    GridRow {
                        Text("Gradient clip norm").foregroundStyle(.secondary)
                        HStack { decimalField($model.options.maxGradientNorm); Text("0 disables clipping").font(.caption).foregroundStyle(.secondary) }
                    }
                    GridRow {
                        Text("Learning rate schedule").foregroundStyle(.secondary)
                        Picker("Learning rate schedule", selection: $model.options.learningRateSchedule) {
                            ForEach(WorkshopSchedule.allCases) { Text($0.title).tag($0) }
                        }.labelsHidden().frame(maxWidth: 180)
                    }
                    GridRow {
                        Text("Warmup epochs").foregroundStyle(.secondary)
                        Stepper(value: $model.options.warmupEpochs, in: 0...(max(1, model.options.epochs) - 1)) {
                            Text(model.options.warmupEpochs, format: .number)
                        }
                    }
                    if model.options.learningRateSchedule == .cosine {
                        GridRow {
                            Text("Final learning rate ratio").foregroundStyle(.secondary)
                            decimalField($model.options.minimumLearningRateRatio)
                        }
                    }
                    GridRow {
                        Text("Random seed").foregroundStyle(.secondary)
                        TextField("Random seed", value: $model.options.seed, format: .number.grouping(.never))
                            .textFieldStyle(.roundedBorder).frame(maxWidth: 180)
                    }
                    GridRow {
                        Text("Training device").foregroundStyle(.secondary)
                        Picker("Training device", selection: $model.options.device) {
                            ForEach(WorkshopDevice.allCases) { Text($0.title).tag($0).disabled($0 == .cpu) }
                        }.labelsHidden().frame(maxWidth: 180)
                    }
                    GridRow {
                        Text("Save checkpoint every").foregroundStyle(.secondary)
                        Stepper(value: $model.options.checkpointEveryEpochs, in: 1...100_000) {
                            Text("\(model.options.checkpointEveryEpochs) epochs")
                        }
                    }
                }.padding(.top, 12)
                Text(model.options.optimizer == .lion ? "Lion usually needs a lower learning rate than AdamW. Start with 0.00003 and betas 0.9 / 0.99, then adjust from your results." : "Accumulation increases the effective batch while keeping only one batch in memory. The last update of each epoch uses the remaining pairs.")
                    .font(.callout).foregroundStyle(.secondary).padding(.top, 10)
            }
            if let plan = model.plan, !plan.requiresClassification, model.availableDatasets.isEmpty {
                Label("No matching map pairs at this image size. Check the matching files or choose a smaller size.", systemImage: "info.circle")
                    .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            if let issue = model.trainingOptionsIssue {
                Label(issue, systemImage: "info.circle").font(.callout).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Text("Training files and results stay in this image folder.")
                .font(.callout).foregroundStyle(.secondary)
        }.disabled(model.studio == nil || model.plan == nil || model.busy)
    }

    private var summaryCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            Label("Training summary", systemImage: "square.stack.3d.up")
                .font(.headline)
            if let plan = model.plan {
                if plan.requiresClassification {
                    Text("Your classifier will identify the datasets before training.")
                        .font(.callout).foregroundStyle(.secondary)
                } else if selectedDatasets.isEmpty {
                    Text("Matching diffuse and target maps appear here after your folder is read.")
                        .font(.callout).foregroundStyle(.secondary)
                } else {
                    Text("\(model.options.optimizer.title) · up to \(model.options.effectiveBatchSize) pairs per update")
                        .font(.callout).foregroundStyle(.secondary)
                    if model.trainingOptionsIssue == nil {
                        Text("\(plannedUpdates.formatted()) optimizer updates over \(model.options.epochs) epochs")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    ForEach(selectedDatasets) { dataset in
                        VStack(alignment: .leading, spacing: 8) {
                            Text(dataset.name).font(.body.weight(.semibold))
                            ForEach(["depth", "roughness", "normal"], id: \.self) { kind in
                                if let count = dataset.pairedCounts[kind], count > 0 {
                                    HStack {
                                        Text(mapTitle(kind)).foregroundStyle(.secondary)
                                        Spacer()
                                        Text("\(count) \(count == 1 ? "pair" : "pairs")").monospacedDigit()
                                    }.font(.callout)
                                    if (1...128).contains(model.options.rank), let profile = model.studio?.categories.first(where: { $0.id == dataset.id })?.modelProfiles?.first(where: { $0.map == kind }) {
                                        Text("\(profile.baseParameters.formatted()) frozen base · \(profile.trainableParameters(rank: model.options.rank).formatted()) trainable LoRA parameters")
                                            .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                                    }
                                }
                            }
                        }
                        if dataset.id != selectedDatasets.last?.id { Divider() }
                    }
                }
            } else {
                Text("Choose Texture Studio and an image folder to see the datasets and maps that will train.")
                    .font(.callout).foregroundStyle(.secondary)
            }
        }.sidebarCard()
    }

    private var outputCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            Label("Your LoRA", systemImage: "shippingbox")
                .font(.headline)
            if model.trainedAdapter != nil {
                Label(model.installed ? "Installed in Texture Studio" : "Ready to export or install", systemImage: "checkmark.circle.fill")
                    .font(.callout).foregroundStyle(WorkshopStyle.accent)
                Text(model.trainedName).font(.body.weight(.medium))
            } else {
                Text("Your finished LoRA can be installed in Texture Studio or exported as a single file.")
                    .font(.callout).foregroundStyle(.secondary)
            }
            Button { model.install() } label: {
                Label("Install in Texture Studio", systemImage: "arrow.down.app")
                    .frame(maxWidth: .infinity)
            }.buttonStyle(.borderedProminent)
                .disabled(model.trainedAdapter == nil || model.busy)
            Button { model.export() } label: {
                Label("Export LoRA…", systemImage: "square.and.arrow.up")
                    .frame(maxWidth: .infinity)
            }.disabled(model.trainedAdapter == nil || model.busy)
            if model.installed {
                Button("Open Texture Studio") { model.openStudio() }
            }
            if !model.savedRuns.isEmpty {
                Divider()
                Text("Saved training runs").font(.headline)
                Picker("Saved run", selection: $model.selectedSavedRunID) {
                    ForEach(model.savedRuns) { run in
                        Text("\(run.name) · \(run.updates.formatted()) updates").tag(run.id)
                    }
                }.labelsHidden().disabled(model.busy)
                Button("Resume Saved Run") { model.resumeSavedRun() }
                    .disabled(model.busy).accessibilityIdentifier("resume-saved-run")
                Text("Continues the saved optimizer, crop order and exact update using that run's settings.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if model.latestCheckpoint != nil || !model.savedRuns.isEmpty {
                Button("Show Saved Checkpoint") { model.revealLatestCheckpoint() }
            }
            Button("Open Runs Folder") { model.revealRuns() }.disabled(model.sourceFolder == nil)
        }.sidebarCard()
    }

    private var footer: some View {
        VStack(spacing: 0) {
            Divider()
            HStack(spacing: 14) {
                if model.busy {
                    if let value = model.progress {
                        ProgressView(value: value).frame(width: 150)
                    } else { ProgressView().controlSize(.small).frame(width: 18) }
                } else {
                    Image(systemName: model.trainedAdapter == nil ? "circle" : "checkmark.circle.fill")
                        .foregroundStyle(model.trainedAdapter == nil ? Color.secondary : WorkshopStyle.accent)
                }
                VStack(alignment: .leading, spacing: 3) {
                    Text(model.status.isEmpty ? "Ready when you are" : model.status).font(.callout).lineLimit(2)
                    if let event = model.events.last, let epoch = event.epoch {
                        Text("Epoch \(epoch)" + (event.update.map { " · Update \($0)/\(event.totalUpdates ?? $0)" } ?? "") +
                             (event.loss.map { " · Loss \($0.formatted(.number.precision(.fractionLength(5))))" } ?? "") +
                             (event.learningRate.map { " · LR \($0.formatted(.number.precision(.fractionLength(1...7))))" } ?? ""))
                            .font(.caption).foregroundStyle(.secondary).monospacedDigit()
                    }
                }
                Spacer()
                if model.trainingActive {
                    Button("Save Checkpoint") { model.saveCheckpoint() }
                        .disabled(model.checkpointPending || model.stoppingAfterSave)
                        .accessibilityIdentifier("save-training-checkpoint")
                    Button("Save & Stop") { model.saveAndStop() }
                        .disabled(model.stoppingAfterSave)
                        .accessibilityIdentifier("save-and-stop-training")
                    Menu {
                        Button("Stop Current Operation") { model.cancel() }
                    } label: { Image(systemName: "ellipsis") }
                        .menuStyle(.borderlessButton).fixedSize()
                        .disabled(model.stoppingAfterSave)
                } else if model.busy {
                    Button("Stop") { model.cancel() }.disabled(model.sourceFolder == nil)
                } else {
                    Label("Native Swift · Apple GPU", systemImage: "desktopcomputer")
                        .font(.callout).foregroundStyle(.secondary)
                    Button { model.train() } label: {
                        Label("Train LoRA", systemImage: "play.fill")
                    }.buttonStyle(.borderedProminent).controlSize(.large)
                        .disabled(!model.canTrain).accessibilityIdentifier("train-adapter")
                }
            }.padding(.horizontal, 28).padding(.vertical, 14)
        }.background(.background)
    }

    private var selectedDatasets: [WorkshopDataset] {
        model.availableDatasets.filter { model.selectedDatasetID.isEmpty || $0.id == model.selectedDatasetID }
    }
    private var plannedUpdates: Int {
        let effective = max(1, model.options.effectiveBatchSize)
        return selectedDatasets.reduce(0) { total, dataset in
            total + dataset.pairedCounts.values.reduce(0) { result, count in
                result + (count / effective + (count % effective == 0 ? 0 : 1)) * model.options.epochs
            }
        }
    }
    private func decimalField(_ value: Binding<Double>) -> some View {
        TextField("Value", value: value, format: .number.precision(.fractionLength(0...7)))
            .textFieldStyle(.roundedBorder).frame(maxWidth: 180)
    }
    private var hasDepthPairs: Bool { selectedDatasets.contains { ($0.pairedCounts["depth"] ?? 0) > 0 } }
    private func mapTitle(_ kind: String) -> String {
        switch kind { case "depth": "Depth"; case "roughness": "Roughness"; default: "Normal" }
    }
}

private extension View {
    func sidebarCard() -> some View {
        padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.background, in: RoundedRectangle(cornerRadius: WorkshopStyle.radius))
            .overlay(RoundedRectangle(cornerRadius: WorkshopStyle.radius).strokeBorder(Color(nsColor: .separatorColor).opacity(0.55)))
    }
}
