import SwiftUI
import WorkshopCore

struct WorkshopView: View {
    @Bindable var model: WorkshopModel
    var startsAutomatically = true
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
                Toggle("Exaggerate features", isOn: $model.options.exaggerateFeatures)
                    .disabled(!hasDepthPairs)
                Text("Exaggerate features strengthens depth relief. Crop positions vary wherever the image has room.")
                    .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }.toggleStyle(.checkbox)
            DisclosureGroup("Advanced options", isExpanded: $advancedOptions) {
                Grid(alignment: .leading, horizontalSpacing: 20, verticalSpacing: 12) {
                    GridRow {
                        Text("Learning rate").foregroundStyle(.secondary)
                        TextField("Learning rate", value: $model.options.learningRate,
                                  format: .number.precision(.fractionLength(1...7)))
                            .textFieldStyle(.roundedBorder).frame(maxWidth: 180)
                    }
                    GridRow {
                        Text("LoRA rank").foregroundStyle(.secondary)
                        Picker("LoRA rank", selection: $model.options.rank) {
                            ForEach([2, 4, 8, 16, 32], id: \.self) { rank in Text(rank, format: .number).tag(rank) }
                        }.labelsHidden().frame(maxWidth: 180)
                    }
                }.padding(.top, 12)
            }
            if let plan = model.plan, !plan.requiresClassification, model.availableDatasets.isEmpty {
                Label("No matching map pairs at this image size. Check the matching files or choose a smaller size.", systemImage: "info.circle")
                    .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
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
                        Text("Epoch \(epoch)" + (event.loss.map { " · Loss \($0.formatted(.number.precision(.fractionLength(5))))" } ?? ""))
                            .font(.caption).foregroundStyle(.secondary).monospacedDigit()
                    }
                }
                Spacer()
                if model.busy {
                    Button("Stop") { model.cancel() }.disabled(model.sourceFolder == nil)
                } else {
                    Label("Local training", systemImage: "desktopcomputer")
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
