# Loom Workshop

Loom Workshop is a native macOS app that trains material LoRAs for the Texture Loom model built into your installed [Texture Studio](https://github.com/gnaservicesinc/Texture_Studio).

Full model training, including experimental RGB noise repair and lighting correction, belongs to [Loom Factory](https://github.com/gnaservicesinc/LoomFactory). Workshop currently trains depth, roughness and normal adapters; photo cleanup adapters are not supported.

Choose Texture Studio, then a folder containing diffuse images with matching depth, roughness, or normal maps. The installed classifier groups the images into datasets automatically. Train one dataset or all available datasets, then export a `.loomlora` file or install it in Texture Studio.

Known diffuse/color and map filenames are recognized automatically. Bump maps never substitute for depth, unrecognized files never default to diffuse, and normal training uses OpenGL maps. Files marked ignored in Loom Factory stay out of Workshop training.

Review flags and notes are informational. Training uses current files, dataset tags and matching map roles; it does not require manual review. Updating a review flag or note during a run does not change training membership.

Texture Studio is checked in `/Applications` first. Choose its application bundle if it is installed elsewhere. That build must contain a complete Texture Loom package produced by Loom Factory; Workshop does not substitute another base model.

Training settings, logs, checkpoints, and adapters stay under `.loom-workshop` in the chosen image folder. Image metadata preserves pair identity across renames. Changing the folder clears the active project; reopening a folder restores its settings. Original image pixels and the installed base weights are preserved.

The image-order checkbox selects a fixed order or a new order each epoch. Crop positions and flips/quarter-turn rotations have separate timing controls: every presentation, once per epoch, or once per run. Positions remain random where the source has room. Normal vectors transform with the paired image. **Exaggerate features** strengthens depth relief during training.

Advanced options expose AdamW, Adam, SGD with momentum, and Lion; learning rate and constant/cosine scheduling with epoch warmup; batch size, gradient accumulation, clipping, decay and optimizer coefficients; LoRA rank and alpha; random seed and Apple GPU selection. The summary derives frozen-base and trainable-LoRA parameter counts from each installed model descriptor. Accumulated gradients are weighted by the actual number of pairs, including the final partial update. Lion generally needs a lower learning rate than AdamW.

Checkpoint cadence saves LoRA deltas, native safetensor optimizer state and deterministic sampling state under the run's `checkpoints` folder at epoch boundaries. The first completed update is also saved. **Save Checkpoint** (Command-S) saves at the next complete update and continues training; **Save & Stop** stops only after that save succeeds. A failed manual save reports the error and keeps the run active. Quitting during training offers **Keep Training** or **Save & Quit**.

Reopening the image folder shows unfinished saved runs with **Resume Saved Run**, **Show Saved Checkpoint**, and **Open Runs Folder**. Continuation restores the exact optimizer update, partial-epoch image order, crop/augmentation state and saved settings. It requires unchanged source pixels, membership metadata and installed base identities and checksums. Older checkpoints without a continuation position remain on disk but cannot be resumed safely. A completed, validated adapter is required for installation; checkpoint files alone are not an installable LoRA. Workshop does not currently expose held-out early stopping. Native sampling retains crop and augmentation refresh timing and is deterministic, with a new random sequence from the retired runtime.

Build with Swift 6.2 on macOS 26 or later:

```sh
swift test
Scripts/build-app.sh
Scripts/install-app.sh
```

`build-app.sh` produces a signed local bundle under `.build/application`. The installer retains any previous app as a backup and refuses to replace a running app. The app trains natively with Swift and MPSGraph on the Apple GPU; it does not create or launch a Python environment. Exported adapters are tied to the exact installed base identity, version, and weight checksums.

SwiftPM fetches the native training library from a pinned Loom Factory revision,
so Workshop builds from its own checkout. When developing both repositories,
set `LOOM_NATIVE_DEVELOPMENT_PATH` to the local Loom Factory checkout to use its
current sources explicitly.

Native SwiftUI layout tests render the startup and populated workflow at the supported window sizes. These are view renderings; they do not certify trained texture quality. Assess a trained model on new materials before deploying it.

The native LoRA graph freezes every installed convolution and bias. Only low-rank down/up tensors receive gradients. Exports include those adapter safetensors plus a fused Core ML model, checked against native inference before publishing the adapter manifest. Stop cancels a shared native token from the responsive project actor and returns after the current bounded GPU operation. CPU training settings saved by older versions are retained, but require selecting Apple GPU because this backend has no CPU trainer.
