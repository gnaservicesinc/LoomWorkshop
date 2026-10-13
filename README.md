# Loom Workshop

Loom Workshop trains a LoRA for the Texture Loom model built into your installed Texture Studio. It is a fresh, independent macOS app.

Choose Texture Studio, then a folder containing diffuse images with matching depth, roughness, or normal maps. The installed classifier groups the images into datasets automatically. Train one dataset or all available datasets, then export a `.loomlora` file or install it in Texture Studio.

Texture Studio is checked in `/Applications` first. Choose its application bundle if it is installed elsewhere. That build must contain a complete Texture Loom package produced by Loom Factory; Workshop does not substitute another base model.

Training settings, its Python environment, logs, and adapters stay under `.loom-workshop` in the chosen image folder. Image metadata preserves pair identity across renames. Changing the folder clears the active project; reopening a folder restores its settings. Original image pixels and the installed base weights are preserved.

The image-order checkbox selects a fixed order or a new order each epoch. Crop positions remain random where the source has room. Random flipping applies to aligned pairs. **Exaggerate features** strengthens depth relief during training.

Build with Swift 6.2 on macOS 26 or later:

```sh
swift test
Scripts/build-app.sh
Scripts/install-app.sh
```

`build-app.sh` produces a signed local bundle under `.build/application`. The installer retains any previous app as a backup and refuses to replace a running app. Runtime setup is automatic and stored in the image folder. Exported adapters are tied to the exact installed base identity, version, and weight checksums.

Native SwiftUI layout tests render the startup and populated workflow at the supported window sizes. These are view renderings; they do not certify trained texture quality. Assess a trained model on new materials before deploying it.
