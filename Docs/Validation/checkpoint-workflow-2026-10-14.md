# Live checkpoint and continuation validation

Workshop now exposes Save Checkpoint, Save & Stop and Save & Quit. Requests run
on the serial training owner after a completed optimizer update. A failed manual
save reports the error and keeps training active; quitting waits for a successful
save receipt. Saved runs and checkpoint folders are accessible from the workspace.

New v3 checkpoints preserve adapter weights, optimizer state, frozen-base and
source identities, exact sample order, augmentation state, epoch metrics and the
next accumulation-window position. Resume Saved Run restores the saved settings.
Older checkpoints lack a safe continuation cursor and are not presented as exact
resumable runs. Installed base weights and source image bytes remain untouched.

The audit also fixed losing completed adapter publication after later failures,
nonfinite-loss handling before updates, post-save refresh reporting and canonical
macOS `/var` versus `/private/var` paths in source checks and package artifacts.
The production build pins the native trainer to Factory commit
`3a8801666df2de6c44b7b478d09f9078a1fee8de`.

The full suite includes 36 core tests and 13 UI tests. Two core checks require
an actual installation or real external base fixture and are skipped by default.
The new partial-epoch test compares uninterrupted training against manual saving,
Save & Stop and exact continuation, including shuffled samples, per-sample flips
and rotations, accumulation, AdamW, warmup and cosine scheduling. Final adapter
weights match within 1e-7. Save receipts, failed saves, retries and quit decisions
have separate tests. Logs and installed-bundle verification are retained outside
the repository in `~/Library/Application Support/Loom Factory/Validation/2026-10-14-checkpoint-audit`.

These tests verify execution and continuation, not learned material quality.
Desktop automation failed with `Sky Computer Use native pipe startup failed`;
offscreen renders and model-state tests are not interactive desktop evidence.
