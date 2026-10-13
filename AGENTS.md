# Loom Workshop

This is fresh code. Do not copy, call, or import Texture Studio's retired trainer, workbench, dataset registry, checkpoints or material-tool apps. Use the new Texture Loom package contract.

The installed Texture Studio application supplies the fixed base models and classifier. Never replace or modify them. A LoRA belongs to that exact model ID, version, dataset version and base-weight hash.

The selected image folder owns Workshop settings, runtime, runs and exports under `.loom-workshop`. Read current physical files; keep image identity and classification metadata on those files. A different folder is a different project. Training uses actual paired diffuse and target maps, with native precision and aligned crops. Do not invent target maps for photographs-only folders.

Present one guided workspace. Categorize with the installed classifier, offer one dataset or all compatible datasets, and export a real LoRA package for Texture Studio. Verify installation before reporting success. Preserve original image bytes and base weights. Support progress and cancellation.

Smoke tests and small training checks are execution evidence, not model-quality qualification.

Before finishing completed work, commit and push task-related changes to this
repository's configured origin and verify the remote branch matches the local
commit. Local checkouts may not survive between runs. Preserve unrelated work,
never force-push shared history, and report any push failure. The user has given
standing authorization for these task-scoped commits and pushes.
