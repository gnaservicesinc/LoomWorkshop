# Loom Workshop runtime

Workshop uses the exact Texture Loom package built into the selected Texture
Studio installation. It verifies the base ID, version and registered file hashes,
classifies diffuse images with that installed classifier, and trains category
LoRA convolution deltas while freezing every original base weight.

The fixed classifier may use a native Create ML image feature or an imported
RGB tensor input. Its declared interface and source labels come from the native
model; labels map to the package's dataset IDs. A classifier does not need
training weights, and Workshop cannot replace it independently of the base.

The architecture and pixel helpers are shared with the fresh Loom Factory
implementation. No retired Texture Studio trainer code is used.

The app provisions Python and pinned dependencies inside the selected image
folder's `.loom-workshop` directory using `setup_runtime.sh <image-folder>`.

## Protocol

```
PYTHONPATH=<Runtime> <private-python> -m loom_workshop plan --request request.json
PYTHONPATH=<Runtime> <private-python> -m loom_workshop train --request request.json
```

JSON schema `loom-workshop-training-v1`: `source_root`, `output_directory`
(inside source `.loom-workshop`), `base_root`, `base_id`, `base_version`,
`adapter_name`, `selected_category_ids` (empty means all classified datasets),
`samples` containing `{id,input,targets:{depth:path,roughness:path,normal:path}}`
and optional `normal_convention` and `category_id`. A supplied category is
accepted only when the current file metadata agrees. Current Factory category
tags are authoritative; a category absent from the installed base is reported
explicitly. Only images without a category are classified. Classification is
stamped on the physical image with the exact base identity and source byte
identity. Changing or clearing a Factory category invalidates the previous
stamp. Settings include `image_size`, `epochs`,
`batch_size`, `learning_rate`, `rank`, `alpha`, `shuffle_each_epoch`,
`flip_crops`, `rotate_crops`, depth-only `exaggerate_features`, `seed`, `device`.

Stdout JSON lines expose `classification`, `plan`, `status`, `started`,
`progress`, `completed`, `cancelled`, `error`. Plan provides category counts and
assigned sample IDs. SIGTERM/SIGINT cancel cooperatively. Completed result
`package_directory` points to a folder that the Swift app exports as `.loomlora`.

The adapter manifest uses `texture-loom.adapter.v1`, exact `baseModel` ID and
version, exact category versions and base weight hashes. Per-map safetensors
contain only LoRA down/up weights. Native Core ML exports fuse those learned
deltas into a deployment copy; original installed weights are never changed.
Every export runs a real prediction comparison against the adapted network.

Training uses actual aligned diffuse/target map pairs at native crop pixels.
Current file presence, role, pair, category and normal convention are checked
before training batches and again before export. Image bytes and all installed
base files are checked before publishing a completed adapter. Moving, removing,
retagging or changing a sampled file halts the run rather than broadening its
dataset or exporting stale results. Small originals remain classifiable but
do not count as training crops at a larger chosen size.

Production training uses the same bounded decoded-pixel cache as Loom Factory:
256 MiB per run, configurable in backend requests as `image_cache_megabytes`
(0…2048, with 0 disabling retention). File and link identities are checked on
every access, and current category/pair metadata is checked before every batch.
Only immutable decoded pixels are reused; crop selection and augmentations stay
random. Loss and gradient norm share one small GPU read before an update. Native
precision, finite-value checks, exact frozen-base checks and export compatibility
validation remain enabled; normal training runs no profiler or anomaly detector.
Photos alone can be classified but cannot provide supervised material-map
learning targets. The runtime does not fabricate targets or claim that copying
the base's predictions trains useful LoRA features. Successful updates and
export do not establish quality on unseen materials.

Tests: `<private-python> -m unittest discover -s RuntimeTests -v`.
