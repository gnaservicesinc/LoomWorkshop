"""Classify paired inputs, train only low-rank deltas, export strict-base adapters."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
import uuid

import coremltools as ct
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file, save_file

from .base_models import MaterialMapNetwork
from .images import classifier_image, classifier_pixels, classifier_tensor, image_dimensions, native_pair
from .lora import adapter_weights, attach_lora, frozen_base_weights, fuse_lora
from .package import file_sha, load_base, resolve
from .metadata import FACTORY_KEY, current_category, factory_category_id, membership_signature, prepare_membership, read_tag, stamp_category
from .pixel_cache import DecodedImageCache


class Cancelled(Exception):
    pass


class Cancellation:
    requested = False
    def signal(self, *_):
        self.requested = True
    def check(self):
        if self.requested:
            raise Cancelled()


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}, allow_nan=False), flush=True)


def write_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def integer(request, field, default, low, high):
    value = request.get(field, default)
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{field} must be an integer from {low} to {high}.")
    return value


def prepare(request):
    if request.get("schema") != "loom-workshop-training-v1":
        raise ValueError("Unsupported Loom Workshop request schema.")
    source = Path(os.path.abspath(os.path.expanduser(request["source_root"])))
    output = Path(os.path.abspath(os.path.expanduser(request["output_directory"])))
    if not source.is_dir() or not output.resolve().is_relative_to(source.resolve() / ".loom-workshop"):
        raise ValueError("Choose an existing image folder; run output must stay in its .loom-workshop directory.")
    emit("status", message="Checking the installed Texture Loom base…")
    base, manifest = load_base(request["base_root"], request["base_id"], request["base_version"])
    samples = request.get("samples", [])
    if not samples or len({sample["id"] for sample in samples}) != len(samples):
        raise ValueError("Choose a folder with distinct diffuse images.")
    fingerprints = {}
    for sample in samples:
        fields = [(sample, "input")] + [(sample.get("targets", {}), key) for key in sample.get("targets", {})]
        for owner, key in fields:
            path = Path(os.path.abspath(os.path.expanduser(owner[key])))
            if not path.is_relative_to(source) or any(part.startswith(".loom-") for part in path.relative_to(source).parts):
                raise ValueError(f"A training image is outside the selected image folder: {path.name}")
            if not path.is_file():
                raise ValueError(f"A training image was moved or deleted: {path.name}")
            owner[key] = str(path)
            fingerprints[str(path)] = file_sha(path)
    prepare_membership(samples)
    size = integer(request, "image_size", 1024, 32, 8192)
    integer(request, "epochs", 10, 1, 100000)
    integer(request, "batch_size", 1, 1, 256)
    integer(request, "image_cache_megabytes", 256, 0, 2048)
    rank = integer(request, "rank", 4, 1, 128)
    for field, value in (("alpha", request.get("alpha", rank)), ("learning_rate", request.get("learning_rate", 0.001))):
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Choose a positive finite {field}.")
    output.mkdir(parents=True, exist_ok=True)
    return source, output, base, manifest, samples, fingerprints, size


def classifier_model(base, manifest):
    descriptor = manifest["classifier"]
    path = resolve(base, descriptor["path"])
    model = ct.models.CompiledMLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY) if path.suffix == ".mlmodelc" else ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)
    interface = {"input": descriptor.get("inputName", "rgb"), "output": descriptor.get("outputName", "dataset")}
    if descriptor.get("architecture") == "create-ml-image-classifier-v1":
        if not all(isinstance(interface[key], str) and interface[key] for key in ("input", "output")):
            raise ValueError("The installed classifier has an unsupported input or output name.")
        kind = descriptor.get("inputType", "image")
        if kind == "image":
            width, height = descriptor.get("inputWidth"), descriptor.get("inputHeight")
            if any(isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 8192 for value in (width, height)):
                raise ValueError("The installed image classifier is missing its native input dimensions. Rebuild Texture Studio with a complete Loom Factory export.")
            if descriptor.get("inputColorSpace") != "RGB":
                raise ValueError("The installed image classifier requires RGB input.")
            interface.update(kind="image", width=width, height=height)
        elif kind == "multiArray":
            shape = descriptor.get("inputShape")
            if (not isinstance(shape, list) or len(shape) != 4 or shape[:2] != [1, 3]
                    or any(isinstance(value, bool) or not isinstance(value, int) for value in shape)
                    or not all(1 <= value <= 8192 for value in shape[2:])):
                raise ValueError("The installed classifier requires a supported RGB tensor input.")
            interface.update(kind="tensor", width=shape[3], height=shape[2])
        else:
            raise ValueError("The installed classifier requires an image or RGB tensor input.")
        if path.suffix != ".mlmodelc":
            description = model.get_spec().description
            feature = next((value for value in description.input if value.name == interface["input"]), None)
            output = next((value for value in description.output if value.name == interface["output"]), None)
            if (len(description.input) != 1 or feature is None or output is None
                    or output.type.WhichOneof("Type") != "stringType"):
                raise ValueError("The installed classifier does not match its declared interface.")
            if kind == "image":
                if (feature.type.WhichOneof("Type") != "imageType"
                        or feature.type.imageType.width != interface["width"]
                        or feature.type.imageType.height != interface["height"]
                        or feature.type.imageType.colorSpace not in (
                            ct.proto.FeatureTypes_pb2.ImageFeatureType.RGB,
                            ct.proto.FeatureTypes_pb2.ImageFeatureType.BGR)):
                    raise ValueError("The installed classifier does not match its declared RGB image interface.")
            elif (feature.type.WhichOneof("Type") != "multiArrayType"
                    or list(feature.type.multiArrayType.shape) != descriptor["inputShape"]
                    or feature.type.multiArrayType.dataType != ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32):
                raise ValueError("The installed classifier does not match its declared RGB tensor interface.")
    else:
        if not descriptor.get("weightsPath"):
            raise ValueError("The installed tensor classifier is missing its training descriptor.")
        metadata_path = resolve(base, str(Path(descriptor["weightsPath"]).parent / "model.json"))
        metadata = json.loads(metadata_path.read_text())
        image_size = metadata.get("image_size", 256)
        if isinstance(image_size, bool) or not isinstance(image_size, int) or not 1 <= image_size <= 8192:
            raise ValueError("The installed tensor classifier has invalid input dimensions.")
        interface.update(kind="tensor", size=image_size)
    return model, interface


def build_plan(request, base, manifest, samples, cancellation):
    emit("status", message="Sorting images with Texture Studio's classifier…")
    classifier = None
    categories = {category["id"]: category for category in manifest["categories"]}
    selected = set(request.get("selected_category_ids", []))
    if not selected.issubset(categories):
        raise ValueError("A selected dataset is unavailable in this Texture Studio version.")
    groups, assignments = {}, []
    for index, sample in enumerate(samples):
        cancellation.check()
        category_id = current_category(sample["input"], manifest)
        requested_category = sample.get("category_id")
        if requested_category is not None and requested_category != category_id:
            raise ValueError("An image's dataset tag changed. Refresh the source folder before training.")
        signature = membership_signature(sample["input"])
        if category_id is None:
            if classifier is None:
                classifier, interface = classifier_model(base, manifest)
            if interface["kind"] == "image":
                pixels = classifier_pixels(sample["input"], interface["width"], interface["height"])
            elif "width" in interface:
                pixels = classifier_tensor(sample["input"], interface["width"], interface["height"])
            else:
                pixels = classifier_image(sample["input"], interface["size"]).numpy()[None]
            result = classifier.predict({interface["input"]: pixels})
            source_label = result.get(interface["output"])
            matching = next((label for label in manifest["classifier"]["labels"]
                             if label.get("sourceLabel", label["id"]) == source_label), None)
            category_id = matching["id"] if matching else None
        if membership_signature(sample["input"]) != signature:
            raise ValueError("An image's metadata changed while it was being classified. Refresh the source folder.")
        if category_id not in categories:
            raise ValueError("The installed classifier returned a dataset missing from its base model.")
        stamp_category(sample["input"], category_id, manifest)
        category = categories[category_id]
        assignments.append({"id": sample["id"], "category_id": category_id, "category_name": category["name"]})
        if not selected or category_id in selected:
            models = {model["map"]: model for model in category["models"]}
            dimensions = image_dimensions(sample["input"])
            for map_kind, target in sample.get("targets", {}).items():
                target_tag = read_tag(target, FACTORY_KEY)
                if target_tag and factory_category_id(target_tag) not in (None, category_id):
                    continue
                if image_dimensions(target) != dimensions:
                    raise ValueError(f"Paired maps need the same native dimensions: {sample['id']}")
                if map_kind in models and min(dimensions) >= request.get("image_size", 1024):
                    groups.setdefault((category_id, map_kind), []).append({"id": sample["id"], "input": sample["input"],
                        "target": target, "normal_convention": sample.get("normal_convention", "opengl")})
        emit("classification", sample_id=sample["id"], category_id=category_id, category_name=category["name"],
             progress=(index + 1) / len(samples), message=f"Sorted image {index + 1} of {len(samples)}")
    counts = [{"id": category_id, "name": category["name"],
               "image_count": sum(row["category_id"] == category_id for row in assignments),
               "maps": {kind: len(items) for (key, kind), items in groups.items() if key == category_id}}
              for category_id, category in categories.items() if any(row["category_id"] == category_id for row in assignments)]
    emit("plan", categories=counts, assignments=assignments, paired_map_count=sum(len(group) for group in groups.values()),
         message="Paired target maps are needed for LoRA learning." if not groups else "The paired images are ready for category LoRA training.")
    return groups, categories


def check_current_membership(paths, signatures, identities, fingerprints, check_bytes=False):
    for path in paths:
        if not Path(path).is_file():
            raise ValueError(f"A training image was moved or deleted: {Path(path).name}")
        if membership_signature(path) != signatures[path]:
            raise ValueError("An image's dataset or pair metadata changed. Refresh the source folder before training.")
        from .metadata import byte_identity
        if byte_identity(path) != identities[path] or (check_bytes and file_sha(path) != fingerprints[path]):
            raise ValueError("An input image changed during training. Refresh the folder and start another run.")


def check_pair_selection(category_id, map_kind, members, manifest):
    for sample in members:
        if current_category(sample["input"], manifest) != category_id:
            raise ValueError("An image's dataset tag changed. Refresh the source folder before training.")
        target_tag = read_tag(sample["target"], FACTORY_KEY)
        if target_tag and factory_category_id(target_tag) not in (None, category_id):
            raise ValueError("A target's dataset tag changed. Refresh the source folder before training.")
        prepare_membership([{**sample, "targets": {map_kind: sample["target"]}}])


def construct_base(base, category, map_kind):
    artifact = next(model for model in category["models"] if model["map"] == map_kind)
    if artifact["architecture"] != MaterialMapNetwork.architecture:
        raise ValueError("This custom base architecture needs a compatible Loom Workshop version.")
    weights_path = resolve(base, artifact["weightsPath"])
    descriptor = json.loads(resolve(base, str(Path(artifact["weightsPath"]).parent / "model.json")).read_text())
    config = descriptor.get("architecture_config") or descriptor.get("architecture")
    if not isinstance(config, dict) or config.get("id") != MaterialMapNetwork.architecture:
        raise ValueError("This installed base is missing the architecture configuration for LoRA training.")
    model = MaterialMapNetwork(map_kind, config["width"], config["blocks"])
    model.load_state_dict(load_file(str(weights_path)), strict=True)
    return model, weights_path


def export_coreml(model, path, image_size):
    with redirect_stdout(sys.stderr):
        model = model.cpu().eval()
        trace = torch.jit.trace(model, torch.zeros(1, 3, 64, 64))
        converted = ct.convert(trace, convert_to="mlprogram", compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.macOS13,
            inputs=[ct.TensorType(name="rgb", shape=(1, 3, ct.RangeDim(32, 8192, default=image_size), ct.RangeDim(32, 8192, default=image_size)))],
            outputs=[ct.TensorType(name="material_map")])
        converted.short_description = "Category LoRA fused with its exact Texture Loom base"
        converted.save(str(path))
    return converted


def device_for(value):
    if value == "auto":
        value = "mps" if torch.backends.mps.is_available() else "cpu"
    if value not in ("mps", "cpu") or (value == "mps" and not torch.backends.mps.is_available()):
        raise ValueError("Choose an available CPU or Apple GPU training device.")
    if value == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    return torch.device(value)


def train(request, cancellation=None, plan_only=False):
    cancellation = cancellation or Cancellation()
    source, output, base, manifest, samples, fingerprints, image_size = prepare(request)
    groups, categories = build_plan(request, base, manifest, samples, cancellation)
    if plan_only:
        return {"groups": groups, "categories": categories}
    if not groups:
        raise ValueError("These images have no matching paired target maps for the installed model. Add matching depth, roughness or normal maps; photos alone cannot train a supervised material LoRA.")
    active_paths = {sample[field] for members in groups.values() for sample in members for field in ("input", "target")}
    fingerprints = {path: sha for path, sha in fingerprints.items() if path in active_paths}
    for (category_id, map_kind), members in groups.items():
        check_pair_selection(category_id, map_kind, members, manifest)
    from .metadata import byte_identity
    signatures = {path: membership_signature(path) for path in fingerprints}
    identities = {path: byte_identity(path) for path in fingerprints}
    check_current_membership(fingerprints, signatures, identities, fingerprints, check_bytes=True)
    package = output / "adapter"
    if (package / "manifest.json").exists():
        raise ValueError("This run already has an adapter. Start a new run to preserve it.")
    package.mkdir(exist_ok=True)
    seed = request.get("seed", 42)
    torch.manual_seed(seed)
    device = device_for(request.get("device", "auto"))
    if device.type == "mps":
        torch.mps.manual_seed(seed)
    rank, alpha = request.get("rank", 4), request.get("alpha", request.get("rank", 4))
    epochs, batch_size = request.get("epochs", 10), request.get("batch_size", 1)
    order_rng, crop_rng = random.Random(seed), random.Random(seed + 1)
    image_cache = DecodedImageCache(request.get("image_cache_megabytes", 256) * 1024 * 1024)
    adapters, total_updates, update = {}, epochs * sum(math.ceil(len(group) / batch_size) for group in groups.values()), 0
    emit("started", total_updates=total_updates, category_count=len({category for category, _ in groups}), device=device.type)
    try:
        for (category_id, map_kind), members in sorted(groups.items()):
            cancellation.check()
            category = categories[category_id]
            model, base_weights_path = construct_base(base, category, map_kind)
            targets = attach_lora(model, rank, alpha)
            before = frozen_base_weights(model)
            model.to(device)
            optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=request.get("learning_rate", 0.001))
            losses = []
            for epoch in range(epochs):
                order = list(range(len(members)))
                if request.get("shuffle_each_epoch", True):
                    order_rng.shuffle(order)
                for start in range(0, len(order), batch_size):
                    cancellation.check()
                    selected = [members[index] for index in order[start:start + batch_size]]
                    paths = {sample[field] for sample in selected for field in ("input", "target")}
                    check_current_membership(paths, signatures, identities, fingerprints)
                    check_pair_selection(category_id, map_kind, selected, manifest)
                    pairs = [native_pair(sample, image_size, crop_rng, map_kind, request.get("flip_crops", True),
                        request.get("rotate_crops", False), map_kind == "depth" and request.get("exaggerate_features", False),
                        cache=image_cache) for sample in selected]
                    rgb = torch.stack([pair[0] for pair in pairs]).to(device)
                    target = torch.stack([pair[1] for pair in pairs]).to(device)
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(rgb)
                    if map_kind == "normal":
                        loss = (1 - (F.normalize(prediction * 2 - 1, dim=1, eps=1e-6) * F.normalize(target * 2 - 1, dim=1, eps=1e-6)).sum(dim=1)).mean()
                    else:
                        loss = F.l1_loss(prediction, target)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 1.0)
                    loss_number, gradient_number = torch.stack((loss.detach(), norm.detach())).cpu().tolist()
                    if not math.isfinite(loss_number):
                        raise ValueError("LoRA training produced a non-finite loss; no adapter was published.")
                    if not math.isfinite(gradient_number):
                        raise ValueError("LoRA gradients became non-finite; no adapter was published.")
                    optimizer.step()
                    update += 1
                    losses.append(loss_number)
                    emit("progress", update=update, total_updates=total_updates, progress=update / total_updates,
                         epoch=epoch + 1, loss=losses[-1], category_id=category_id, map=map_kind,
                         message=f"{category['name']} · {map_kind} · epoch {epoch + 1} of {epochs}")
                    del rgb, target, prediction, loss, pairs
            after = frozen_base_weights(model)
            if any(not torch.equal(before[key], after[key]) for key in before):
                raise ValueError("Frozen base weights changed; adapter export was stopped.")
            cancellation.check()
            folder = package / "models" / ("dataset-" + __import__("hashlib").sha256(category_id.encode()).hexdigest()[:16]) / map_kind
            folder.mkdir(parents=True, exist_ok=True)
            delta = adapter_weights(model)
            if not all(torch.isfinite(value).all() for value in delta.values()):
                raise ValueError("LoRA weights contain non-finite values.")
            save_file(delta, str(folder / "adapter.safetensors"), metadata={"schema": "texture-loom.lora-weights.v1", "rank": str(rank), "alpha": str(alpha)})
            fused = fuse_lora(model)
            native = export_coreml(fused, folder / "adapter.mlpackage", image_size)
            # Real runtime compatibility proof; an untested native export cannot
            # silently replace this installed model's category map.
            proof_rgb = np.random.default_rng(seed).random((1, 3, 64, 64), dtype=np.float32)
            expected = fused(torch.from_numpy(proof_rgb)).detach().numpy()
            actual = native.predict({"rgb": proof_rgb})["material_map"]
            np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=3e-4)
            relative = str(folder.relative_to(package))
            entry = {"map": map_kind, "format": "coreml", "path": relative + "/adapter.mlpackage",
                     "weightsPath": relative + "/adapter.safetensors", "baseSHA256": file_sha(base_weights_path)}
            adapters.setdefault(category_id, []).append(entry)
            write_json(folder / "adapter.json", {"schema": "loom-workshop-lora-v1", "base_id": manifest["id"],
                "base_version": manifest["version"], "category_id": category_id, "category_version": category["version"],
                "map": map_kind, "base_sha256": entry["baseSHA256"], "rank": rank, "alpha": alpha, "targets": targets,
                "architecture_config": fused.descriptor(), "epochs": epochs, "updates": len(losses), "final_loss": losses[-1],
                "base_weights_frozen": True, "coreml_maximum_absolute_difference": float(np.max(np.abs(actual - expected))),
                "decoded_image_cache": image_cache.statistics(),
                "quality_note": "Adapter completion does not measure prediction quality on unseen materials."})
            emit("status", message=f"{category['name']} {map_kind} LoRA exported and checked.")
            del model, optimizer, fused
        cancellation.check()
        check_current_membership(fingerprints, signatures, identities, fingerprints, check_bytes=True)
        for (category_id, map_kind), members in groups.items():
            check_pair_selection(category_id, map_kind, members, manifest)
        # Revalidate the immutable installed base before binding the final adapter.
        _, current_manifest = load_base(base, manifest["id"], manifest["version"])
        if current_manifest != manifest:
            raise ValueError("Texture Studio's built-in package changed during training. Reload the application before training.")
        files = [{"path": str(path.relative_to(package)), "sha256": file_sha(path)}
                 for path in sorted(package.rglob("*")) if path.is_file()]
        assembled = {"schema": "texture-loom.adapter.v1", "id": str(uuid.uuid4()),
                     "name": request.get("adapter_name", "My Texture LoRA"), "version": "1.0.0",
                     "baseModel": {"id": manifest["id"], "version": manifest["version"]},
                     "categories": [{"id": key, "version": categories[key]["version"], "adapters": value} for key, value in sorted(adapters.items())],
                     "files": files, "createdAt": datetime.now(timezone.utc).isoformat()}
        write_json(package / "manifest.json", assembled)
        result = {"status": "completed", "package_directory": str(package), "updates": update,
                  "base_weights_frozen": True, "originals_preserved": True}
        write_json(output / "result.json", result)
        emit("completed", **result, message="Your category LoRA is ready for Texture Studio.")
        return assembled
    except Cancelled:
        write_json(output / "result.json", {"status": "cancelled", "updates": update})
        emit("cancelled", message="LoRA training stopped. The installed base and original images were preserved.")
        raise
    except Exception as error:
        write_json(output / "result.json", {"status": "failed", "updates": update, "message": str(error)})
        raise


def run_cli(command, request_path):
    cancellation = Cancellation()
    for kind in (signal.SIGTERM, signal.SIGINT):
        signal.signal(kind, cancellation.signal)
    try:
        request = json.loads(Path(request_path).read_text())
        train(request, cancellation, plan_only=command == "plan")
        return 0
    except Cancelled:
        return 130
    except Exception as error:
        emit("error", message=str(error), error_type=type(error).__name__)
        return 1
