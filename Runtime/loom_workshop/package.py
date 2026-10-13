"""Strict installed-base identities and registered-file integrity."""
import hashlib
import json
from pathlib import Path
import uuid


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(root, relative):
    parts = str(relative).split("/")
    if not relative or str(relative).startswith("/") or "\\" in str(relative) or any(part in ("", ".", "..") for part in parts):
        raise ValueError("A model package path is outside its own folder.")
    cursor = root
    for part in parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("Installed model artifacts cannot be symbolic links.")
    if not cursor.resolve().is_relative_to(root.resolve()):
        raise ValueError("A model package path escapes its own folder.")
    return cursor


def load_base(root, expected_id, expected_version):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != "texture-loom.package.v1":
        raise ValueError("This Texture Studio installation has no compatible Texture Loom base package.")
    uuid.UUID(manifest["id"])
    if manifest["id"] != expected_id or manifest["version"] != expected_version:
        raise ValueError("Texture Studio's base model changed. Reload the installed application before training.")
    files = manifest.get("files", [])
    inventory = {entry["path"]: entry["sha256"] for entry in files}
    if not inventory or len(inventory) != len(files):
        raise ValueError("The installed base package has an empty or repeated file inventory.")
    for name, sha in inventory.items():
        file = resolve(root, name)
        if not file.is_file() or file_sha(file) != sha.lower():
            raise ValueError(f"An installed base file is missing or changed: {name}")
    categories = manifest.get("categories", [])
    labels = manifest.get("classifier", {}).get("labels", [])
    category_names = {item["id"]: item["name"] for item in categories}
    label_values = [item.get("sourceLabel", item["id"]) for item in labels]
    if (not categories or len(category_names) != len(categories)
            or len({item["id"] for item in labels}) != len(labels)
            or len(set(label_values)) != len(labels)
            or any(not isinstance(value, str) or not value for value in label_values)
            or set(category_names) != {item["id"] for item in labels}
            or any(category_names.get(item["id"]) != item["name"] for item in labels)):
        raise ValueError("The installed classifier and dataset catalog do not match.")
    artifacts = [manifest["classifier"]] + [model for category in categories for model in category["models"]]
    for artifact in artifacts:
        keys = ["path"]
        if artifact.get("weightsPath"):
            keys.append("weightsPath")
        elif artifact is not manifest["classifier"]:
            raise ValueError("Installed material models must include their training weights.")
        for key in keys:
            target = resolve(root, artifact[key])
            if not target.exists() or not any(name == artifact[key] or name.startswith(artifact[key] + "/") for name in inventory):
                raise ValueError("The installed model is missing native exports or training weights.")
            if target.is_dir():
                for file in target.rglob("*"):
                    if file.is_symlink():
                        raise ValueError("Model exports cannot contain symbolic links.")
                    if file.is_file() and file.name != ".DS_Store" and str(file.relative_to(root)) not in inventory:
                        raise ValueError("The installed model contains an unregistered artifact file.")
    return root, manifest
