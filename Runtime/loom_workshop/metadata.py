"""Image-owned membership. No folder inventory is persisted in project state."""
import errno
import ctypes
import hashlib
import json
import os
from pathlib import Path

WORKSHOP_KEY = "org.gnaservicesinc.loom-workshop.image.v1"
FACTORY_KEY = "com.gnaservicesinc.loomfactory.image.v1"


def raw_tag(path, key):
    if hasattr(os, "getxattr"):
        return os.getxattr(path, key)
    library = ctypes.CDLL(None, use_errno=True)
    function = library.getxattr
    function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_int]
    function.restype = ctypes.c_ssize_t
    arguments = os.fsencode(path), key.encode()
    length = function(*arguments, None, 0, 0, 0)
    if length < 0:
        raise OSError(ctypes.get_errno(), "Image metadata could not be read")
    if length > 65536:
        raise ValueError("Image metadata is too large.")
    buffer = ctypes.create_string_buffer(length)
    count = function(*arguments, buffer, length, 0, 0)
    if count < 0:
        raise OSError(ctypes.get_errno(), "Image metadata could not be read")
    return buffer.raw[:count]


def set_raw_tag(path, key, value):
    if hasattr(os, "setxattr"):
        os.setxattr(path, key, value)
        return
    library = ctypes.CDLL(None, use_errno=True)
    function = library.setxattr
    function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_int]
    function.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(value)
    if function(os.fsencode(path), key.encode(), buffer, len(value), 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Image metadata could not be written")


def read_tag(path, key):
    try:
        value = raw_tag(path, key)
    except OSError as error:
        if error.errno in (errno.ENODATA, getattr(errno, "ENOATTR", 93)):
            return None
        raise ValueError(f"Image metadata could not be read: {Path(path).name}") from error
    try:
        tag = json.loads(value)
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"Image metadata needs repair: {Path(path).name}") from error
    if not isinstance(tag, dict):
        raise ValueError(f"Image metadata needs repair: {Path(path).name}")
    return tag


def write_tag(path, tag):
    try:
        set_raw_tag(path, WORKSHOP_KEY, json.dumps(tag, allow_nan=False).encode())
    except OSError as error:
        raise ValueError(f"Image metadata could not be saved: {Path(path).name}") from error


def factory_category_id(tag):
    name = (tag.get("category") or "").strip().lower()
    return "dataset-" + hashlib.sha256(name.encode()).hexdigest()[:16] if name else None


def byte_identity(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime


def current_category(path, manifest):
    """Current Factory tags are authoritative; classifiers fill only absent tags."""
    catalog = {category["id"] for category in manifest["categories"]}
    factory = read_tag(path, FACTORY_KEY)
    if factory and factory.get("ignored", False):
        raise ValueError("This image was ignored in Loom Factory. Refresh the source folder.")
    factory_category = factory_category_id(factory) if factory else None
    factory_name = ((factory or {}).get("category") or "").strip() or None
    if factory_category:
        if factory_category not in catalog:
            raise ValueError("This dataset has no model in this Texture Studio build.")
        return factory_category
    tag = read_tag(path, WORKSHOP_KEY)
    if tag:
        size, modified = byte_identity(path)
        if (tag.get("baseID") == manifest["id"] and tag.get("baseVersion") == manifest["version"]
                and tag.get("classifiedBytes") == size and tag.get("classifiedModified") == modified
                and tag.get("categoryID") in catalog and tag.get("factoryCategory") == factory_name):
            return tag["categoryID"]
    return None


def prepare_membership(samples):
    """Initialize new pairs only; an existing role or pair never silently changes."""
    used = {}
    for sample in samples:
        input_path = sample["input"]
        input_tag = read_tag(input_path, WORKSHOP_KEY)
        factory_input = read_tag(input_path, FACTORY_KEY)
        if input_tag and input_tag.get("id") != sample["id"]:
            raise ValueError("An image's identity metadata changed. Refresh the source folder before training.")
        pair_id = (factory_input or {}).get("pairingKey") or (input_tag or {}).get("pairID") or "pair-" + sample["id"]
        fields = [(input_path, "diffuse")] + [(path, role) for role, path in sample.get("targets", {}).items()]
        if "normal" in sample.get("targets", {}) and sample.get("normal_convention", "opengl").lower() == "directx":
            raise ValueError("DirectX normal maps are excluded from training.")
        for path, role in fields:
            if role not in ("diffuse", "depth", "roughness", "normal"):
                raise ValueError("Choose a supported diffuse, depth, roughness or normal map role.")
            physical_path = str(Path(path).resolve())
            if physical_path in used and used[physical_path] != (pair_id, role):
                raise ValueError("One physical image cannot belong to different training pairs or map roles.")
            used[physical_path] = (pair_id, role)
            factory = read_tag(path, FACTORY_KEY)
            tag = read_tag(path, WORKSHOP_KEY)
            active = factory or tag
            if active:
                if active.get("ignored", False):
                    raise ValueError(f"This image was ignored in Loom Factory: {Path(path).name}")
                actual_role = active.get("role")
                if role == "normal" and (active.get("normalConvention") or "opengl").lower() == "directx":
                    raise ValueError("DirectX normal maps are excluded from training.")
                actual_pair = active.get("pairingKey") if factory else active.get("pairID")
                if actual_role != role or actual_pair != pair_id:
                    raise ValueError(f"A map's current image metadata no longer matches its pair: {Path(path).name}")
                if role == "normal" and active.get("normalConvention") and active["normalConvention"].lower() != sample.get("normal_convention", "opengl").lower():
                    raise ValueError("A normal map's convention changed. Refresh the source folder before training.")
            if tag is None:
                tag = {"id": sample["id"] if role == "diffuse" else sample["id"] + "-" + role,
                       "pairID": pair_id, "role": role}
                write_tag(path, tag)
    return used


def stamp_category(path, category_id, manifest):
    tag = read_tag(path, WORKSHOP_KEY)
    if tag is None:
        raise ValueError("The source image lost its metadata during classification.")
    size, modified = byte_identity(path)
    factory_name = ((read_tag(path, FACTORY_KEY) or {}).get("category") or "").strip() or None
    tag.update(categoryID=category_id, baseID=manifest["id"], baseVersion=manifest["version"],
               classifiedBytes=size, classifiedModified=modified, factoryCategory=factory_name)
    write_tag(path, tag)


def membership_signature(path):
    """Only data-selection fields; changing a review note does not alter membership."""
    result = []
    for key in (FACTORY_KEY, WORKSHOP_KEY):
        tag = read_tag(path, key)
        if tag is None:
            result.append(None)
        elif key == FACTORY_KEY:
            result.append({name: tag.get(name) for name in ("id", "pairingKey", "role", "category", "verified", "normalConvention", "ignored")})
        else:
            result.append({name: tag.get(name) for name in ("id", "pairID", "role", "categoryID", "baseID", "baseVersion", "classifiedBytes", "classifiedModified", "factoryCategory", "normalConvention")})
    return result
