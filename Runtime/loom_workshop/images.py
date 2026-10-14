"""Pixel decoding and paired native crops. Sources are never rewritten."""

from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


def image_dimensions(path):
    suffix = Path(path).suffix.lower()
    if suffix in (".heic", ".heif"):
        import pillow_heif
        return pillow_heif.open_heif(path).size
    if suffix == ".exr":
        value = stored_high_precision_pixels(path)[0]
        return value.shape[1], value.shape[0]
    with Image.open(path) as image:
        return image.size


def stored_high_precision_pixels(path):
    """Return native numeric pixels for formats Pillow would quantize or omit."""
    suffix = Path(path).suffix.lower()
    if suffix == ".png":
        with open(path, "rb") as stream:
            header = stream.read(26)
        if len(header) >= 26 and header[:8] == b"\x89PNG\r\n\x1a\n" and header[24] == 16:
            if header[25] == 0:  # Pillow retains native16-bit scalar PNG codes.
                with Image.open(path) as image:
                    value = np.array(image, dtype=np.uint16)
                return value[..., None], 16
            import png
            with open(path, "rb") as stream:
                width, height, rows, info = png.Reader(file=stream).asDirect()
                value = np.vstack([np.asarray(row, dtype=np.uint16) for row in rows])
            return value.reshape(height, width, info["planes"]), 16
    elif suffix in (".tif", ".tiff"):
        import tifffile
        with tifffile.TiffFile(path) as file:
            page = file.pages[0]
            value = page.asarray()
            if value.ndim == 3 and page.axes.startswith("S"):
                value = np.moveaxis(value, 0, -1)
            if value.ndim == 2:
                value = value[..., None]
            bits = page.bitspersample
            if isinstance(bits, tuple):
                if len(set(bits)) != 1:
                    raise ValueError("TIFF channels must use the same numeric precision.")
                bits = bits[0]
            return value, None if np.issubdtype(value.dtype, np.floating) else bits
    elif suffix in (".heic", ".heif"):
        import pillow_heif
        image = pillow_heif.open_heif(path, convert_hdr_to_8bit=False, hdr_to_16bit=True)
        value = np.asarray(image)[:, :image.size[0]].copy()
        if value.ndim == 2:
            value = value[..., None]
        return value, 16 if value.dtype == np.uint16 else 8
    elif suffix == ".exr":
        import OpenEXR
        with OpenEXR.File(str(path)) as image:
            channels = image.channels()
            if "RGB" in channels:
                value = channels["RGB"].pixels.copy()
            elif "RGBA" in channels:
                value = channels["RGBA"].pixels[..., :3].copy()
            elif "Y" in channels:
                value = channels["Y"].pixels[..., None].copy()
            elif len(channels) == 1:
                value = next(iter(channels.values())).pixels[..., None].copy()
            else:
                raise ValueError("Choose an EXR with RGB or a single scalar channel for this map.")
        if value.dtype == object:
            raise ValueError("Deep EXR images cannot be used as flat texture maps.")
        return value.astype(np.float32), None
    return None


def normalized_pixels(value, bits):
    value = value.astype(np.float32)
    if bits is not None:
        value = value / float((1 << int(bits)) - 1)
    if not np.isfinite(value).all():
        raise ValueError("The image contains non-finite numeric pixels.")
    return value


def read_rgb(path: str) -> np.ndarray:
    precise = stored_high_precision_pixels(path)
    if precise is not None:
        value = normalized_pixels(*precise)
        if value.shape[-1] < 3:
            value = np.repeat(value[..., :1], 3, axis=2)
        value = value[..., :3]
        if Path(path).suffix.lower() == ".exr":
            # EXR diffuse is linear light; the deployment input uses sRGB.
            value = np.clip(value, 0, 1)
            value = np.where(value <= 0.0031308, value * 12.92,
                             1.055 * np.power(value, 1 / 2.4) - 0.055)
        return np.clip(value, 0, 1)
    with Image.open(path) as image:
        # Training diffuse files use their stored pixel orientation. Applying EXIF
        # orientation to just one map would break correspondence with targets.
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def read_target(path: str, map_kind: str, normal_convention: str = "OpenGL") -> np.ndarray:
    precise = stored_high_precision_pixels(path)
    if precise is not None:
        value = normalized_pixels(*precise)
        if value.min() < 0 or value.max() > 1:
            raise ValueError(f"Target numeric pixels must be in [0,1]: {Path(path).name}")
        if map_kind == "normal":
            if value.shape[-1] < 3:
                raise ValueError("Normal targets need three color channels.")
            value = value[..., :3].copy()
            if normal_convention.lower() == "directx":
                value[..., 1] = 1.0 - value[..., 1]
            return value
        if value.shape[-1] >= 3 and not (np.array_equal(value[..., 0], value[..., 1]) and
                                        np.array_equal(value[..., 0], value[..., 2])):
            raise ValueError(f"Scalar target has different color channels: {Path(path).name}")
        return value[..., :1].copy()
    with Image.open(path) as image:
        if map_kind == "normal":
            value = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            if normal_convention.lower() == "directx":
                value[..., 1] = 1.0 - value[..., 1]
            return value
        if image.mode not in ("1", "L", "I", "F") and not image.mode.startswith("I;16"):
            # Scalar RGB assets are accepted only when their channels agree;
            # converting a colored map to luminance would silently change targets.
            color = np.asarray(image.convert("RGB"))
            if not (np.array_equal(color[..., 0], color[..., 1]) and
                    np.array_equal(color[..., 0], color[..., 2])):
                raise ValueError(f"Scalar target has different color channels: {Path(path).name}")
            return color[..., :1].astype(np.float32) / 255.0
        raw = np.asarray(image)
        if image.mode == "F":
            value = raw.astype(np.float32)
        elif image.mode == "1":
            value = raw.astype(np.float32)
        elif image.mode == "L":
            value = raw.astype(np.float32) / 255.0
        else:
            if raw.min() < 0 or raw.max() > 65535:
                raise ValueError(f"Scalar target must use 8-bit, 16-bit or [0,1] float pixels: {Path(path).name}")
            # Pillow exposes 16-bit PNG as mode I on some versions. Scale by
            # the file's code range, never by its observed min/max.
            value = raw.astype(np.float32) / 65535.0
        if not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
            raise ValueError(f"Target contains pixels outside [0,1]: {Path(path).name}")
        return value[..., np.newaxis]


def exaggerate_depth(value: torch.Tensor) -> torch.Tensor:
    """Anchored nonlinear contrast. Constant maps and original endpoints stay fixed."""
    low, high = value.amin(), value.amax()
    span = high - low
    if float(span) <= 1e-12:
        return value
    normalized = (value - low) / span
    strength = value.new_tensor(2.0)
    stretched = (torch.tanh((normalized * 2.0 - 1.0) * strength) / torch.tanh(strength) + 1.0) * 0.5
    return low + stretched * span


def native_pair(sample: dict, size: int, rng: random.Random, map_kind: str,
                flip: bool, rotate: bool, exaggerate: bool, cache=None, augmentation_rng=None):
    augmentation_rng = augmentation_rng or rng
    convention = sample.get("normal_convention", "OpenGL")
    if cache is None:
        rgb = read_rgb(sample["input"])
        target = read_target(sample["target"], map_kind, convention)
    else:
        rgb = cache.read(sample["input"], "rgb", read_rgb)
        target = cache.read(sample["target"], (map_kind, convention.lower()),
                            lambda path: read_target(path, map_kind, convention))
    if rgb.shape[:2] != target.shape[:2]:
        raise ValueError(f"Paired maps need the same native dimensions: {sample['id']}")
    height, width = rgb.shape[:2]
    if min(width, height) < size:
        raise ValueError(f"Source is smaller than the selected {size} × {size} crop: {sample['id']}")
    left, top = rng.randrange(width - size + 1), rng.randrange(height - size + 1)
    rgb = rgb[top:top + size, left:left + size].copy()
    target = target[top:top + size, left:left + size].copy()
    horizontal_flip = flip and augmentation_rng.random() < 0.5
    vertical_flip = flip and augmentation_rng.random() < 0.5
    if horizontal_flip:
        rgb, target = rgb[:, ::-1].copy(), target[:, ::-1].copy()
        if map_kind == "normal":
            target[..., 0] = 1.0 - target[..., 0]
    if vertical_flip:
        rgb, target = rgb[::-1].copy(), target[::-1].copy()
        if map_kind == "normal":
            target[..., 1] = 1.0 - target[..., 1]
    turns = augmentation_rng.randrange(4) if rotate else 0
    if turns:
        rgb, target = np.rot90(rgb, turns).copy(), np.rot90(target, turns).copy()
        if map_kind == "normal":
            # Image row coordinates point down while OpenGL normal Y points up.
            x, y = target[..., 0].copy(), target[..., 1].copy()
            for _ in range(turns):
                x, y = 1.0 - y, x
            target[..., 0], target[..., 1] = x, y
    rgb_tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy())
    target_tensor = torch.from_numpy(target.transpose(2, 0, 1).copy())
    if exaggerate:
        target_tensor = exaggerate_depth(target_tensor)
    return rgb_tensor, target_tensor, {"x": left, "y": top, "size": size, "quarter_turns": turns,
                                       "horizontal_flip": bool(horizontal_flip), "vertical_flip": bool(vertical_flip)}


def classifier_pixels(path: str, width: int, height: int) -> Image.Image:
    # Create ML takes a native image feature. Only this temporary thumbnail is
    # quantized; originals and paired map-training pixels remain untouched.
    value = torch.from_numpy(read_rgb(path).transpose(2, 0, 1).copy())[None]
    thumbnail = F.interpolate(value, (height, width), mode="bilinear", align_corners=False)[0]
    pixels = np.rint(thumbnail.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
    return Image.fromarray(pixels)


def classifier_tensor(path: str, width: int, height: int) -> np.ndarray:
    """An imported RGB tensor classifier's temporary native input grid."""
    value = torch.from_numpy(read_rgb(path).transpose(2, 0, 1).copy())[None]
    return F.interpolate(value, (height, width), mode="bilinear", align_corners=False).numpy()


def classifier_image(path: str, size: int) -> torch.Tensor:
    # The classifier sees a centered thumbnail, preserving source precision
    # until its float resize. Native map training never uses this resize.
    value = read_rgb(path)
    height, width = value.shape[:2]
    edge = min(height, width)
    top, left = (height - edge) // 2, (width - edge) // 2
    square = torch.from_numpy(value[top:top + edge, left:left + edge].transpose(2, 0, 1).copy())[None]
    return F.interpolate(square, (size, size), mode="bilinear", align_corners=False)[0]
