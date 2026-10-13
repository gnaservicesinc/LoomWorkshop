"""Optimization must preserve current files, precision and fresh random crops."""
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Runtime"))
from loom_workshop import images
from loom_workshop.pixel_cache import DecodedImageCache


class DecodedPixelsTests(unittest.TestCase):
    def test_repeated_native_pairs_decode_once_and_match_all_random_transforms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ramp = np.random.default_rng(3).integers(0, 256, (64, 64), dtype=np.uint8)
            Image.fromarray(np.repeat(ramp[..., None], 3, axis=2)).save(root / "rgb.png")
            Image.fromarray(ramp.astype(np.uint16) * 257).save(root / "depth.png")
            sample = {"id": "pair", "input": str(root / "rgb.png"), "target": str(root / "depth.png")}
            reference_rng, cached_rng = random.Random(7), random.Random(7)
            cache = DecodedImageCache(1024 * 1024)
            expected = [images.native_pair(sample, 32, reference_rng, "depth", True, True, True) for _ in range(6)]
            with patch.object(images, "read_rgb", wraps=images.read_rgb) as rgb_decoder, \
                 patch.object(images, "read_target", wraps=images.read_target) as target_decoder:
                actual = [images.native_pair(sample, 32, cached_rng, "depth", True, True, True, cache) for _ in range(6)]
            self.assertEqual(rgb_decoder.call_count, 1)
            self.assertEqual(target_decoder.call_count, 1)
            self.assertEqual(cache.hits, 10)
            for reference, cached in zip(expected, actual):
                torch.testing.assert_close(reference[0], cached[0], atol=0, rtol=0)
                torch.testing.assert_close(reference[1], cached[1], atol=0, rtol=0)
                self.assertEqual(reference[2], cached[2])
            self.assertGreater(len({(item[2]["x"], item[2]["y"]) for item in actual}), 1)

    def test_budget_evicts_and_never_retains_an_oversized_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / str(index) for index in range(3)]
            for path in paths:
                path.write_bytes(b"pixels")
            cache = DecodedImageCache(32)
            loader = lambda path: np.ones((2, 2), dtype=np.float32)
            for path in paths:
                cache.read(path, "rgb", loader)
            self.assertEqual(cache.cached_bytes, 32)
            cache.read(paths[0], "rgb", loader)
            self.assertEqual(cache.misses, 4)
            cache.read(paths[0], "large", lambda path: np.ones(64, dtype=np.float32))
            self.assertEqual(cache.cached_bytes, 32)
            disabled = DecodedImageCache(0)
            disabled.read(paths[0], "rgb", loader)
            self.assertEqual(disabled.cached_bytes, 0)

    def test_changed_removed_or_repointed_files_cannot_return_stale_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, link = root / "first", root / "second", root / "link"
            first.write_bytes(b"a")
            second.write_bytes(b"bb")
            link.symlink_to(first)
            cache = DecodedImageCache(1024)
            loader = lambda path: np.frombuffer(Path(path).read_bytes(), dtype=np.uint8).copy()
            self.assertEqual(cache.read(link, "rgb", loader).tolist(), [97])
            first.write_bytes(b"ccc")
            self.assertEqual(cache.read(link, "rgb", loader).tolist(), [99, 99, 99])
            link.unlink()
            link.symlink_to(second)
            self.assertEqual(cache.read(link, "rgb", loader).tolist(), [98, 98])
            link.unlink()
            with self.assertRaises(FileNotFoundError):
                cache.read(link, "rgb", loader)

    def test_image_change_during_decode_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pixels"
            path.write_bytes(b"a")
            def changing_loader(current):
                Path(current).write_bytes(b"changed")
                return np.zeros(4, dtype=np.float32)
            cache = DecodedImageCache(1024)
            with self.assertRaisesRegex(ValueError, "changed while"):
                cache.read(path, "rgb", changing_loader)
            self.assertEqual(cache.cached_bytes, 0)

    def test_normal_conventions_have_distinct_cache_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.fromarray(np.tile(np.array([128, 32, 255], dtype=np.uint8), (32, 32, 1))).save(root / "normal.png")
            sample = {"id": "normal", "input": str(root / "normal.png"), "target": str(root / "normal.png")}
            cache = DecodedImageCache(1024 * 1024)
            gl = images.native_pair(sample, 32, random.Random(1), "normal", False, False, False, cache)[1]
            dx = images.native_pair({**sample, "normal_convention": "DirectX"}, 32, random.Random(1), "normal", False, False, False, cache)[1]
            torch.testing.assert_close(dx[1], 1 - gl[1], atol=0, rtol=0)
            torch.testing.assert_close(dx[0], gl[0], atol=0, rtol=0)
            self.assertEqual(cache.misses, 3)
