"""Bounded, run-local decoded pixels; current files remain authoritative."""

from collections import OrderedDict
from pathlib import Path

import numpy as np


def file_identity(path):
    path = Path(path)
    link, source = path.lstat(), path.stat()
    return tuple((value.st_dev, value.st_ino, value.st_size,
                  value.st_mtime_ns, value.st_ctime_ns) for value in (link, source))


class DecodedImageCache:
    """Keep bounded validated pixels, never dataset membership.

    Every hit checks the current link and physical file. Callers still check
    image metadata before selecting a pair. Crops copy these read-only arrays,
    so random transforms cannot modify the original or cached pixels.
    """

    def __init__(self, maximum_bytes):
        self.maximum_bytes = maximum_bytes
        self.cached_bytes = 0
        self.hits = self.misses = 0
        self._entries = OrderedDict()

    def read(self, path, variant, loader):
        key = (str(path), variant)
        identity = file_identity(path)
        previous = self._entries.pop(key, None)
        if previous is not None:
            previous_identity, pixels = previous
            if previous_identity == identity:
                self._entries[key] = previous
                self.hits += 1
                return pixels
            self.cached_bytes -= pixels.nbytes
        self.misses += 1
        pixels = np.ascontiguousarray(loader(path))
        if file_identity(path) != identity:
            raise ValueError(f"An image changed while its pixels were read: {Path(path).name}")
        if pixels.nbytes <= self.maximum_bytes:
            while self._entries and self.cached_bytes + pixels.nbytes > self.maximum_bytes:
                _, (_, discarded) = self._entries.popitem(last=False)
                self.cached_bytes -= discarded.nbytes
            pixels.setflags(write=False)
            self._entries[key] = (identity, pixels)
            self.cached_bytes += pixels.nbytes
        return pixels

    def statistics(self):
        return {"maximum_bytes": self.maximum_bytes, "cached_bytes": self.cached_bytes,
                "hits": self.hits, "misses": self.misses}
