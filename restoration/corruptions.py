"""NumPy/Pillow corruptions shared by training, manifests and inference.

Rectangle masks are disjoint so their union has the requested area (within
pixel rounding). A random horizontal band belongs to each rectangle; random
positions and sizes within that band provide reproducible variation.
"""
from __future__ import annotations

import math
import numpy as np
from PIL import Image

CLASSES = ("clean", "salt", "blur", "occlusion")
LEVELS = ("low", "medium", "high")


def rectangles(rng, size: int, count: int, fraction: float):
    cuts = np.linspace(0, size, count + 1, dtype=int)
    target = round(size * size * fraction)
    areas = [target // count + int(i < target % count) for i in range(count)]
    result = []
    for i, area in enumerate(areas):
        band = int(cuts[i + 1] - cuts[i])
        min_w = math.ceil(area / band)
        shapes = [(width, max(1, min(band, round(area / width)))) for width in range(min_w, size + 1)]
        best_error = min(abs(width * height - area) for width, height in shapes)
        candidates = [(width, height) for width, height in shapes
                      if abs(width * height - area) <= best_error + 3]
        width, height = candidates[int(rng.integers(len(candidates)))]
        x = int(rng.integers(0, size - width + 1))
        y = int(rng.integers(cuts[i], cuts[i + 1] - height + 1))
        result.append([x, y, width, height])
    return result


def config(kind: str, seed: int, level: str | None = None, size: int = 128):
    if kind not in CLASSES:
        raise ValueError(f"Unknown corruption: {kind}")
    if level is not None and level not in LEVELS:
        raise ValueError(f"Unknown severity: {level}")
    rng = np.random.default_rng(seed)
    index = LEVELS.index(level) if level is not None else None
    cfg = {"kind": kind, "seed": int(seed), "level": level or "sampled"}
    if kind == "salt":
        cfg["probability"] = [0.03, 0.08, 0.15][index] if index is not None else float(rng.uniform(0.02, 0.15))
    elif kind == "blur":
        cfg["kernel"] = [3, 5, 7][index] if index is not None else int(rng.choice([3, 5, 7]))
        cfg["sigma"] = [0.7, 1.5, 2.5][index] if index is not None else float(rng.uniform(0.5, 2.5))
    elif kind == "occlusion":
        count = index + 1 if index is not None else int(rng.integers(1, 4))
        fraction = [0.10, 0.20, 0.35][index] if index is not None else float(rng.uniform(0.10, 0.35))
        cfg.update(rectangles=rectangles(rng, size, count, fraction), requested_area=fraction)
    return cfg


def apply(image: np.ndarray, cfg: dict) -> np.ndarray:
    """Image is HWC float32 RGB in [0, 1]. Blur uses the specified finite kernel."""
    out = np.asarray(image, dtype=np.float32).copy()
    kind = cfg["kind"]
    if kind == "salt":
        rng = np.random.default_rng(cfg["seed"])
        selected = rng.random(out.shape[:2]) < cfg["probability"]
        white = rng.integers(0, 2, size=out.shape[:2]).astype(np.float32)
        out[selected] = white[selected, None]
    elif kind == "blur":
        radius = int(cfg["kernel"]) // 2
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        weights = np.exp(-(x * x) / (2 * cfg["sigma"] ** 2))
        weights /= weights.sum()
        for axis in (0, 1):
            pads = [(0, 0)] * 3
            pads[axis] = (radius, radius)
            padded = np.pad(out, pads, mode="reflect")
            out = sum(weight * np.take(padded, np.arange(out.shape[axis]) + i, axis=axis)
                      for i, weight in enumerate(weights))
    elif kind == "occlusion":
        for x, y, width, height in cfg["rectangles"]:
            out[y:y + height, x:x + width] = 0
    elif kind != "clean":
        raise ValueError(kind)
    return np.clip(out, 0, 1).astype(np.float32)


def read_rgb(path, size=128):
    with Image.open(path) as image:
        from PIL import ImageOps
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image.resize((size, size), Image.Resampling.BICUBIC), dtype=np.float32) / 255
