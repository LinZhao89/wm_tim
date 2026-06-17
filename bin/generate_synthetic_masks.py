"""Convert synthetic wafer folders into paired image/mask training data.

Input layout:
    synthetic/<category>/*.png

Output layout:
    output/images/train/<category>/*.png
    output/masks/train/<category>/*_mask.png
    output/images/val/<category>/*.png
    output/masks/val/<category>/*_mask.png
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


CATEGORY_DEFAULTS = {
    "center": {"mode": "coherent", "threshold": 1.25, "min_size": 3},
    "donut": {"mode": "coherent", "threshold": 1.15, "min_size": 3},
    "edge-loc": {"mode": "edge_coherent", "threshold": 1.15, "min_size": 3, "min_radius": 0.55},
    "edge-ring": {"mode": "edge_coherent", "threshold": 1.05, "min_size": 3, "min_radius": 0.60},
    "loc": {"mode": "coherent", "threshold": 1.15, "min_size": 3},
    "near-full": {"mode": "all_white", "threshold": 0.0, "min_size": 1},
    "random": {"mode": "all_white", "threshold": 0.0, "min_size": 1},
    "scratch": {"mode": "all_white", "threshold": 0.0, "min_size": 1},
}


def _class_map(image: Image.Image):
    gray = np.asarray(image.convert("L"))
    classes = np.zeros_like(gray, dtype=np.float32)
    classes[(gray > 30) & (gray <= 200)] = 1.0
    classes[gray > 200] = 2.0
    white = gray > 200
    wafer = gray > 30
    return classes, white, wafer


def _wafer_radius_map(wafer: np.ndarray):
    ys, xs = np.where(wafer)
    if len(xs) == 0:
        return np.ones_like(wafer, dtype=np.float32)
    cx = (xs.min() + xs.max()) / 2.0
    cy = (ys.min() + ys.max()) / 2.0
    radius = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2.0
    radius = max(radius, 1.0)
    grid_y, grid_x = np.indices(wafer.shape)
    return np.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2) / radius


def _uniform_filter(values: np.ndarray, size: int):
    pad = size // 2
    padded = np.pad(values, pad, mode="constant")
    output = np.zeros_like(values, dtype=np.float32)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            window = padded[row:row + size, col:col + size]
            output[row, col] = float(window.mean())
    return output


def _connected_components(mask: np.ndarray):
    labels = np.zeros(mask.shape, dtype=np.int32)
    components = []
    current = 0
    height, width = mask.shape
    for start_y, start_x in zip(*np.where(mask & (labels == 0))):
        current += 1
        stack = [(int(start_y), int(start_x))]
        labels[start_y, start_x] = current
        pixels = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if ny == y and nx == x:
                        continue
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = current
                        stack.append((ny, nx))
        components.append(pixels)
    return labels, components


def _remove_small_components(mask: np.ndarray, min_size: int):
    if min_size <= 1:
        return mask
    labels, components = _connected_components(mask)
    if not components:
        return mask
    keep = np.zeros(len(components) + 1, dtype=bool)
    for idx, pixels in enumerate(components, start=1):
        keep[idx] = len(pixels) >= min_size
    return keep[labels]


def _scratch_filter(mask: np.ndarray):
    labels, components = _connected_components(mask)
    if not components:
        return mask
    keep = np.zeros(len(components) + 1, dtype=bool)
    for idx, pixels in enumerate(components, start=1):
        ys = [pixel[0] for pixel in pixels]
        xs = [pixel[1] for pixel in pixels]
        height = max(ys) - min(ys) + 1
        width = max(xs) - min(xs) + 1
        area = len(pixels)
        elongation = max(height, width) / max(1, min(height, width))
        keep[idx] = area >= 2 and (elongation >= 2.0 or max(height, width) >= 4)
    return keep[labels]


def build_mask(image: Image.Image, category: str, window_size: int = 3):
    classes, white, wafer = _class_map(image)
    settings = CATEGORY_DEFAULTS.get(category.lower(), CATEGORY_DEFAULTS["loc"])
    mode = settings["mode"]
    if mode == "all_white":
        mask = white & wafer
    else:
        mean_map = _uniform_filter(classes, size=window_size)
        mask = white & wafer & (mean_map >= settings["threshold"])
        if mode == "edge_coherent":
            radius_map = _wafer_radius_map(wafer)
            mask &= radius_map >= settings.get("min_radius", 0.55)
        elif mode == "scratch":
            mask = _scratch_filter(mask)
    mask = _remove_small_components(mask, int(settings["min_size"]))
    return mask.astype(np.uint8) * 255


def _split_files(files, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    files = np.array(sorted(files), dtype=object)
    order = rng.permutation(len(files))
    val_count = int(round(len(files) * val_ratio))
    val_indices = set(order[:val_count].tolist())
    train, val = [], []
    for index, path in enumerate(files.tolist()):
        (val if index in val_indices else train).append(path)
    return train, val


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("synthetic_root")
    parser.add_argument("output_root")
    parser.add_argument("--val_ratio", default=0.2, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--window_size", default=3, type=int)
    parser.add_argument("--limit_per_category", default=0, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    synthetic_root = args.synthetic_root
    output_root = args.output_root
    val_ratio = args.val_ratio
    seed = args.seed
    window_size = args.window_size
    limit_per_category = args.limit_per_category
    overwrite = args.overwrite
    synthetic_root = Path(synthetic_root)
    output_root = Path(output_root)
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for category_dir in sorted(path for path in synthetic_root.iterdir() if path.is_dir()):
        files = [
            path
            for path in category_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if limit_per_category > 0:
            files = files[:limit_per_category]
        train_files, val_files = _split_files(files, val_ratio, seed)
        summary[category_dir.name] = {"train": len(train_files), "val": len(val_files), "empty_masks": 0}
        for split, split_files in (("train", train_files), ("val", val_files)):
            image_dir = output_root / "images" / split / category_dir.name
            mask_dir = output_root / "masks" / split / category_dir.name
            image_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            for image_path in split_files:
                image = Image.open(image_path).convert("RGB")
                mask = build_mask(image, category_dir.name, window_size=window_size)
                if not np.any(mask):
                    summary[category_dir.name]["empty_masks"] += 1
                image.save(image_dir / image_path.name)
                Image.fromarray(mask, mode="L").save(mask_dir / f"{image_path.stem}_mask.png")
    with open(output_root / "mask_generation_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
