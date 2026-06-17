"""Generate formula-based synthetic wafer maps with exact masks.

Output layout:
    output/images/train/<category>/*.png
    output/masks/train/<category>/*_mask.png
    output/images/val/<category>/*.png
    output/masks/val/<category>/*_mask.png
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


BACKGROUND_COLOR = np.array((68, 1, 84), dtype=np.uint8)
NORMAL_COLOR = np.array((32, 144, 140), dtype=np.uint8)
DEFECT_COLOR = np.array((253, 231, 36), dtype=np.uint8)

CATEGORIES = (
    "Center",
    "Loc",
    "Edge-Loc",
    "Donut",
    "Edge-Ring",
    "Near-full",
    "Random",
    "Scratch",
)


def wafer_mask(size: int, rng: np.random.Generator, jitter: float = 0.02):
    center = (size - 1) / 2
    cx = center + rng.normal(0, jitter * size)
    cy = center + rng.normal(0, jitter * size)
    rx = size * rng.uniform(0.435, 0.465)
    ry = size * rng.uniform(0.435, 0.465)
    yy, xx = np.indices((size, size))
    mask = ((xx - cx) ** 2 / (rx ** 2)) + ((yy - cy) ** 2 / (ry ** 2)) <= 1
    return mask, cx, cy, rx, ry


def sample_by_probability(probability: np.ndarray, wafer: np.ndarray, rng: np.random.Generator):
    probability = np.clip(probability, 0, 1)
    return wafer & (rng.random(probability.shape) < probability)


def add_wafer_speckle(defect: np.ndarray, wafer: np.ndarray, rng: np.random.Generator, probability_range):
    probability = rng.uniform(*probability_range)
    return defect | (wafer & (rng.random(wafer.shape) < probability))


def angular_window(xx, yy, cx, cy, rng, coverage: float):
    """Return a circular angular mask covering a random fraction of the wafer."""
    if coverage >= 0.995:
        return np.ones_like(xx, dtype=bool)
    theta = np.mod(np.arctan2(yy - cy, xx - cx), 2 * math.pi)
    start = rng.uniform(0, 2 * math.pi)
    width = coverage * 2 * math.pi
    return np.mod(theta - start, 2 * math.pi) <= width


def gaussian_blob(wafer, cx, cy, rx, ry, rng, edge=False, centered=False):
    yy, xx = np.indices(wafer.shape)
    wafer_radius = min(rx, ry)
    if centered:
        px, py = cx + rng.normal(0, 1.0), cy + rng.normal(0, 1.0)
    elif edge:
        angle = rng.uniform(0, 2 * math.pi)
        radius = rng.uniform(0.66, 0.91) * wafer_radius
        px = cx + radius * math.cos(angle)
        py = cy + radius * math.sin(angle)
    else:
        angle = rng.uniform(0, 2 * math.pi)
        radius = rng.uniform(0.18, 0.66) * wafer_radius
        px = cx + radius * math.cos(angle)
        py = cy + radius * math.sin(angle)
    if centered:
        sigma = rng.uniform(0.13, 0.31) * wafer_radius
    elif edge:
        sigma = rng.choice([
            rng.uniform(0.055, 0.095) * wafer_radius,
            rng.uniform(0.10, 0.18) * wafer_radius,
            rng.uniform(0.19, 0.28) * wafer_radius,
        ])
    else:
        sigma = rng.choice([
            rng.uniform(0.06, 0.11) * wafer_radius,
            rng.uniform(0.12, 0.22) * wafer_radius,
            rng.uniform(0.23, 0.34) * wafer_radius,
        ])
    x_scale = sigma * rng.uniform(0.75, 1.45)
    y_scale = sigma * rng.uniform(0.75, 1.45)
    rotation = rng.uniform(0, math.pi)
    dx = xx - px
    dy = yy - py
    xr = dx * math.cos(rotation) + dy * math.sin(rotation)
    yr = -dx * math.sin(rotation) + dy * math.cos(rotation)
    distance = (xr / max(x_scale, 1e-6)) ** 2 + (yr / max(y_scale, 1e-6)) ** 2
    soft_blob = np.exp(-0.5 * distance)
    core_threshold = rng.uniform(0.18, 0.46)
    probability = np.where(soft_blob >= core_threshold, soft_blob * rng.uniform(0.82, 0.98), 0)
    if rng.random() < 0.35:
        halo = (soft_blob >= core_threshold * rng.uniform(0.45, 0.75)) & (soft_blob < core_threshold)
        probability = np.where(halo, rng.uniform(0.10, 0.25), probability)
    return sample_by_probability(probability, wafer, rng)


def blob_probability(xx, yy, px, py, sigma, rng, elongated=False):
    x_scale = sigma * rng.uniform(0.65, 1.55)
    y_scale = sigma * rng.uniform(0.65, 1.55)
    if elongated:
        if rng.random() < 0.5:
            x_scale *= rng.uniform(1.6, 2.8)
        else:
            y_scale *= rng.uniform(1.6, 2.8)
    rotation = rng.uniform(0, math.pi)
    dx = xx - px
    dy = yy - py
    xr = dx * math.cos(rotation) + dy * math.sin(rotation)
    yr = -dx * math.sin(rotation) + dy * math.cos(rotation)
    distance = (xr / max(x_scale, 1e-6)) ** 2 + (yr / max(y_scale, 1e-6)) ** 2
    return np.exp(-0.5 * distance)


def loc_pattern(wafer, cx, cy, rx, ry, rng, edge=False):
    yy, xx = np.indices(wafer.shape)
    wafer_radius = min(rx, ry)
    probability = np.zeros_like(wafer, dtype=float)
    mode = rng.choice(
        ["compact", "large_cluster", "multi_spot", "diffuse_speckle", "sector_cap"] if edge else
        ["compact", "large_cluster", "multi_spot", "diffuse_speckle"],
        p=[0.18, 0.21, 0.20, 0.18, 0.23] if edge else [0.28, 0.24, 0.25, 0.23],
    )
    if mode == "sector_cap":
        angle = rng.uniform(0, 2 * math.pi)
        radial = rng.uniform(0.72, 0.98) * wafer_radius
        px = cx + radial * math.cos(angle)
        py = cy + radial * math.sin(angle)
        score = ((xx - cx) ** 2 / (rx ** 2)) + ((yy - cy) ** 2 / (ry ** 2))
        cap = wafer & (score >= rng.uniform(0.54, 0.78)) & angular_window(xx, yy, cx, cy, rng, rng.uniform(0.10, 0.32))
        probability = np.where(cap, rng.uniform(0.56, 0.94), 0)
        focus = blob_probability(xx, yy, px, py, rng.uniform(0.11, 0.20) * wafer_radius, rng, elongated=True)
        probability = np.maximum(probability, np.where(focus > rng.uniform(0.18, 0.36), focus * rng.uniform(0.55, 0.95), 0))
    else:
        n_blobs = 1 if mode in {"compact", "large_cluster"} else int(rng.integers(2, 5))
        for _ in range(n_blobs):
            angle = rng.uniform(0, 2 * math.pi)
            if edge:
                radial = rng.uniform(0.66, 0.95) * wafer_radius
            else:
                radial = rng.choice([
                    rng.uniform(0.05, 0.35) * wafer_radius,
                    rng.uniform(0.35, 0.72) * wafer_radius,
                ], p=[0.45, 0.55])
            px = cx + radial * math.cos(angle) + rng.normal(0, 0.025 * wafer_radius)
            py = cy + radial * math.sin(angle) + rng.normal(0, 0.025 * wafer_radius)
            if mode == "compact":
                sigma = rng.uniform(0.055, 0.115) * wafer_radius
            elif mode == "large_cluster":
                sigma = rng.uniform(0.13, 0.26) * wafer_radius
            elif mode == "diffuse_speckle":
                sigma = rng.uniform(0.15, 0.30) * wafer_radius
            else:
                sigma = rng.uniform(0.045, 0.12) * wafer_radius
            blob = blob_probability(xx, yy, px, py, sigma, rng, elongated=(rng.random() < 0.28))
            threshold = rng.uniform(0.14, 0.42) if mode != "diffuse_speckle" else rng.uniform(0.05, 0.22)
            probability = np.maximum(probability, np.where(blob >= threshold, blob * rng.uniform(0.60, 0.98), 0))
    if mode in {"diffuse_speckle", "large_cluster"} or rng.random() < 0.35:
        local_support = probability > 0
        if local_support.any():
            probability = np.where(local_support & (rng.random(wafer.shape) < rng.uniform(0.04, 0.13)), rng.uniform(0.25, 0.55), probability)
    return sample_by_probability(probability, wafer, rng)


def ring_pattern(wafer, cx, cy, rx, ry, rng, donut=False):
    yy, xx = np.indices(wafer.shape)
    dcx = cx + rng.normal(0, 0.045 * rx)
    dcy = cy + rng.normal(0, 0.045 * ry)
    erx = rx * rng.uniform(0.86, 1.08)
    ery = ry * rng.uniform(0.86, 1.08)
    score = ((xx - dcx) ** 2 / (erx ** 2)) + ((yy - dcy) ** 2 / (ery ** 2))
    theta = np.mod(np.arctan2(yy - dcy, xx - dcx), 2 * math.pi)
    angular_texture = (
        1.0
        + rng.uniform(-0.35, 0.35) * np.sin(theta * rng.integers(1, 4) + rng.uniform(0, 2 * math.pi))
        + rng.uniform(-0.20, 0.20) * np.cos(theta * rng.integers(2, 6) + rng.uniform(0, 2 * math.pi))
    )
    angular_texture = np.clip(angular_texture, 0.35, 1.65)

    if donut:
        mode = rng.choice(["speckled", "cloud", "thick", "solid", "partial"], p=[0.34, 0.28, 0.20, 0.13, 0.05])
        if mode == "speckled":
            lower = rng.uniform(0.20, 0.50)
            upper = rng.uniform(0.68, 1.02)
            base_prob = rng.uniform(0.22, 0.52)
        elif mode == "cloud":
            lower = rng.uniform(0.08, 0.34)
            upper = rng.uniform(0.72, 1.04)
            base_prob = rng.uniform(0.18, 0.43)
        elif mode == "thick":
            lower = rng.uniform(0.26, 0.48)
            upper = rng.uniform(0.76, 1.03)
            base_prob = rng.uniform(0.62, 0.88)
        elif mode == "solid":
            lower = rng.uniform(0.16, 0.32)
            upper = rng.uniform(0.72, 0.98)
            base_prob = rng.uniform(0.86, 0.99)
        else:
            lower = rng.uniform(0.28, 0.54)
            upper = rng.uniform(0.68, 1.01)
            base_prob = rng.uniform(0.58, 0.88)
    else:
        mode = rng.choice(["edge_full", "edge_sparse", "edge_partial"], p=[0.46, 0.30, 0.24])
        upper = rng.uniform(0.87, 1.03)
        lower = upper - rng.uniform(0.035, 0.17)
        base_prob = rng.uniform(0.50, 0.94) if mode != "edge_sparse" else rng.uniform(0.20, 0.48)

    upper = min(upper, 1.06)
    band = wafer & (score >= lower) & (score <= upper)
    if donut and mode == "partial":
        band &= angular_window(xx, yy, dcx, dcy, rng, rng.uniform(0.35, 0.72))
    elif not donut and mode == "edge_partial":
        band &= angular_window(xx, yy, dcx, dcy, rng, rng.uniform(0.25, 0.70))
    elif rng.random() < 0.18:
        gap = angular_window(xx, yy, dcx, dcy, rng, rng.uniform(0.05, 0.14))
        band &= ~gap

    radial_center = (lower + upper) / 2
    radial_width = max((upper - lower) / 2, 1e-4)
    radial_softness = np.exp(-0.5 * ((score - radial_center) / radial_width) ** 2)
    probability = np.where(band, base_prob * angular_texture * (0.65 + 0.45 * radial_softness), 0)
    if donut and mode in {"speckled", "cloud"}:
        pepper = wafer & (score >= max(0.05, lower - 0.18)) & (score <= min(1.05, upper + 0.12))
        probability = np.where(pepper & (probability == 0), rng.uniform(0.025, 0.09), probability)
        lobe_angle = rng.uniform(0, 2 * math.pi)
        lobe = np.cos(theta - lobe_angle)
        lobe_mask = pepper & (lobe > rng.uniform(0.15, 0.55))
        probability = np.where(lobe_mask, np.maximum(probability, rng.uniform(0.12, 0.35)), probability)
    return sample_by_probability(probability, wafer, rng)


def near_full_pattern(wafer, rng):
    return wafer & (rng.random(wafer.shape) < rng.uniform(0.72, 0.92))


def random_pattern(wafer, rng):
    return wafer & (rng.random(wafer.shape) < rng.uniform(0.18, 0.35))


def scratch_pattern(wafer, cx, cy, rx, ry, rng):
    yy, xx = np.indices(wafer.shape)
    wafer_radius = min(rx, ry)
    choice = rng.choice(
        ["single_line", "parallel", "broad_band", "crescent", "arc", "broken", "edge_sweep"],
        p=[0.20, 0.16, 0.18, 0.16, 0.12, 0.12, 0.06],
    )
    width = int(rng.integers(1, max(3, int(0.07 * wafer.shape[0]))))
    length_gate = np.ones_like(wafer, dtype=bool)
    if choice in {"single_line", "parallel", "broad_band", "broken"}:
        angle = rng.choice([
            rng.uniform(-0.25, 0.25),
            rng.uniform(math.pi / 2 - 0.25, math.pi / 2 + 0.25),
            rng.uniform(-0.95, 0.95),
        ], p=[0.22, 0.22, 0.56])
        distance = (xx - cx) * math.sin(angle) - (yy - cy) * math.cos(angle)
        offset = rng.uniform(-0.55, 0.55) * wafer_radius
        if choice == "parallel":
            spacing = rng.uniform(0.10, 0.22) * wafer_radius
            candidate = (np.abs(distance - offset) <= width) | (np.abs(distance - offset - spacing) <= max(1, width - 1))
        elif choice == "broad_band":
            candidate = np.abs(distance - offset) <= rng.uniform(2.0, 4.2) * width
        else:
            candidate = np.abs(distance - offset) <= width
        along = (xx - cx) * math.cos(angle) + (yy - cy) * math.sin(angle)
        if choice == "broken":
            phase = rng.uniform(0, 2 * math.pi)
            dash = np.sin(along / rng.uniform(3.0, 7.5) + phase) > rng.uniform(-0.45, 0.25)
            length_gate &= dash
        elif rng.random() < 0.40:
            center_along = rng.uniform(-0.30, 0.30) * wafer.shape[0]
            length_gate &= np.abs(along - center_along) <= rng.uniform(0.35, 0.85) * wafer.shape[0]
    elif choice == "crescent":
        angle = rng.uniform(0, 2 * math.pi)
        rc = rng.uniform(0.20, 0.62) * wafer_radius
        xr = cx + rc * math.cos(angle)
        yr = cy + rc * math.sin(angle)
        sigma = rng.uniform(0.34, 0.88) * wafer_radius
        radius = np.sqrt((xx - xr) ** 2 + (yy - yr) ** 2)
        candidate = np.abs(radius - sigma) <= rng.uniform(1.0, 2.4) * width
        length_gate = angular_window(xx, yy, xr, yr, rng, rng.uniform(0.18, 0.52))
    elif choice == "arc":
        sigma = rng.uniform(0.45, 0.92) * wafer_radius
        radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        candidate = np.abs(radius - sigma) <= rng.uniform(1.0, 2.4) * width
        length_gate = angular_window(xx, yy, cx, cy, rng, rng.uniform(0.16, 0.45))
    else:
        score = ((xx - cx) ** 2 / (rx ** 2)) + ((yy - cy) ** 2 / (ry ** 2))
        candidate = wafer & (score >= rng.uniform(0.62, 0.90))
        length_gate = angular_window(xx, yy, cx, cy, rng, rng.uniform(0.10, 0.28))
    line_prob = rng.uniform(0.42, 0.82) if choice == "broad_band" else rng.uniform(0.64, 0.96)
    probability = np.where(candidate & wafer & length_gate, line_prob, 0)
    if rng.random() < 0.75:
        support = candidate & wafer
        probability = np.where(support & (rng.random(wafer.shape) < rng.uniform(0.015, 0.065)), rng.uniform(0.18, 0.45), probability)
    return sample_by_probability(probability, wafer, rng)


def create_sample(category: str, size: int, rng: np.random.Generator):
    wafer, cx, cy, rx, ry = wafer_mask(size, rng)
    if category == "good":
        defect = np.zeros_like(wafer, dtype=bool)
    elif category == "Center":
        defect = gaussian_blob(wafer, cx, cy, rx, ry, rng, centered=True)
    elif category == "Loc":
        defect = loc_pattern(wafer, cx, cy, rx, ry, rng)
        defect = add_wafer_speckle(defect, wafer, rng, (0.008, 0.045))
    elif category == "Edge-Loc":
        defect = loc_pattern(wafer, cx, cy, rx, ry, rng, edge=True)
        defect = add_wafer_speckle(defect, wafer, rng, (0.012, 0.060))
    elif category == "Donut":
        defect = ring_pattern(wafer, cx, cy, rx, ry, rng, donut=True)
    elif category == "Edge-Ring":
        defect = ring_pattern(wafer, cx, cy, rx, ry, rng, donut=False)
    elif category == "Near-full":
        defect = near_full_pattern(wafer, rng)
    elif category == "Random":
        defect = random_pattern(wafer, rng)
    elif category == "Scratch":
        defect = scratch_pattern(wafer, cx, cy, rx, ry, rng)
        defect = add_wafer_speckle(defect, wafer, rng, (0.012, 0.055))
    else:
        raise ValueError(f"Unknown category: {category}")
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :] = BACKGROUND_COLOR
    image[wafer] = NORMAL_COLOR
    image[defect] = DEFECT_COLOR
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[defect] = 255
    return image, mask, int(wafer.sum()), int(defect.sum())


def save_split(output_root, split, category, index, image, mask):
    image_dir = output_root / "images" / split / category
    mask_dir = output_root / "masks" / split / category
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    name = f"{category.lower().replace('-', '_')}_{index:05d}.png"
    Image.fromarray(image, mode="RGB").save(image_dir / name)
    Image.fromarray(mask, mode="L").save(mask_dir / f"{Path(name).stem}_mask.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root")
    parser.add_argument("--size", default=52, type=int)
    parser.add_argument("--size_min", default=None, type=int)
    parser.add_argument("--size_max", default=None, type=int)
    parser.add_argument("--count_per_category", default=1000, type=int)
    parser.add_argument("--good_count", default=1000, type=int)
    parser.add_argument("--val_ratio", default=0.2, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    if (args.size_min is None) != (args.size_max is None):
        raise ValueError("--size_min and --size_max must be provided together.")
    if args.size_min is not None and args.size_min > args.size_max:
        raise ValueError("--size_min must be <= --size_max.")
    summary = {}
    all_categories = list(CATEGORIES) + ["good"]
    for category in all_categories:
        count = args.good_count if category == "good" else args.count_per_category
        val_count = int(round(count * args.val_ratio))
        summary[category] = {"train": count - val_count, "val": val_count, "empty_masks": 0, "mean_defect_pixels": 0}
        defect_pixels = []
        for index in range(count):
            split = "val" if index < val_count else "train"
            sample_size = int(rng.integers(args.size_min, args.size_max + 1)) if args.size_min is not None else args.size
            image, mask, _, defect_count = create_sample(category, sample_size, rng)
            if category != "good" and defect_count == 0:
                image, mask, _, defect_count = create_sample(category, sample_size, rng)
            if defect_count == 0:
                summary[category]["empty_masks"] += 1
            defect_pixels.append(defect_count)
            save_split(output_root, split, category, index, image, mask)
        summary[category]["mean_defect_pixels"] = float(np.mean(defect_pixels))
    with open(output_root / "formula_generation_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
