"""Wafer foreground estimation and polar patch metadata."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GeometryConfig:
    radial_bins: int = 4
    angular_bins: int = 8
    min_wafer_coverage: float = 0.5
    background_threshold: float = 0.2
    radial_neighbors: int = 1
    angular_neighbors: int = 1

    @property
    def num_bins(self) -> int:
        return 1 + max(0, self.radial_bins - 1) * self.angular_bins


def estimate_wafer_geometry(images: torch.Tensor, threshold: float = 0.2):
    """Return foreground mask, center and radius for each raw image tensor."""
    masks, centers, radii = [], [], []
    for image in images:
        _, height, width = image.shape
        corners = torch.stack(
            [image[:, 0, 0], image[:, 0, -1], image[:, -1, 0], image[:, -1, -1]],
            dim=1,
        )
        background = corners.median(dim=1).values[:, None, None]
        valid = torch.linalg.vector_norm(image - background, dim=0) > threshold
        ys, xs = torch.where(valid)
        if len(xs) == 0:
            masks.append(torch.zeros((height, width), device=image.device, dtype=torch.bool))
            centers.append(torch.tensor([(width - 1) / 2, (height - 1) / 2], device=image.device))
            radii.append(torch.tensor(1.0, device=image.device))
            continue
        center_x = (xs.min() + xs.max()).float() / 2
        center_y = (ys.min() + ys.max()).float() / 2
        radius = torch.maximum(xs.max() - xs.min(), ys.max() - ys.min()).float() / 2
        radius = radius.clamp_min(1.0)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=image.device),
            torch.arange(width, device=image.device),
            indexing="ij",
        )
        circle = (grid_x - center_x).square() + (grid_y - center_y).square() <= radius.square()
        masks.append(valid & circle)
        centers.append(torch.stack([center_x, center_y]))
        radii.append(radius)
    return torch.stack(masks)[:, None], torch.stack(centers), torch.stack(radii)


def patch_geometry(images: torch.Tensor, patch_shape: Tuple[int, int], config: GeometryConfig):
    """Map raw images to per-patch coverage, polar coordinates and bin IDs."""
    mask, centers, radii = estimate_wafer_geometry(images, config.background_threshold)
    height, width = images.shape[-2:]
    patch_h, patch_w = patch_shape
    coverage = F.adaptive_avg_pool2d(mask.float(), patch_shape).flatten(1)
    xs = (torch.arange(patch_w, device=images.device) + 0.5) * width / patch_w - 0.5
    ys = (torch.arange(patch_h, device=images.device) + 0.5) * height / patch_h - 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    dx = grid_x[None] - centers[:, 0, None, None]
    dy = grid_y[None] - centers[:, 1, None, None]
    radius = torch.sqrt(dx.square() + dy.square()) / radii[:, None, None]
    angle = torch.remainder(torch.atan2(dy, dx), 2 * torch.pi)
    radial = torch.clamp((radius * config.radial_bins).long(), 0, config.radial_bins - 1)
    angular = torch.clamp((angle / (2 * torch.pi) * config.angular_bins).long(), 0, config.angular_bins - 1)
    bin_ids = torch.where(radial == 0, torch.zeros_like(radial), 1 + (radial - 1) * config.angular_bins + angular)
    valid = (coverage >= config.min_wafer_coverage) & (radius.flatten(1) <= 1.0)
    return {
        "valid": valid,
        "coverage": coverage,
        "radius": radius.flatten(1),
        "angle": angle.flatten(1),
        "bin_ids": bin_ids.flatten(1),
        "wafer_mask": mask,
    }


def decode_bin(bin_id: int, config: GeometryConfig) -> Tuple[int, int]:
    if bin_id == 0:
        return 0, 0
    value = bin_id - 1
    return 1 + value // config.angular_bins, value % config.angular_bins


def encode_bin(radial: int, angular: int, config: GeometryConfig) -> int:
    if radial <= 0:
        return 0
    return 1 + (radial - 1) * config.angular_bins + angular % config.angular_bins


def neighboring_bins(bin_id: int, config: GeometryConfig, expand: int = 1) -> List[int]:
    radial, angular = decode_bin(int(bin_id), config)
    if radial == 0:
        result = {0}
        if config.radial_bins > 1:
            result.update(encode_bin(1, a, config) for a in range(config.angular_bins))
        return sorted(result)
    result = set()
    radial_span = config.radial_neighbors * expand
    angular_span = config.angular_neighbors * expand
    for r in range(max(0, radial - radial_span), min(config.radial_bins - 1, radial + radial_span) + 1):
        if r == 0:
            result.add(0)
        else:
            for offset in range(-angular_span, angular_span + 1):
                result.add(encode_bin(r, angular + offset, config))
    return sorted(result)
