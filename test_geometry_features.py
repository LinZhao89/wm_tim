import numpy as np
import pytest
import torch
from PIL import Image

from patchcore.geometry import (
    GeometryConfig,
    encode_bin,
    neighboring_bins,
    patch_geometry,
)
from patchcore.sampler import GeometryAwareCoresetSampler


def wafer_image(height=32, width=32, shift_x=0):
    image = torch.zeros(3, height, width)
    yy, xx = torch.meshgrid(
        torch.arange(height), torch.arange(width), indexing="ij")
    mask = (xx - (width // 2 + shift_x)).square() + (
        yy - height // 2).square() <= 10 ** 2
    image[:, mask] = 1
    return image


def test_patch_geometry_filters_background_and_handles_shift():
    config = GeometryConfig(radial_bins=4, angular_bins=8)
    metadata = patch_geometry(
        torch.stack([wafer_image(), wafer_image(shift_x=2)]), (8, 8), config)
    assert metadata["valid"].shape == (2, 64)
    assert metadata["valid"].sum() > 0
    assert metadata["valid"].sum() < 128
    assert metadata["bin_ids"].max() < config.num_bins


def test_angular_neighbors_wrap():
    config = GeometryConfig(radial_bins=4, angular_bins=8)
    first = encode_bin(2, 0, config)
    assert encode_bin(2, 7, config) in neighboring_bins(first, config)


def test_geometry_sampler_is_deterministic_and_balanced():
    features = np.random.RandomState(1).normal(size=(40, 16)).astype(np.float32)
    bins = np.repeat(np.arange(4), 10)
    sampler = GeometryAwareCoresetSampler(0.2, torch.device("cpu"), seed=7)
    first_features, first_bins, first_indices = sampler.run_with_metadata(
        features, bins)
    second_features, second_bins, second_indices = sampler.run_with_metadata(
        features, bins)
    assert len(first_features) == 8
    assert np.array_equal(first_indices, second_indices)
    assert set(first_bins) == {0, 1, 2, 3}


def test_synthetic_pairing_and_empty_mask(tmp_path):
    from patchcore.datasets.synthetic_masks import SyntheticMaskDataset
    for subdir in ("images/train/a", "masks/train/a"):
        (tmp_path / subdir).mkdir(parents=True)
    Image.new("RGB", (16, 16), "white").save(
        tmp_path / "images/train/a/sample.png")
    mask = Image.new("L", (16, 16))
    mask.putpixel((8, 8), 255)
    mask.save(tmp_path / "masks/train/a/sample_mask.png")
    dataset = SyntheticMaskDataset(tmp_path, "train", 16, 16)
    sample = dataset[0]
    assert sample["image"].shape == (3, 16, 16)
    assert sample["mask"].sum() == 1
