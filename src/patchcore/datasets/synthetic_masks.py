"""Paired synthetic wafer-map image and mask dataset."""

from enum import Enum
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from patchcore.datasets.wafer_transforms import (
    DEFAULT_WAFER_BACKGROUND,
    transform_wafer,
)
from patchcore.datasets.wm811k import WM811K_MEAN, WM811K_STD, wm811kDataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "val"


def _normalized_stem(path: Path) -> str:
    stem = path.stem
    return stem[:-5] if stem.lower().endswith("_mask") else stem


def _filter_image_and_mask(image, mask, filter_window_size, filter_threshold):
    """Apply WM811K noise filtering to an image and remove matching mask pixels."""
    gray_img_arr = np.array(image.convert("L"))
    class_map = np.ones_like(gray_img_arr, dtype=np.float32)
    class_map[gray_img_arr > 200] = 2
    class_map[(gray_img_arr > 100) & (gray_img_arr <= 200)] = 0.5
    class_map[gray_img_arr <= 100] = 0

    from scipy.ndimage import uniform_filter

    mean_map = uniform_filter(class_map, size=filter_window_size, mode="constant")
    removed_anomaly_pixels = (class_map == 2) & (mean_map < filter_threshold)

    filtered_image = wm811kDataset.constrained_mean_filter(
        image, filter_window_size, filter_threshold)
    mask_arr = np.array(mask.convert("L"))
    mask_arr[removed_anomaly_pixels] = 0
    filtered_mask = Image.fromarray(mask_arr, mode="L")
    return filtered_image, filtered_mask


class SyntheticMaskDataset(Dataset):
    def __init__(
        self,
        root,
        split="train",
        resize=64,
        imagesize=64,
        allow_empty_masks=False,
        transform_mode="resize_pad",
        pad_color=DEFAULT_WAFER_BACKGROUND,
        apply_filter=True,
        filter_window_size=3,
        filter_threshold=1.25,
    ):
        self.root = Path(root)
        self.resize = resize
        self.imagesize = imagesize
        self.allow_empty_masks = allow_empty_masks
        self.transform_mode = transform_mode
        self.pad_color = tuple(pad_color)
        self.apply_filter = apply_filter
        self.filter_window_size = filter_window_size
        self.filter_threshold = filter_threshold
        image_root = self.root / "images" / split
        mask_root = self.root / "masks" / split
        if not image_root.is_dir() or not mask_root.is_dir():
            raise ValueError(f"Expected {image_root} and {mask_root} directories.")
        masks = {}
        for path in mask_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                key = (path.parent.relative_to(mask_root), _normalized_stem(path))
                if key in masks:
                    raise ValueError(f"Duplicate mask for {key}: {masks[key]} and {path}")
                masks[key] = path
        self.samples = []
        for image_path in sorted(image_root.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            key = (image_path.parent.relative_to(image_root), _normalized_stem(image_path))
            mask_path = masks.pop(key, None)
            if mask_path is None:
                raise ValueError(f"Missing mask for synthetic image {image_path}")
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                if image.size != mask.size:
                    raise ValueError(f"Image/mask size mismatch: {image_path}, {mask_path}")
                if not self.allow_empty_masks and mask.convert("L").getbbox() is None:
                    raise ValueError(f"Empty anomaly mask: {mask_path}")
            self.samples.append((image_path, mask_path))
        if masks:
            raise ValueError(f"Masks without images: {list(masks.values())[:3]}")
        if not self.samples:
            raise ValueError(f"No paired synthetic samples found under {self.root}")

    def __len__(self):
        return len(self.samples)

    def _transform(self, image, interpolation):
        fill = 0 if image.mode == "L" else self.pad_color
        return transform_wafer(
            image, self.resize, self.imagesize, interpolation,
            mode=self.transform_mode, fill=fill)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.apply_filter:
            image, mask = _filter_image_and_mask(
                image, mask, self.filter_window_size, self.filter_threshold)
        raw_image = TF.to_tensor(self._transform(image, TF.InterpolationMode.BILINEAR))
        mask = TF.to_tensor(self._transform(mask, TF.InterpolationMode.NEAREST)) > 0.5
        image = TF.normalize(raw_image, WM811K_MEAN, WM811K_STD)
        return {"image": image, "raw_image": raw_image, "mask": mask.float(), "image_path": str(image_path)}


class SyntheticMaskPatchCoreDataset(Dataset):
    """PatchCore-compatible synthetic wafer dataset with exact masks.

    Training uses only synthetic good wafers. Testing uses the validation split
    and exact generated masks, either for all categories or one requested class.
    """

    def __init__(
        self,
        source,
        classname="all",
        resize=64,
        imagesize=64,
        split=DatasetSplit.TRAIN,
        transform_mode="resize_pad",
        pad_color=DEFAULT_WAFER_BACKGROUND,
        apply_filter=True,
        filter_window_size=3,
        filter_threshold=1.25,
        **kwargs,
    ):
        self.source = Path(source)
        self.classname = classname or "all"
        self.resize = resize
        self.imagesize = (3, imagesize, imagesize)
        self._imagesize_value = imagesize
        self.split = split
        self.transform_mode = transform_mode
        self.pad_color = tuple(pad_color)
        self.apply_filter = apply_filter
        self.filter_window_size = filter_window_size
        self.filter_threshold = filter_threshold
        self.samples = self._collect_samples()
        if not self.samples:
            raise ValueError(f"No synthetic samples found for split={split} class={classname}")
        self.data_to_iterate = [
            ["synthetic_formula", category, str(image_path), str(mask_path)]
            for category, image_path, mask_path in self.samples
        ]

    def _collect_samples(self):
        if self.split == DatasetSplit.TRAIN:
            categories = ["good"]
            split_name = "train"
        else:
            split_name = "val"
            image_root = self.source / "images" / split_name
            if self.classname in {None, "all", "*"}:
                categories = sorted(path.name for path in image_root.iterdir() if path.is_dir())
            else:
                categories = [self.classname]
        samples = []
        for category in categories:
            image_dir = self.source / "images" / split_name / category
            mask_dir = self.source / "masks" / split_name / category
            if not image_dir.is_dir() or not mask_dir.is_dir():
                raise ValueError(f"Missing synthetic image/mask folder for {split_name}/{category}")
            masks = {
                _normalized_stem(path): path
                for path in mask_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            }
            for image_path in sorted(image_dir.iterdir()):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                mask_path = masks.get(_normalized_stem(image_path))
                if mask_path is None:
                    raise ValueError(f"Missing mask for synthetic image {image_path}")
                with Image.open(image_path) as image, Image.open(mask_path) as mask:
                    if image.size != mask.size:
                        raise ValueError(f"Image/mask size mismatch: {image_path}, {mask_path}")
                samples.append((category, image_path, mask_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def _transform(self, image, interpolation):
        fill = 0 if image.mode == "L" else self.pad_color
        return transform_wafer(
            image, self.resize, self._imagesize_value, interpolation,
            mode=self.transform_mode, fill=fill)

    def __getitem__(self, index):
        category, image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.apply_filter:
            image, mask = _filter_image_and_mask(
                image, mask, self.filter_window_size, self.filter_threshold)
        raw_image = TF.to_tensor(self._transform(image, TF.InterpolationMode.BILINEAR))
        image = TF.normalize(raw_image, WM811K_MEAN, WM811K_STD)
        mask = (TF.to_tensor(self._transform(mask, TF.InterpolationMode.NEAREST)) > 0.5).float()
        return {
            "image": image,
            "raw_image": raw_image,
            "mask": mask,
            "classname": "synthetic_formula",
            "anomaly": category,
            "is_anomaly": int(category != "good"),
            "image_name": str(image_path.relative_to(self.source)),
            "image_path": str(image_path),
        }
