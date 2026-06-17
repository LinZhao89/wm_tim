import os
from typing import List, Tuple, Optional

from PIL import Image
import numpy as np
from scipy.ndimage import label, binary_erosion, generate_binary_structure
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from patchcore.datasets.wm811k import WM811K_MEAN, WM811K_STD, wm811kDataset


class WM811KAnomalyClassDataset(Dataset):
    """
    Labeled anomaly classification dataset.

    Expects a directory structure like:
        root/
          Center/
            *.png|*.jpg
          Donut/
            *.png|*.jpg
          ... (8 classes total)

    This dataset is intended for training a discriminative classifier on a
    small labeled set, then used to classify anomalies detected by PatchCore.
    
    IMPORTANT: Set apply_filter=True to match PatchCore preprocessing if your
    PatchCore model was trained with filtering enabled!
    """

    def __init__(
        self,
        root: str,
        resize: int = 256,
        imagesize: int = 224,
        apply_filter: bool = True,
        extra_augs: bool = True,
        class_order: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.apply_filter = apply_filter

        # Build class list from subfolders
        classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        if class_order:
            # Keep only classes present and sort by provided order
            present = [c for c in class_order if c in classes]
            self.classes = present
        else:
            self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Collect samples
        self.samples: List[Tuple[str, int]] = []
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        for cls in self.classes:
            cls_dir = os.path.join(root, cls)
            for fname in sorted(os.listdir(cls_dir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext in exts:
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))

        # Transforms
        tfs: List = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
        ]

        # Optional stronger augs for small datasets
        if extra_augs:
            tfs = [
                transforms.Resize(resize),
                transforms.CenterCrop(imagesize),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            ]

        tfs.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=WM811K_MEAN, std=WM811K_STD),
        ])
        self.transform = transforms.Compose(tfs)

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def constrained_mean_filter(image: Image.Image, filter_window_size: int = 3, threshold: float = 1.25) -> Image.Image:
        """Apply the same constrained mean filter used by PatchCore datasets."""
        return wm811kDataset.constrained_mean_filter(
            image,
            filter_window_size=filter_window_size,
            threshold=threshold,
        )

    def __getitem__(self, idx: int):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        
        # Apply same preprocessing as PatchCore if enabled
        if self.apply_filter:
            img = self.constrained_mean_filter(img, filter_window_size=3, threshold=1.25)
        
        img_t = self.transform(img)
        return {"image": img_t, "label": target, "path": path}
