"""Train an anomaly type classifier on a small labeled set (e.g., syn).

Usage (PowerShell):
  $env:PYTHONPATH='src'; python .\bin\train_anomaly_classifier.py \
      C:\path\to\wm811k\syn --save_path C:\tmp\anom_cls.pth --epochs 30 --batch_size 32
"""
import os
import sys

# Ensure src is in pythonpath and takes precedence over installed packages
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Force reload of patchcore if it was already loaded from site-packages
for key in list(sys.modules.keys()):
    if key.startswith("patchcore"):
        del sys.modules[key]

import math
import logging
from collections import Counter
from typing import Optional
import numpy as np

import click
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from patchcore.datasets.wm811k_cls import WM811KAnomalyClassDataset
from patchcore.classifier import AnomalyClassifier

# Debug import source
try:
    import patchcore.classifier
    print(f"DEBUG: patchcore.classifier loaded from: {patchcore.classifier.__file__}")
except Exception:
    pass

LOGGER = logging.getLogger("anom_cls_train")


@click.command()
@click.argument("data_root", type=click.Path(exists=True, file_okay=False))
@click.option("--save_path", type=str, required=True, help="Where to save classifier weights (.pth)")
@click.option("--model_name", type=str, default="tf_efficientnet_b3_ns", show_default=True)
@click.option("--epochs", type=int, default=25, show_default=True)
@click.option("--batch_size", type=int, default=32, show_default=True)
@click.option("--learning_rate", type=float, default=2e-4, show_default=True)
@click.option("--weight_decay", type=float, default=1e-4, show_default=True)
@click.option("--val_split", type=float, default=0.15, show_default=True)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--freeze_backbone_epochs", type=int, default=0, show_default=True, help="Freeze backbone for first N epochs (transfer learning stage).")
@click.option("--use_mixup", is_flag=True, help="Enable MixUp augmentation during training.")
@click.option("--mixup_alpha", type=float, default=0.2, show_default=True, help="Alpha for Beta distribution in MixUp.")
@click.option("--label_smoothing", type=float, default=0.0, show_default=True, help="Apply label smoothing in CrossEntropyLoss (ignored if focal loss enabled).")
@click.option("--use_class_weights", is_flag=True, help="Use inverse-frequency class weights for the loss.")
@click.option("--use_focal_loss", is_flag=True, help="Replace CE with focal loss (applies after optional class weighting).")
@click.option("--focal_gamma", type=float, default=2.0, show_default=True, help="Focusing parameter gamma for focal loss.")
@click.option("--apply_filter/--no-apply_filter", is_flag=True, default=True, show_default=True, help="Apply constrained_mean_filter preprocessing (MUST match PatchCore training setting!).")
@click.option("--resize", type=int, default=256, show_default=True, help="Resize dimension before crop")
@click.option("--imagesize", type=int, default=224, show_default=True, help="Final crop size")
def main(data_root: str, save_path: str, model_name: str, epochs: int, batch_size: int,
         learning_rate: float, weight_decay: float, val_split: float, gpu,
         freeze_backbone_epochs: int, use_mixup: bool, mixup_alpha: float, label_smoothing: float,
         use_class_weights: bool, use_focal_loss: bool, focal_gamma: float,
         apply_filter: bool, resize: int, imagesize: int):
    device = torch.device(f"cuda:{gpu[0]}" if torch.cuda.is_available() and gpu else "cpu")
    logging.basicConfig(level=logging.INFO)
    LOGGER.info(f"Using device: {device}")
    LOGGER.info(f"Preprocessing: apply_filter={apply_filter}, resize={resize}, imagesize={imagesize}")

    # Dataset
    full_ds = WM811KAnomalyClassDataset(
        data_root, 
        resize=resize,
        imagesize=imagesize,
        apply_filter=apply_filter,
        extra_augs=True
    )
    num_classes = len(full_ds.classes)
    assert num_classes >= 2, "Need at least 2 classes in data_root"
    val_len = int(math.ceil(val_split * len(full_ds)))
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = AnomalyClassifier(num_classes=num_classes, model_name=model_name).to(device)

    # Loss & Optim
    # Estimate class counts from training subset (fallback uniform if unavailable)
    class_weights_tensor = None
    if use_class_weights:
        counter = Counter()
        if isinstance(train_ds, torch.utils.data.Subset):
            for idx in train_ds.indices:  # type: ignore[attr-defined]
                _, label = full_ds.samples[idx]
                counter[label] += 1
        else:
            for _, label in full_ds.samples:
                counter[label] += 1
        weights = []
        for cls_idx in range(num_classes):
            count = counter.get(cls_idx, 0)
            if count == 0:
                weights.append(1.0)
            else:
                weights.append(train_len / (num_classes * count))
        class_weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    if use_focal_loss:
        if label_smoothing > 0:
            LOGGER.warning("Label smoothing is ignored when using focal loss.")

        class FocalLoss(nn.Module):
            def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, reduction: str = "mean") -> None:
                super().__init__()
                self.gamma = gamma
                self.weight = weight
                self.reduction = reduction

            def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                ce = nn.functional.cross_entropy(logits, target, reduction="none", weight=self.weight)
                pt = torch.exp(-ce)
                loss = ((1 - pt) ** self.gamma) * ce
                if self.reduction == "mean":
                    return loss.mean()
                if self.reduction == "sum":
                    return loss.sum()
                return loss

        criterion = FocalLoss(gamma=focal_gamma, weight=class_weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=label_smoothing)

    # Freeze backbone params if requested (head stays trainable)
    if freeze_backbone_epochs > 0:
        for p in model.backbone.parameters():
            p.requires_grad = False
        optimizer = optim.AdamW(model.head.parameters(), lr=learning_rate, weight_decay=weight_decay)
        LOGGER.info(f"Freezing backbone for first {freeze_backbone_epochs} epoch(s).")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_acc = 0.0
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        # If we just finished freeze period, unfreeze backbone and rebuild optimizer (single time)
        if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs + 1:
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate * 0.3, weight_decay=weight_decay)
            LOGGER.info(f"Unfroze backbone at epoch {epoch}. New lr={learning_rate * 0.3:.2e}")
        running_loss = 0.0
        correct = 0
        total = 0
        for batch in tqdm(train_loader, desc=f"Train {epoch}/{epochs}"):
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            if use_mixup:
                if imgs.size(0) < 2:
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                else:
                    lam = np.random.beta(mixup_alpha, mixup_alpha)
                    perm = torch.randperm(imgs.size(0))
                    mixed_imgs = lam * imgs + (1 - lam) * imgs[perm]
                    logits = model(mixed_imgs)
                    loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels[perm])
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        # Validation
        model.eval()
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Valid"):
                imgs = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(imgs)
                preds = logits.argmax(1)
                v_correct += (preds == labels).sum().item()
                v_total += imgs.size(0)
        val_acc = v_correct / max(1, v_total)
        scheduler.step()

        LOGGER.info(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_name": model_name,
                "num_classes": num_classes,
                "state_dict": model.state_dict(),
                "classes": full_ds.classes,
            }, save_path)
            LOGGER.info(f"Saved best model to {save_path} (val_acc={best_acc:.4f})")


if __name__ == "__main__":
    main()
