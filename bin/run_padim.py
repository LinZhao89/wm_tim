"""PaDiM baseline for WM811K with the established wafer preprocessing protocol."""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import patchcore.backbones
from patchcore.datasets.wm811k import DatasetSplit, wm811kDataset
from patchcore.datasets.synthetic_masks import DatasetSplit as SyntheticSplit
from patchcore.datasets.synthetic_masks import SyntheticMaskPatchCoreDataset


class FeatureExtractor(torch.nn.Module):
    def __init__(self, backbone, layers):
        super().__init__()
        self.backbone = backbone
        self.layers = layers
        self.outputs = {}
        for name in layers:
            backbone._modules[name].register_forward_hook(self._hook(name))

    def _hook(self, name):
        def store(_, __, output):
            self.outputs[name] = output
        return store

    def forward(self, images):
        self.outputs = {}
        self.backbone(images)
        return self.outputs


def features(extractor, images, layers, target_hw):
    with torch.no_grad():
        outputs = extractor(images)
    maps = [outputs[layer] for layer in layers]
    maps = [F.interpolate(item, size=target_hw, mode="bilinear", align_corners=False) for item in maps]
    return torch.cat(maps, dim=1)


def pixel_auc(scores, masks):
    values = np.asarray(scores).reshape(-1)
    labels = (np.asarray(masks).reshape(-1) > 0).astype(np.uint8)
    return float(roc_auc_score(labels, values)) if np.unique(labels).size == 2 else float("nan")


def collect_scores(loader, extractor, layers, target_hw, indices, mean, inv_cov, device):
    image_scores, image_labels, masks, maps = [], [], [], []
    for batch in loader:
        embedding = features(extractor, batch["image"].to(device), layers, target_hw)
        embedding = embedding[:, indices].permute(0, 2, 3, 1).cpu().numpy()
        centered = embedding - mean[None]
        distances = np.einsum("bhwd,hwdk,bhwk->bhw", centered, inv_cov, centered, optimize=True)
        distances = np.sqrt(np.maximum(distances, 0))
        resized = F.interpolate(torch.from_numpy(distances).unsqueeze(1), size=batch["mask"].shape[-2:], mode="bilinear", align_corners=False).squeeze(1).numpy()
        resized = np.asarray([ndimage.gaussian_filter(item, sigma=4) for item in resized])
        maps.extend(resized)
        masks.extend(batch["mask"].numpy()[:, 0])
        image_scores.extend(resized.reshape(len(resized), -1).max(axis=1))
        image_labels.extend(batch["is_anomaly"].numpy().tolist())
    return np.asarray(image_scores), np.asarray(image_labels), np.asarray(maps), np.asarray(masks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--synthetic_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resize", type=int, default=128)
    parser.add_argument("--imagesize", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_train_images", type=int, default=1000)
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    common = dict(resize=args.resize, imagesize=args.imagesize, transform_mode="resize_pad", apply_filter=True, filter_window_size=3, filter_threshold=1.25)
    dataset_root = Path(args.dataset_root)
    train = wm811kDataset(str(dataset_root.parent), dataset_root.name, split=DatasetSplit.TRAIN, **common)
    test = wm811kDataset(str(dataset_root.parent), dataset_root.name, split=DatasetSplit.TEST, **common)
    pixel = SyntheticMaskPatchCoreDataset(args.synthetic_root, "all", split=SyntheticSplit.TEST, **common)
    rng = np.random.default_rng(args.seed)
    selected = rng.choice(len(train), size=min(args.max_train_images, len(train)), replace=False)
    train_loader = DataLoader(Subset(train, selected), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=0)
    pixel_loader = DataLoader(pixel, batch_size=args.batch_size, shuffle=False, num_workers=0)
    layers = ["layer1", "layer2", "layer3"]
    backbone = patchcore.backbones.load("wideresnet50").to(device).eval()
    extractor = FeatureExtractor(backbone, layers).to(device).eval()
    target_hw = (args.imagesize // 4, args.imagesize // 4)
    train_embeddings = []
    for batch in train_loader:
        train_embeddings.append(features(extractor, batch["image"].to(device), layers, target_hw).cpu())
    train_embeddings = torch.cat(train_embeddings).numpy().transpose(0, 2, 3, 1)
    channels = train_embeddings.shape[-1]
    indices = np.random.default_rng(args.seed).choice(channels, size=min(args.embedding_dim, channels), replace=False)
    train_embeddings = train_embeddings[..., indices]
    mean = train_embeddings.mean(axis=0)
    centered = train_embeddings - mean
    covariance = np.einsum("nhwd,nhwk->hwdk", centered, centered, optimize=True) / max(len(train_embeddings) - 1, 1)
    covariance += np.eye(covariance.shape[-1], dtype=np.float32)[None, None] * 0.01
    inv_cov = np.linalg.inv(covariance)
    scores, labels, _, _ = collect_scores(test_loader, extractor, layers, target_hw, indices, mean, inv_cov, device)
    _, _, pixel_maps, pixel_masks = collect_scores(pixel_loader, extractor, layers, target_hw, indices, mean, inv_cov, device)
    metrics = {
        "instance_auroc": float(roc_auc_score(labels, scores)),
        "full_pixel_auroc": pixel_auc(pixel_maps, pixel_masks),
        "anomaly_pixel_auroc": pixel_auc(pixel_maps[np.asarray(pixel_masks).reshape(len(pixel_masks), -1).sum(axis=1) > 0], pixel_masks[np.asarray(pixel_masks).reshape(len(pixel_masks), -1).sum(axis=1) > 0]),
    }
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics)); writer.writeheader(); writer.writerow(metrics)
    with (output / "image_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_score", "is_anomaly"]); writer.writeheader(); writer.writerows({"image_score": float(s), "is_anomaly": int(y)} for s, y in zip(scores, labels))
    print(metrics)


if __name__ == "__main__":
    main()
