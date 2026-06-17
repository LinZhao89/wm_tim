"""Train layer-wise CBAM modules from synthetic anomaly masks."""
import click
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

import patchcore.backbones
import patchcore.common
from patchcore.datasets.synthetic_masks import SyntheticMaskDataset
from patchcore.networks.cbam import CBAM


def dice_loss(logits, targets):
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum((1, 2, 3))
    denominator = probabilities.sum((1, 2, 3)) + targets.sum((1, 2, 3))
    return (1 - (2 * intersection + 1) / (denominator + 1)).mean()


def pro_auc(scores, masks, steps=50):
    values = []
    thresholds = np.linspace(float(scores.min()), float(scores.max()), steps)
    for threshold in thresholds:
        prediction = scores >= threshold
        false_positive = prediction[~masks].mean() if np.any(~masks) else 0.0
        overlap = prediction[masks].mean() if np.any(masks) else 0.0
        if false_positive <= 0.3:
            values.append((false_positive, overlap))
    if len(values) < 2:
        return 0.0
    values.sort()
    x, y = np.asarray(values).T
    return float(np.trapezoid(y, x) / 0.3)


@click.command()
@click.argument("synthetic_root", type=click.Path(exists=True, file_okay=False))
@click.option("--save_path", required=True, type=click.Path(dir_okay=False))
@click.option("--backbone_name", default="wideresnet50", show_default=True)
@click.option("--layer", "layers", multiple=True, default=("layer2", "layer3"))
@click.option("--resize", default=256, show_default=True)
@click.option("--imagesize", default=224, show_default=True)
@click.option("--epochs", default=30, show_default=True)
@click.option("--batch_size", default=8, show_default=True)
@click.option("--learning_rate", default=1e-4, show_default=True)
@click.option("--gpu", default=0, show_default=True)
@click.option("--max_train_batches", default=None, type=int)
@click.option("--max_val_batches", default=None, type=int)
@click.option(
    "--transform_mode",
    type=click.Choice(["resize_pad", "resize_only", "resize_crop"]),
    default="resize_pad",
    show_default=True,
)
def main(synthetic_root, save_path, backbone_name, layers, resize, imagesize,
         epochs, batch_size, learning_rate, gpu, max_train_batches,
         max_val_batches, transform_mode):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    backbone = patchcore.backbones.load(backbone_name).to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    extractor = patchcore.common.NetworkFeatureAggregator(
        backbone, list(layers), device)
    dimensions = extractor.feature_dimensions((3, imagesize, imagesize))
    modules = torch.nn.ModuleList([CBAM(d) for d in dimensions]).to(device)
    heads = torch.nn.ModuleList([
        torch.nn.Conv2d(d, 1, kernel_size=1) for d in dimensions]).to(device)
    optimizer = torch.optim.Adam(
        list(modules.parameters()) + list(heads.parameters()), lr=learning_rate)
    loaders = {
        split: DataLoader(
            SyntheticMaskDataset(
                synthetic_root, split, resize, imagesize,
                allow_empty_masks=True, transform_mode=transform_mode),
            batch_size=batch_size, shuffle=split == "train")
        for split in ("train", "val")}
    best = -1.0
    for epoch in range(epochs):
        modules.train()
        for batch_index, batch in enumerate(loaders["train"]):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            with torch.no_grad():
                features = extractor(images)
            loss = 0.0
            for index, layer in enumerate(layers):
                original = features[layer]
                attended = modules[index](original)
                target = F.interpolate(
                    masks, size=original.shape[-2:], mode="nearest")
                logits = heads[index](attended)
                normal = 1 - target
                preservation = ((attended - original).square() * normal).mean()
                loss = loss + F.binary_cross_entropy_with_logits(
                    logits, target) + dice_loss(logits, target) + 0.05 * preservation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        modules.eval()
        score_batches, mask_batches = [], []
        with torch.no_grad():
            for batch_index, batch in enumerate(loaders["val"]):
                if max_val_batches is not None and batch_index >= max_val_batches:
                    break
                images = batch["image"].to(device)
                masks = batch["mask"].numpy().astype(bool)
                features = extractor(images)
                layer_scores = []
                for index, layer in enumerate(layers):
                    logits = heads[index](modules[index](features[layer]))
                    layer_scores.append(F.interpolate(
                        torch.sigmoid(logits), size=(imagesize, imagesize),
                        mode="bilinear", align_corners=False))
                score_batches.append(
                    torch.stack(layer_scores).mean(0).cpu().numpy()[:, 0])
                mask_batches.append(masks[:, 0])
        metric = pro_auc(
            np.concatenate(score_batches), np.concatenate(mask_batches))
        click.echo(f"epoch={epoch + 1} val_aupro={metric:.5f}")
        if metric > best:
            best = metric
            torch.save({
                "backbone": backbone_name,
                "layers": list(layers),
                "dimensions": dimensions,
                "resize": resize,
                "imagesize": imagesize,
                "transform_mode": transform_mode,
                "reduction": 16,
                "spatial_kernel": 7,
                "state_dict": modules.state_dict(),
                "validation_aupro": metric,
            }, save_path)


if __name__ == "__main__":
    main()
