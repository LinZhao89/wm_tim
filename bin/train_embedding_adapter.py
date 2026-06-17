"""Train a mask-guided residual adapter on aggregated patch embeddings."""
import itertools
from pathlib import Path
import click
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import patchcore.backbones
import patchcore.common
import patchcore.sampler
from patchcore.datasets.synthetic_masks import SyntheticMaskDataset
from patchcore.datasets.wm811k import DatasetSplit, wm811kDataset
from patchcore.geometry import neighboring_bins
from patchcore.geometry_patchcore import PatchCore
from patchcore.networks.embedding_adapter import ResidualEmbeddingAdapter
from train_mask_cbam import pro_auc


def collect_real(core, loader, device, limit=20000):
    features, bins = [], []
    with torch.no_grad():
        for batch in loader:
            values, _, geometry = core._embed(
                batch["image"].to(device), batch["raw_image"].to(device),
                detach=False, provide_patch_shapes=True)
            valid = geometry["valid"].reshape(-1)
            features.append(values[valid].detach().cpu())
            bins.append(geometry["bin_ids"].reshape(-1)[valid].detach().cpu())
            if sum(len(value) for value in features) >= limit:
                break
    return torch.cat(features)[:limit].to(device), torch.cat(bins)[:limit].to(device)


def derive_margins(features, bins):
    margins = {}
    for bin_id in bins.unique().tolist():
        values = features[bins == bin_id][:2048]
        if len(values) < 2:
            margins[int(bin_id)] = 1.0
            continue
        distances = torch.cdist(values, values)
        distances.fill_diagonal_(float("inf"))
        nearest = distances.min(dim=1).values
        margins[int(bin_id)] = 1.5 * torch.quantile(nearest, 0.95).item()
    return margins


@click.command()
@click.argument("synthetic_root", type=click.Path(exists=True, file_okay=False))
@click.argument("real_normal_root", type=click.Path(exists=True, file_okay=False))
@click.option("--subdataset", required=True)
@click.option("--cbam_checkpoint", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--save_path", required=True, type=click.Path(dir_okay=False))
@click.option("--backbone_name", default="wideresnet50", show_default=True)
@click.option("--layer", "layers", multiple=True, default=("layer2", "layer3"))
@click.option("--target_dimension", default=1024, show_default=True)
@click.option("--resize", default=256, show_default=True)
@click.option("--imagesize", default=224, show_default=True)
@click.option("--epochs", default=20, show_default=True)
@click.option("--batch_size", default=4, show_default=True)
@click.option("--learning_rate", default=1e-4, show_default=True)
@click.option("--gpu", default=0, show_default=True)
@click.option("--real_feature_limit", default=20000, show_default=True)
@click.option("--max_train_batches", default=None, type=int)
@click.option("--max_val_batches", default=None, type=int)
@click.option(
    "--transform_mode",
    type=click.Choice(["resize_pad", "resize_only", "resize_crop"]),
    default="resize_pad",
    show_default=True,
)
def main(synthetic_root, real_normal_root, subdataset, cbam_checkpoint,
         save_path, backbone_name, layers, target_dimension, resize, imagesize,
         epochs, batch_size, learning_rate, gpu, real_feature_limit,
         max_train_batches, max_val_batches, transform_mode):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    backbone = patchcore.backbones.load(backbone_name)
    backbone.name = backbone_name
    core = PatchCore(device)
    core.load(
        backbone, list(layers), device, (3, imagesize, imagesize),
        1024, target_dimension, featuresampler=patchcore.sampler.IdentitySampler(),
        nn_method=patchcore.common.FaissNN(False, 4),
        cbam_checkpoint=cbam_checkpoint)
    real_dataset = wm811kDataset(
        real_normal_root, subdataset, resize=resize, imagesize=imagesize,
        split=DatasetSplit.TRAIN, transform_mode=transform_mode)
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
    real_features, real_bins = collect_real(
        core, real_loader, device, limit=real_feature_limit)
    real_targets = F.normalize(real_features, dim=-1)
    margins = derive_margins(real_targets, real_bins)
    adapter = ResidualEmbeddingAdapter(target_dimension).to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=learning_rate)
    train_loader = DataLoader(
        SyntheticMaskDataset(
            synthetic_root, "train", resize, imagesize,
            allow_empty_masks=True, transform_mode=transform_mode),
        batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        SyntheticMaskDataset(
            synthetic_root, "val", resize, imagesize,
            allow_empty_masks=True, transform_mode=transform_mode),
        batch_size=batch_size, shuffle=False)
    best = -1.0
    for epoch in range(epochs):
        adapter.train()
        for batch_index, batch in enumerate(train_loader):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            base, shapes, geometry = core._embed(
                batch["image"].to(device), batch["raw_image"].to(device),
                detach=False, provide_patch_shapes=True)
            adapted = adapter(base)
            occupancy = F.adaptive_avg_pool2d(
                batch["mask"].to(device), tuple(shapes[0])).reshape(-1)
            normal_mask = occupancy <= 0.01
            anomaly_mask = occupancy >= 0.10
            synthetic_bins = geometry["bin_ids"].reshape(-1)
            real_indices = torch.randint(
                len(real_targets), (min(4096, len(real_targets)),), device=device)
            sampled_real = real_features[real_indices]
            sampled_real_bins = real_bins[real_indices]
            identity = F.mse_loss(adapter(sampled_real), F.normalize(sampled_real, dim=-1))
            alignment = torch.tensor(0.0, device=device)
            for bin_id in synthetic_bins[normal_mask].unique():
                synthetic_values = adapted[normal_mask & (synthetic_bins == bin_id)]
                real_values = adapter(sampled_real[sampled_real_bins == bin_id])
                if len(synthetic_values) and len(real_values):
                    alignment = alignment + F.mse_loss(
                        synthetic_values.mean(0), real_values.mean(0))
            margin_loss = torch.tensor(0.0, device=device)
            for bin_id in synthetic_bins[anomaly_mask].unique():
                allowed = neighboring_bins(int(bin_id), core.geometry_config)
                reference_mask = torch.zeros_like(real_bins, dtype=torch.bool)
                for candidate in allowed:
                    reference_mask |= real_bins == candidate
                references = adapter(real_features[reference_mask][:4096]).detach()
                anomalies = adapted[anomaly_mask & (synthetic_bins == bin_id)]
                if len(references) and len(anomalies):
                    distance = torch.cdist(anomalies, references).min(dim=1).values
                    margin_loss = margin_loss + F.relu(
                        margins.get(int(bin_id), 1.0) - distance).mean()
            loss = identity + alignment + margin_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        adapter.eval()
        score_batches, mask_batches = [], []
        with torch.no_grad():
            adapted_real = adapter(real_features)
            for batch_index, batch in enumerate(val_loader):
                if max_val_batches is not None and batch_index >= max_val_batches:
                    break
                base, shapes, geometry = core._embed(
                    batch["image"].to(device), batch["raw_image"].to(device),
                    detach=False, provide_patch_shapes=True)
                values = adapter(base)
                scores = torch.zeros(len(values), device=device)
                bins = geometry["bin_ids"].reshape(-1)
                valid = geometry["valid"].reshape(-1)
                for bin_id in bins[valid].unique():
                    allowed = neighboring_bins(int(bin_id), core.geometry_config)
                    reference_mask = torch.zeros_like(real_bins, dtype=torch.bool)
                    for candidate in allowed:
                        reference_mask |= real_bins == candidate
                    query = valid & (bins == bin_id)
                    scores[query] = torch.cdist(
                        values[query], adapted_real[reference_mask][:4096]
                    ).min(dim=1).values
                grid = shapes[0]
                score_map = scores.reshape(
                    len(batch["image"]), 1, grid[0], grid[1])
                score_map = F.interpolate(
                    score_map, size=(imagesize, imagesize),
                    mode="bilinear", align_corners=False)
                score_batches.append(score_map.cpu().numpy()[:, 0])
                mask_batches.append(batch["mask"].numpy()[:, 0].astype(bool))
        metric = pro_auc(
            np.concatenate(score_batches), np.concatenate(mask_batches))
        click.echo(f"epoch={epoch + 1} val_aupro={metric:.5f}")
        if metric > best:
            best = metric
            torch.save({
                "dimension": target_dimension,
                "dropout": 0.1,
                "state_dict": adapter.state_dict(),
                "validation_aupro": metric,
                "margins": margins,
                "transform_mode": transform_mode,
            }, save_path)


if __name__ == "__main__":
    main()
