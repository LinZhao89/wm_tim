"""Compute mean and std statistics for wafermap dataset normalization.

Run this script once on your training data to compute dataset-specific
normalization statistics instead of using ImageNet defaults.

Usage (PowerShell):
    $env:PYTHONPATH='src'; python .\bin\compute_dataset_stats.py dataset/wm811k prepare_dataset_train_ratio10p --imagesize 128
"""
import sys
import click
import torch
import numpy as np
from torchvision import transforms
import PIL

# Add parent to path for imports
sys.path.insert(0, 'src')
from patchcore.datasets.wm811k import wm811kDataset, DatasetSplit


@click.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.argument("subdataset", type=str)
@click.option("--imagesize", default=224, type=int, help="Image size to use")
@click.option("--resize", default=256, type=int, help="Initial resize before crop")
@click.option("--batch_size", default=32, type=int, help="Batch size for computation")
@click.option("--num_workers", default=4, type=int, help="DataLoader workers")
def main(data_path, subdataset, imagesize, resize, batch_size, num_workers):
    """Compute mean and std for wafermap dataset normalization."""
    
    print(f"Computing statistics for dataset: {subdataset}")
    print(f"Data path: {data_path}")
    print(f"Image size: {imagesize}, Resize: {resize}")
    print("-" * 60)
    
    # Create a minimal dataset without normalization to compute raw stats
    # We'll temporarily monkey-patch the dataset to skip normalization
    dataset = wm811kDataset(
        source=data_path,
        classname=subdataset,
        resize=resize,
        imagesize=imagesize,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
    )
    
    # Replace transform to only resize/crop/toTensor (no normalize)
    dataset.transform_img = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(imagesize),
        transforms.ToTensor(),  # converts to [0, 1]
    ])
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    print(f"Processing {len(dataset)} training images...")
    
    # Compute mean and std across all images
    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_samples = 0
    
    for batch in dataloader:
        images = batch['image']  # [B, 3, H, W]
        batch_samples = images.size(0)
        
        # Compute mean per channel
        mean += images.mean([0, 2, 3]) * batch_samples
        # Compute std per channel
        std += images.std([0, 2, 3]) * batch_samples
        
        n_samples += batch_samples
    
    mean = mean / n_samples
    std = std / n_samples
    
    print("\n" + "=" * 60)
    print("COMPUTED STATISTICS:")
    print("=" * 60)
    print(f"Mean (R, G, B): [{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}]")
    print(f"Std  (R, G, B): [{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]")
    print("=" * 60)
    
    print("\n📋 Copy-paste this into wm811k.py (replace IMAGENET_MEAN/STD):")
    print("-" * 60)
    print(f"WM811K_MEAN = [{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}]")
    print(f"WM811K_STD = [{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]")
    print("-" * 60)
    
    # Also compute per-channel min/max for reference
    print("\n📊 Additional statistics (for reference):")
    all_pixels = []
    for batch in dataloader:
        images = batch['image']
        all_pixels.append(images.view(images.size(0), 3, -1))
    
    all_pixels = torch.cat(all_pixels, dim=0)  # [N, 3, H*W]
    
    for c, name in enumerate(['R', 'G', 'B']):
        pixels_c = all_pixels[:, c, :]
        print(f"  {name}: min={pixels_c.min():.4f}, max={pixels_c.max():.4f}, "
              f"median={pixels_c.median():.4f}")
    
    print("\n✅ Done! Update wm811k.py with the computed values above.")


if __name__ == "__main__":
    main()
