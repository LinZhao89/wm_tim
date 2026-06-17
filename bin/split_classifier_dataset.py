"""Split wafermap test data into train/val/test for supervised classifier training.

This script creates a stratified split to prevent data leakage.

Usage (PowerShell):
  $env:PYTHONPATH='src'; python .\bin\split_classifier_dataset.py `
      --source dataset/wm811k/prepare_dataset_train_ratio10p/test `
      --output dataset/wm811k/classifier_data `
      --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 `
      --seed 42
"""
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import click
from tqdm import tqdm


@click.command()
@click.option("--source", type=click.Path(exists=True, file_okay=False), required=True,
              help="Source folder with class subfolders (e.g., test/)")
@click.option("--output", type=click.Path(file_okay=False), required=True,
              help="Output folder for train/val/test splits")
@click.option("--train_ratio", type=float, default=0.7, show_default=True,
              help="Proportion for training set")
@click.option("--val_ratio", type=float, default=0.15, show_default=True,
              help="Proportion for validation set")
@click.option("--test_ratio", type=float, default=0.15, show_default=True,
              help="Proportion for test set")
@click.option("--seed", type=int, default=42, show_default=True,
              help="Random seed for reproducibility")
@click.option("--exclude_good", is_flag=True,
              help="Exclude 'good' class from splitting (keep only anomaly classes)")
@click.option("--symlink", is_flag=True,
              help="Create symlinks instead of copying files (faster, less disk space)")
def main(source: str, output: str, train_ratio: float, val_ratio: float, 
         test_ratio: float, seed: int, exclude_good: bool, symlink: bool):
    """
    Create stratified train/val/test splits from a flat class folder structure.
    
    Input structure:
        source/
          Center/*.png
          Donut/*.png
          ...
    
    Output structure:
        output/
          train/
            Center/*.png
            Donut/*.png
          val/
            Center/*.png
          test/
            Center/*.png
    """
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    assert abs(total - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {total}"
    
    random.seed(seed)
    source_path = Path(source)
    output_path = Path(output)
    
    # Collect all class folders
    class_folders = [d for d in source_path.iterdir() if d.is_dir()]
    if exclude_good:
        class_folders = [d for d in class_folders if d.name.lower() != "good"]
    
    print(f"Found {len(class_folders)} classes: {[d.name for d in class_folders]}")
    
    # Collect files per class
    files_per_class: Dict[str, List[Path]] = defaultdict(list)
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    
    for class_folder in class_folders:
        class_name = class_folder.name
        for file_path in class_folder.iterdir():
            if file_path.suffix.lower() in exts:
                files_per_class[class_name].append(file_path)
    
    # Print class distribution
    print("\nClass distribution:")
    for cls, files in sorted(files_per_class.items()):
        print(f"  {cls}: {len(files)} images")
    
    # Create output directories
    for split in ["train", "val", "test"]:
        for class_name in files_per_class.keys():
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Stratified split per class
    split_stats = defaultdict(lambda: defaultdict(int))
    
    for class_name, files in tqdm(files_per_class.items(), desc="Splitting classes"):
        # Shuffle files
        random.shuffle(files)
        n = len(files)
        
        # Calculate split indices
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # Remaining goes to test to ensure we use all samples
        
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]
        
        # Copy or symlink files
        for split_name, split_files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            for file_path in split_files:
                dest = output_path / split_name / class_name / file_path.name
                
                if symlink:
                    # Create relative symlink
                    dest.symlink_to(file_path.resolve())
                else:
                    # Copy file
                    shutil.copy2(file_path, dest)
                
                split_stats[split_name][class_name] += 1
    
    # Print split statistics
    print("\n" + "="*60)
    print("Split Statistics:")
    print("="*60)
    for split in ["train", "val", "test"]:
        print(f"\n{split.upper()}:")
        total = 0
        for class_name in sorted(split_stats[split].keys()):
            count = split_stats[split][class_name]
            total += count
            print(f"  {class_name:15s}: {count:4d} images")
        print(f"  {'TOTAL':15s}: {total:4d} images")
    
    print("\n" + "="*60)
    print(f"✓ Dataset split complete! Output: {output_path}")
    print("="*60)
    
    # Print suggested training command
    print("\nSuggested training command (PowerShell):")
    print(f"$env:PYTHONPATH='src'; python .\\bin\\train_anomaly_classifier.py `")
    print(f"    {output_path / 'train'} `")
    print(f"    --save_path results/classifier_weights.pth `")
    print(f"    --epochs 50 --batch_size 32 `")
    print(f"    --freeze_backbone_epochs 5 `")
    print(f"    --use_focal_loss --focal_gamma 2.0 `")
    print(f"    --use_class_weights")


if __name__ == "__main__":
    main()
