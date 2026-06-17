"""Diagnose classifier training issues and check for data leakage.

Usage (PowerShell):
  $env:PYTHONPATH='src'; python .\bin\diagnose_classifier.py `
      --train_data dataset/wm811k/classifier_data/train `
      --test_data dataset/wm811k/prepare_dataset_train_ratio10p/test
"""
import os
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Set

import click


def collect_image_paths(root: str) -> Dict[str, Set[str]]:
    """Collect all image paths per class."""
    paths_per_class = defaultdict(set)
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    
    root_path = Path(root)
    for class_folder in root_path.iterdir():
        if not class_folder.is_dir():
            continue
        class_name = class_folder.name
        for img_path in class_folder.iterdir():
            if img_path.suffix.lower() in exts:
                # Store filename only for comparison
                paths_per_class[class_name].add(img_path.name)
    
    return paths_per_class


@click.command()
@click.option("--train_data", type=click.Path(exists=True, file_okay=False), required=True,
              help="Path to training data folder")
@click.option("--test_data", type=click.Path(exists=True, file_okay=False), required=True,
              help="Path to test data folder")
def main(train_data: str, test_data: str):
    """
    Diagnose potential issues with classifier training setup.
    
    Checks:
    1. Data leakage (overlapping files between train and test)
    2. Class distribution balance
    3. Sufficient samples per class
    """
    print("="*80)
    print("CLASSIFIER TRAINING DIAGNOSTIC")
    print("="*80)
    
    # Collect data
    print("\n[1/4] Collecting training data...")
    train_paths = collect_image_paths(train_data)
    
    print("[2/4] Collecting test data...")
    test_paths = collect_image_paths(test_data)
    
    # Check 1: Data leakage
    print("\n" + "="*80)
    print("[3/4] Checking for data leakage...")
    print("="*80)
    
    leakage_found = False
    for class_name in set(train_paths.keys()) | set(test_paths.keys()):
        train_set = train_paths.get(class_name, set())
        test_set = test_paths.get(class_name, set())
        overlap = train_set & test_set
        
        if overlap:
            leakage_found = True
            print(f"⚠️  WARNING: Class '{class_name}' has {len(overlap)} overlapping files!")
            print(f"   Examples: {list(overlap)[:3]}")
    
    if not leakage_found:
        print("✓ No data leakage detected - train and test sets are disjoint")
    else:
        print("\n❌ DATA LEAKAGE DETECTED!")
        print("   Your model has seen test data during training.")
        print("   This explains poor generalization despite high training accuracy.")
        print("\n   FIX: Use bin/split_classifier_dataset.py to create proper splits")
    
    # Check 2: Class distribution
    print("\n" + "="*80)
    print("[4/4] Class Distribution Analysis")
    print("="*80)
    
    print("\nTRAINING SET:")
    train_counts = {cls: len(paths) for cls, paths in train_paths.items()}
    total_train = sum(train_counts.values())
    for cls in sorted(train_counts.keys()):
        count = train_counts[cls]
        pct = 100 * count / max(1, total_train)
        print(f"  {cls:15s}: {count:4d} images ({pct:5.1f}%)")
    print(f"  {'TOTAL':15s}: {total_train:4d} images")
    
    print("\nTEST SET:")
    test_counts = {cls: len(paths) for cls, paths in test_paths.items()}
    total_test = sum(test_counts.values())
    for cls in sorted(test_counts.keys()):
        count = test_counts[cls]
        pct = 100 * count / max(1, total_test)
        print(f"  {cls:15s}: {count:4d} images ({pct:5.1f}%)")
    print(f"  {'TOTAL':15s}: {total_test:4d} images")
    
    # Check 3: Sample size warnings
    print("\n" + "="*80)
    print("Recommendations")
    print("="*80)
    
    min_train_samples = min(train_counts.values()) if train_counts else 0
    max_train_samples = max(train_counts.values()) if train_counts else 0
    
    if min_train_samples < 30:
        print(f"⚠️  WARNING: Smallest class has only {min_train_samples} training samples")
        print("   Recommendation: Aim for at least 50-100 samples per class")
        print("   Consider: Data augmentation, transfer learning, or class balancing")
    
    if max_train_samples > 3 * min_train_samples:
        print(f"⚠️  WARNING: Class imbalance detected (ratio {max_train_samples/max(1,min_train_samples):.1f}:1)")
        print("   Recommendation: Use --use_class_weights or oversample minority classes")
    
    if not leakage_found and min_train_samples >= 30:
        print("✓ Dataset appears properly configured for training")
    
    # Additional checks
    print("\n" + "="*80)
    print("Additional Checks")
    print("="*80)
    
    # Check if any classes are missing from train or test
    train_only = set(train_paths.keys()) - set(test_paths.keys())
    test_only = set(test_paths.keys()) - set(train_paths.keys())
    
    if train_only:
        print(f"⚠️  Classes only in TRAIN: {train_only}")
        print("   Model will be trained on these but never evaluated")
    
    if test_only:
        print(f"❌ Classes only in TEST: {test_only}")
        print("   Model cannot predict these classes (zero-shot problem)")
        print("   FIX: Ensure all test classes have training examples")
    
    if not train_only and not test_only:
        print("✓ Train and test sets have matching class labels")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
