"""Export WM811K wafermap images after constrained mean filtering.

Example (PowerShell):
    $env:PYTHONPATH='src'; python .\bin\export_filtered_wm811k.py \
        dataset/wm811k prepare_dataset_train_ratio10p exports/filtered \
        --split train --split test --imagesize 128 --resize 128
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import click
from PIL import Image
from torchvision import transforms

# Ensure local imports resolve when running as a script.
sys.path.insert(0, "src")

from patchcore.datasets.wm811k import DatasetSplit, wm811kDataset  # noqa: E402

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def _iter_images(split_root: Path) -> Iterable[Path]:
    for path in sorted(split_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            yield path


def _build_transform(resize: int, imagesize: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(imagesize),
    ])


def _process_image(
    image_path: Path,
    window_size: int,
    threshold: float,
    pil_transform: Optional[transforms.Compose],
) -> Image.Image:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
    filtered = wm811kDataset.constrained_mean_filter(
        rgb,
        filter_window_size=window_size,
        threshold=threshold,
    )
    if pil_transform is not None:
        filtered = pil_transform(filtered)
    return filtered


def _export_split(
    dataset_root: Path,
    split: DatasetSplit,
    output_dir: Path,
    window_size: int,
    threshold: float,
    transform: Optional[transforms.Compose],
    limit_per_class: Optional[int],
    overwrite: bool,
) -> Tuple[int, int]:
    split_root = dataset_root / split.value
    if not split_root.exists():
        raise click.UsageError(f"Split directory not found: {split_root}")

    written, skipped = 0, 0
    for anomaly_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
        exported_for_class = 0
        for image_path in _iter_images(anomaly_dir):
            if limit_per_class is not None and exported_for_class >= limit_per_class:
                break

            rel_path = image_path.relative_to(split_root)
            destination = output_dir / split.value / rel_path
            destination.parent.mkdir(parents=True, exist_ok=True)

            if destination.exists() and not overwrite:
                skipped += 1
                continue

            processed = _process_image(
                image_path,
                window_size=window_size,
                threshold=threshold,
                pil_transform=transform,
            )
            processed.save(destination)
            exported_for_class += 1
            written += 1

    return written, skipped


@click.command()
@click.argument("data_path", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.argument("subdataset", type=str)
@click.argument("output_dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path))
@click.option(
    "--split",
    "splits",
    type=click.Choice([DatasetSplit.TRAIN.value, DatasetSplit.TEST.value], case_sensitive=False),
    multiple=True,
    default=(DatasetSplit.TRAIN.value, DatasetSplit.TEST.value),
    show_default=True,
    help="Dataset splits to export.",
)
@click.option("--resize", default=256, show_default=True, help="Resize before saving (match training).")
@click.option("--imagesize", default=224, show_default=True, help="Center crop size before saving.")
@click.option(
    "--match-training-shape/--keep-original-size",
    default=True,
    show_default=True,
    help="Apply the same resize+crop used during training before saving.",
)
@click.option("--filter-window-size", default=3, show_default=True, help="Constrained mean filter window size.")
@click.option("--filter-threshold", default=1.25, show_default=True, help="Constrained mean threshold.")
@click.option(
    "--limit-per-class",
    default=None,
    type=int,
    help="Optional cap on the number of images exported for each class.",
)
@click.option("--overwrite", is_flag=True, help="Overwrite files if they already exist.")
def main(
    data_path: Path,
    subdataset: str,
    output_dir: Path,
    splits: Tuple[str, ...],
    resize: int,
    imagesize: int,
    match_training_shape: bool,
    filter_window_size: int,
    filter_threshold: float,
    limit_per_class: Optional[int],
    overwrite: bool,
) -> None:
    """Write filtered WM811K images for visual inspection."""

    dataset_root = data_path / subdataset
    if not dataset_root.exists():
        raise click.UsageError(f"Dataset '{subdataset}' not found inside {data_path}")

    output_subdir = output_dir / subdataset
    output_subdir.mkdir(parents=True, exist_ok=True)

    transform = _build_transform(resize, imagesize) if match_training_shape else None

    # Deduplicate splits while preserving order
    requested_splits = []
    seen = set()
    for split in splits:
        split_lower = split.lower()
        if split_lower not in seen:
            seen.add(split_lower)
            requested_splits.append(split_lower)

    if not requested_splits:
        requested_splits = [DatasetSplit.TRAIN.value, DatasetSplit.TEST.value]

    summary = []
    for split_name in requested_splits:
        ds_split = DatasetSplit(split_name)
        written, skipped = _export_split(
            dataset_root=dataset_root,
            split=ds_split,
            output_dir=output_subdir,
            window_size=filter_window_size,
            threshold=filter_threshold,
            transform=transform,
            limit_per_class=limit_per_class,
            overwrite=overwrite,
        )
        summary.append((split_name, written, skipped))

    click.echo("Filtered image export complete:\n")
    for split_name, written, skipped in summary:
        click.echo(f"  {split_name}: wrote {written} files" + (f", skipped {skipped}" if skipped else ""))


if __name__ == "__main__":
    main()
