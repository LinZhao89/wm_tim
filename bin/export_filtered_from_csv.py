"""Export constrained-mean-filtered WM811K images selected from a CSV of anomaly scores.

Example (PowerShell):
    $env:PYTHONPATH='src'; python .\bin\export_filtered_from_csv.py \
        results\anomaly_scores.csv exports\low_scores \
        --base-path . --score-threshold 0.3 --resize 256 --imagesize 224
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Optional

import click
from PIL import Image
from torchvision import transforms

# Ensure local imports resolve when running as a script.
sys.path.insert(0, "src")

from patchcore.datasets.wm811k import wm811kDataset  # noqa: E402


def _build_transform(resize: int, imagesize: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(imagesize),
    ])


def _apply_filter(
    image_path: Path,
    filter_window_size: int,
    filter_threshold: float,
    pil_transform: Optional[transforms.Compose],
) -> Image.Image:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
    filtered = wm811kDataset.constrained_mean_filter(
        rgb,
        filter_window_size=filter_window_size,
        threshold=filter_threshold,
    )
    if pil_transform is not None:
        filtered = pil_transform(filtered)
    return filtered


@click.command()
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path))
@click.option("--base-path", type=click.Path(path_type=Path), default=Path("."), show_default=True,
              help="Base directory to resolve relative image paths from the CSV.")
@click.option("--score-threshold", type=float, default=0.3, show_default=True,
              help="Only export rows whose anomaly score is <= this value.")
@click.option("--filter-window-size", type=int, default=3, show_default=True,
              help="Constrained mean filter window size.")
@click.option("--filter-threshold", type=float, default=1.25, show_default=True,
              help="Constrained mean filter threshold.")
@click.option("--resize", type=int, default=256, show_default=True,
              help="Resize before saving (match training).")
@click.option("--imagesize", type=int, default=224, show_default=True,
              help="Center crop size before saving.")
@click.option("--match-training-shape/--keep-original-size", default=True, show_default=True,
              help="Apply the same resize+crop used during training before saving.")
@click.option("--limit", type=int, default=None,
              help="Optional cap on number of images to export (after filtering by score).")
@click.option("--overwrite", is_flag=True, help="Overwrite files if they already exist.")
def main(
    csv_path: Path,
    output_dir: Path,
    base_path: Path,
    score_threshold: float,
    filter_window_size: int,
    filter_threshold: float,
    resize: int,
    imagesize: int,
    match_training_shape: bool,
    limit: Optional[int],
    overwrite: bool,
) -> None:
    """Export filtered images for rows whose score <= threshold."""

    output_dir.mkdir(parents=True, exist_ok=True)
    transform = _build_transform(resize, imagesize) if match_training_shape else None

    written = 0
    skipped = 0
    selected = 0

    with csv_path.open("r", newline="", encoding="utf-8") as cf:
        reader = csv.DictReader(cf)
        if "image" not in reader.fieldnames or "score" not in reader.fieldnames:
            raise click.UsageError("CSV must contain 'image' and 'score' columns.")
        for row in reader:
            try:
                score = float(row["score"])
            except (TypeError, ValueError):
                continue
            if score > score_threshold:
                continue
            selected += 1
            if limit is not None and written >= limit:
                break

            rel_image = Path(row["image"])
            image_path = rel_image if rel_image.is_absolute() else (base_path / rel_image)
            if not image_path.exists():
                skipped += 1
                continue

            destination = output_dir / rel_image
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and not overwrite:
                skipped += 1
                continue

            filtered = _apply_filter(
                image_path,
                filter_window_size=filter_window_size,
                filter_threshold=filter_threshold,
                pil_transform=transform,
            )
            filtered.save(destination)
            written += 1

    click.echo(
        f"Export complete: selected {selected} rows (score <= {score_threshold}), "
        f"wrote {written} files", nl=False
    )
    if skipped:
        click.echo(f", skipped {skipped} (missing or existing)")
    else:
        click.echo()


if __name__ == "__main__":
    main()
