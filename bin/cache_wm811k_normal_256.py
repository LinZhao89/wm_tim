"""Cache filtered 256x256 WM811K normal images for expensive baselines."""
import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from patchcore.datasets.wm811k import DatasetSplit, wm811kDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = Path(args.dataset_root)
    output = Path(args.output_dir)
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    dataset = wm811kDataset(
        str(root.parent), root.name, split=DatasetSplit.TRAIN,
        resize=256, imagesize=256, transform_mode="resize_pad",
        apply_filter=True, filter_window_size=3, filter_threshold=1.25,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)
    manifest = output / "manifest.csv"
    existing = set()
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as handle:
            existing = {row["source"] for row in csv.DictReader(handle)}
    rows = []
    for index, batch in enumerate(loader, 1):
        source = batch["image_path"][0]
        filename = hashlib.sha1(source.encode("utf-8")).hexdigest() + ".png"
        destination = images / filename
        if not destination.exists():
            array = (batch["raw_image"][0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(array).save(destination)
        if source not in existing:
            rows.append({"source": source, "cached_image": str(destination)})
        if index % 100 == 0:
            print(f"cached={index}/{len(dataset)}", flush=True)
    write_header = not manifest.exists()
    with manifest.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "cached_image"])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"CACHE_COMPLETE images={len(dataset)} dir={output}")


if __name__ == "__main__":
    main()
