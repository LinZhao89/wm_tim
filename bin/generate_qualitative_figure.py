"""Render real and synthetic qualitative PatchCore localization examples."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import patchcore.common
from patchcore.datasets.synthetic_masks import SyntheticMaskPatchCoreDataset, DatasetSplit as SyntheticSplit
from patchcore.datasets.wm811k import wm811kDataset, DatasetSplit
from patchcore.geometry_patchcore import PatchCore


ROOT = Path(r"E:\lin\wm_tim")
MODEL_DIR = ROOT / "results" / "wm811k_resolution_256" / "geometry_old_cbam_256_filter_w3_t1p25_p001_n500_scoreonly_20260623" / "models" / "wm811k_prepare_dataset_original"
OUT = ROOT / "paper" / "qualitative_results.png"


def choose(dataset, names):
    chosen = []
    wanted = set(names)
    for index, record in enumerate(dataset.data_to_iterate):
        if record[1] in wanted and all(item[0] != record[1] for item in chosen):
            chosen.append((record[1], index))
    return chosen


def image_from_tensor(tensor):
    return np.clip(tensor.detach().cpu().permute(1, 2, 0).numpy(), 0, 1)


def score_map(core, sample):
    with torch.no_grad():
        _, maps = core._predict(
            sample["image"].unsqueeze(0), sample["raw_image"].unsqueeze(0))
    value = np.asarray(maps[0], dtype=np.float32)
    maximum = value.max()
    return value / maximum if maximum > 0 else value


def draw_map(axis, array, title, cmap="magma"):
    axis.imshow(array, cmap=cmap, vmin=0, vmax=1)
    axis.set_title(title, fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    core = PatchCore(device)
    core.load_from_path(str(MODEL_DIR), device, patchcore.common.FaissNN(False, 4))
    core.eval()

    real = wm811kDataset(
        str(ROOT / "dataset" / "wm811k"), "prepare_dataset_original",
        resize=256, imagesize=256, split=DatasetSplit.TEST,
        transform_mode="resize_pad", apply_filter=True,
        filter_window_size=3, filter_threshold=1.25)
    synthetic = SyntheticMaskPatchCoreDataset(
        str(ROOT / "dataset" / "wm811k" / "synthetic_formula"), classname="all",
        resize=256, imagesize=256, split=SyntheticSplit.TEST,
        transform_mode="resize_pad", apply_filter=True,
        filter_window_size=3, filter_threshold=1.25)

    real_items = choose(real, ["good", "Donut", "Edge-Ring", "Scratch"])
    synthetic_items = choose(synthetic, ["Center", "Edge-Loc", "Near-full", "Scratch"])
    if len(real_items) != 4 or len(synthetic_items) != 4:
        raise RuntimeError("Could not find all requested qualitative categories.")

    figure, axes = plt.subplots(2, 12, figsize=(18, 5.4), constrained_layout=True)
    for axis in axes.ravel():
        axis.axis("off")

    for column, (category, index) in enumerate(real_items):
        sample = real[index]
        start = column * 3
        image = image_from_tensor(sample["raw_image"])
        anomaly = score_map(core, sample)
        draw_map(axes[0, start], image, "Input")
        draw_map(axes[0, start + 1], anomaly, "Prediction")
        axes[0, start + 1].set_title(f"{category}\nPrediction", fontsize=9, fontweight="bold", pad=8)
        overlay = image.copy()
        axes[0, start + 2].imshow(overlay)
        axes[0, start + 2].imshow(anomaly, cmap="magma", alpha=0.55, vmin=0, vmax=1)
        axes[0, start + 2].set_title("Overlay", fontsize=9)
        axes[0, start + 2].set_xticks([])
        axes[0, start + 2].set_yticks([])
    for column, (category, index) in enumerate(synthetic_items):
        sample = synthetic[index]
        start = column * 3
        image = image_from_tensor(sample["raw_image"])
        mask = sample["mask"][0].numpy()
        anomaly = score_map(core, sample)
        draw_map(axes[1, start], image, "Input")
        draw_map(axes[1, start + 1], mask, "Exact mask", cmap="gray")
        axes[1, start + 1].set_title(f"{category}\nExact mask", fontsize=9, fontweight="bold", pad=8)
        draw_map(axes[1, start + 2], anomaly, "Prediction")
    figure.text(0.01, 0.985, "(a) Real test maps", ha="left", va="top", fontsize=12, fontweight="bold")
    figure.text(0.01, 0.49, "(b) Synthetic validation maps", ha="left", va="top", fontsize=12, fontweight="bold")

    figure.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()
