"""Evaluate a trained EfficientAD checkpoint without retraining."""
import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from patchcore.datasets.wm811k import DatasetSplit, wm811kDataset
from patchcore.datasets.synthetic_masks import DatasetSplit as SyntheticSplit, SyntheticMaskPatchCoreDataset
from run_efficientad import CachedNormals, auc, evaluate


def sampled_values(values, limit=2048):
    values = values.flatten()
    if values.numel() <= limit:
        return values
    return values[torch.randperm(values.numel(), device=values.device)[:limit]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--synthetic_root", required=True)
    parser.add_argument("--cached_train_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    device = torch.device("cuda:0")
    module_path = ROOT / "wm_baseline_env/Lib/site-packages/anomalib/models/image/efficient_ad/torch_model.py"
    spec = importlib.util.spec_from_file_location("efficientad_torch", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.EfficientAdModel(384, module.EfficientAdModelSize.S, padding=False, pad_maps=True).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    values_st, values_ae = [], []
    cached_loader = DataLoader(CachedNormals(args.cached_train_dir), batch_size=2, shuffle=False, num_workers=0)
    with torch.no_grad():
        for index, batch in enumerate(cached_loader, 1):
            output = model(batch["raw_image"].to(device), normalize=False)
            values_st.append(sampled_values(output["map_st"]))
            values_ae.append(sampled_values(output["map_ae"]))
            if index % 500 == 0:
                print(f"quantile_batch={index}/{len(cached_loader)}", flush=True)
    values_st = torch.cat(values_st)
    values_ae = torch.cat(values_ae)
    model.quantiles["qa_st"].data = torch.quantile(values_st, 0.9)
    model.quantiles["qb_st"].data = torch.quantile(values_st, 0.995)
    model.quantiles["qa_ae"].data = torch.quantile(values_ae, 0.9)
    model.quantiles["qb_ae"].data = torch.quantile(values_ae, 0.995)
    root = Path(args.dataset_root)
    common = dict(resize=256, imagesize=256, transform_mode="resize_pad", apply_filter=True, filter_window_size=3, filter_threshold=1.25)
    test = wm811kDataset(str(root.parent), root.name, split=DatasetSplit.TEST, **common)
    pixel = SyntheticMaskPatchCoreDataset(args.synthetic_root, "all", split=SyntheticSplit.TEST, **common)
    scores, labels, _, _ = evaluate(model, DataLoader(test, batch_size=2, shuffle=False, num_workers=0), device)
    _, _, maps, masks = evaluate(model, DataLoader(pixel, batch_size=2, shuffle=False, num_workers=0), device)
    anomaly = np.asarray(masks).reshape(len(masks), -1).sum(axis=1) > 0
    from sklearn.metrics import roc_auc_score
    result = {"instance_auroc": float(roc_auc_score(labels, scores)), "full_pixel_auroc": auc(maps, masks), "anomaly_pixel_auroc": auc(maps[anomaly], masks[anomaly])}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result)
        writer.writeheader(); writer.writerow(result)
    print(result)


if __name__ == "__main__":
    main()
