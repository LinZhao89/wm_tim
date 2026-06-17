"""Classify detected anomalies into 8 classes using a trained classifier.

Requires a CSV with per-image anomaly scores (produced by run_patchcore.py when
--save_image_scores is set). Filters by a score threshold and classifies only
the anomalies above the threshold.

Usage (PowerShell):
  $env:PYTHONPATH='src'; python .\bin\classify_anomalies.py \
      C:\path\to\wm811k\prepare_dataset_train_ratio10p \
      --scores_csv C:\path\to\run\image_scores.csv \
      --classifier_weights C:\tmp\anom_cls.pth \
      --save_csv C:\path\to\run\anomaly_classifications.csv \
      --threshold 0.5
"""
import os
import sys

# Ensure src is in pythonpath and takes precedence over installed packages
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Force reload of patchcore if it was already loaded from site-packages
for key in list(sys.modules.keys()):
    if key.startswith("patchcore"):
        del sys.modules[key]

import csv
import json
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import click
import torch
import numpy as np
from scipy.ndimage import label, uniform_filter
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from typing import Tuple

from patchcore.classifier import AnomalyClassifier
from patchcore.datasets.wm811k import WM811K_MEAN, WM811K_STD
from patchcore.datasets.wm811k_cls import WM811KAnomalyClassDataset

LOGGER = logging.getLogger("anom_cls_infer")


def _build_transform(resize: int = 256, imagesize: int = 224):
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(imagesize),
        transforms.ToTensor(),
        transforms.Normalize(mean=WM811K_MEAN, std=WM811K_STD),
    ])


@click.command()
@click.argument("data_root", type=click.Path(exists=True, file_okay=False))
@click.option("--scores_csv", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--classifier_weights", multiple=True, type=click.Path(exists=True, dir_okay=False), required=True, help="One or more classifier checkpoints; provide multiple --classifier_weights or a comma-separated string once.")
@click.option("--save_csv", type=str, required=True)
@click.option("--metrics_path", type=str, default=None, help="Optional JSON file to store accuracy/precision/recall/F1 metrics.")
@click.option("--threshold", type=float, default=0.5, show_default=True)
@click.option("--tta", is_flag=True, help="Enable Test-Time Augmentation (flip ensemble)")
@click.option("--resize", type=int, default=256, show_default=True, help="Resize applied before center crop")
@click.option("--imagesize", type=int, default=224, show_default=True, help="Final crop size")
@click.option("--apply_filter/--no-apply_filter", is_flag=True, default=True, show_default=True, help="Apply constrained_mean_filter prior to transforms (match training).")
@click.option("--apply_filter_grayscale", is_flag=True, help="Apply constrained_mean_filter but return grayscale image (3-channel) instead of false-color.")
@click.option("--filter_window_size", type=int, default=3, show_default=True, help="Window size for constrained mean filter.")
@click.option("--filter_threshold", type=float, default=1.25, show_default=True, help="Threshold for constrained mean filter.")
@click.option("--refine_threshold", type=float, default=0.5, show_default=True, help="Threshold for max component ratio (MCR) to distinguish Near-full (>thresh) from Random (<=thresh).")
@click.option("--slcr_threshold", type=float, default=0.1, show_default=True, help="Threshold for second-largest component ratio (SLCR). Near-full requires SLCR < thresh.")
@click.option("--report_path", type=str, default=None, help="Path to save detailed classification report (CSV).")
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
def main(data_root: str, scores_csv: str, classifier_weights: Sequence[str], save_csv: str, threshold: float, tta: bool,
         resize: int, imagesize: int, apply_filter: bool, apply_filter_grayscale: bool, filter_window_size: int, filter_threshold: float, refine_threshold: float, slcr_threshold: float, report_path: Optional[str], gpu, metrics_path: Optional[str]):
    device = torch.device(f"cuda:{gpu[0]}" if torch.cuda.is_available() and gpu else "cpu")
    logging.basicConfig(level=logging.INFO)
    LOGGER.info(f"Using device: {device}")

    # Parse weights (support comma separated when provided once)
    if len(classifier_weights) == 1 and isinstance(classifier_weights[0], str) and "," in classifier_weights[0]:
        classifier_weights = tuple([w.strip() for w in classifier_weights[0].split(",") if w.strip()])

    # Load classifier ensemble
    ensemble = []
    classes = None
    use_density_global = False

    for w in classifier_weights:
        ckpt = torch.load(w, map_location=device)
        
        model = AnomalyClassifier(
            num_classes=ckpt.get("num_classes", 8), 
            model_name=ckpt.get("model_name", "tf_efficientnet_b3_ns")
        )
        model.load_state_dict(ckpt["state_dict"])  # type: ignore[index]
        if classes is None:
            classes = ckpt.get("classes", None)
        ensemble.append(model.to(device).eval())

    tfm = _build_transform(resize=resize, imagesize=imagesize)

    def constrained_mean_filter_grayscale(wbm: Image.Image, filter_window_size: int, threshold: float) -> Image.Image:
        """Apply constrained mean filter but return grayscale image (3-channel)."""
        gray_img_arr = np.array(wbm.convert("L"))

        # Vectorized thresholding into three semantic regions.
        class_map = np.ones_like(gray_img_arr, dtype=np.float32)
        class_map[gray_img_arr > 200] = 2  # bright anomalies/noise
        class_map[(gray_img_arr > 100) & (gray_img_arr <= 200)] = 0.5  # wafer surface
        class_map[gray_img_arr <= 100] = 0  # background

        mean_map = uniform_filter(class_map, size=filter_window_size, mode="constant")

        filtered_classes = class_map.copy()
        anomaly_mask = (class_map == 2) & (mean_map < threshold)
        filtered_classes[anomaly_mask] = 1  # treat as wafer after smoothing

        # Reconstruct grayscale image
        gray_out = np.zeros_like(gray_img_arr, dtype=np.uint8)
        
        # Use mean intensity from original image for each class to preserve look
        def _mean_gray(mask, default):
            if np.any(mask):
                return int(np.mean(gray_img_arr[mask]))
            return default

        # Original masks for color sampling
        mask_bg = class_map == 0
        mask_wafer = class_map == 0.5
        mask_anom = class_map == 2
        
        bg_val = _mean_gray(mask_bg, 0)
        wafer_val = _mean_gray(mask_wafer, 127)
        anom_val = _mean_gray(mask_anom, 255)

        # Apply to filtered classes
        # 0 -> Background
        # 0.5, 1 -> Wafer
        # 2 -> Anomaly
        
        gray_out[filtered_classes == 0] = bg_val
        gray_out[(filtered_classes == 0.5) | (filtered_classes == 1)] = wafer_val
        gray_out[filtered_classes == 2] = anom_val
        
        return Image.fromarray(gray_out).convert("RGB")

    def prepare_base_image(img: Image.Image) -> Image.Image:
        if apply_filter_grayscale:
            return constrained_mean_filter_grayscale(img, filter_window_size, filter_threshold)
        if apply_filter:
            return WM811KAnomalyClassDataset.constrained_mean_filter(
                img, 
                filter_window_size=filter_window_size, 
                threshold=filter_threshold
            )
        return img

    def refine_prediction(img: Image.Image, pred_label: str, prob: float) -> Tuple[str, float]:
        """Refine Random vs Near-full using connected components."""
        if pred_label not in ["Random", "Near-full"]:
            return pred_label, prob
            
        gray = np.array(img.convert("L"))
        anomaly_mask = gray > 200
        total_anomaly = np.sum(anomaly_mask)
        
        if total_anomaly == 0:
            return pred_label, prob
            
        labeled, n_components = label(anomaly_mask)
        if n_components == 0:
            return pred_label, prob
            
        counts = np.bincount(labeled.ravel())
        # counts[0] is background, so we need at least 2 bins to have a component
        if len(counts) < 2:
            return pred_label, prob
            
        # Get component sizes (excluding background) and sort descending
        component_sizes = np.sort(counts[1:])[::-1]
        
        max_size = component_sizes[0]
        max_ratio = float(max_size) / float(total_anomaly)
        
        second_max_size = component_sizes[1] if len(component_sizes) > 1 else 0
        slcr = float(second_max_size) / float(total_anomaly)
        
        # Heuristic:
        # Near-full: One massive blob dominates (MCR > thresh) AND second blob is tiny (SLCR < slcr_thresh)
        # Random: Many blobs, so SLCR is likely higher
        
        if max_ratio > refine_threshold and slcr < slcr_threshold:
            return "Near-full", prob
        else:
            return "Random", prob

    def tta_variants(img: Image.Image):
        base = prepare_base_image(img)
        if not tta:
            return [tfm(base)], base
        img_h = base.transpose(Image.FLIP_LEFT_RIGHT)
        img_v = base.transpose(Image.FLIP_TOP_BOTTOM)
        img_hv = img_h.transpose(Image.FLIP_TOP_BOTTOM)
        return [tfm(base), tfm(img_h), tfm(img_v), tfm(img_hv)], base

    # Read scores and classify images above threshold
    results: List[List[str]] = [["image", "score", "pred_class", "prob"]]
    y_true: List[str] = []
    y_pred: List[str] = []
    confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def resolve_image_path(img_rel: str) -> str:
        """Resolve image path whether absolute, relative, or already rooted."""
        if not img_rel:
            return img_rel
        candidates = []
        norm_input = os.path.normpath(img_rel)
        candidates.append(norm_input)
        # If already absolute but nested duplication, strip leading data_root once
        if norm_input.startswith(os.path.normpath(data_root)):
            trimmed = norm_input[len(os.path.normpath(data_root)):].lstrip(os.sep)
            if trimmed:
                candidates.append(os.path.join(data_root, trimmed))
        # Always try joining with data_root when not absolute
        if not os.path.isabs(norm_input):
            candidates.append(os.path.join(data_root, norm_input))
        # Deduplicate candidates while preserving order
        seen = set()
        for cand in candidates:
            cand_norm = os.path.normpath(cand)
            if cand_norm in seen:
                continue
            seen.add(cand_norm)
            if os.path.isfile(cand_norm):
                return cand_norm
        return norm_input
    def infer_label(img_path: str) -> str | None:
        rel = os.path.relpath(img_path, data_root)
        parts = [p for p in rel.replace("\\", "/").split("/") if p]
        candidate = parts[1] if len(parts) >= 2 else (parts[0] if parts else None)
        if classes:
            for part in reversed(parts):
                if part in classes:
                    candidate = part
                    break
        return candidate

    with open(scores_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_rel = row.get("image") or row.get("path")
            score = float(row.get("image_score_mean", row.get("score", 0)))
            if score < threshold:
                continue
            # Resolve absolute path relative to dataset root if needed
            img_path = resolve_image_path(img_rel)
            if not os.path.isfile(img_path):
                LOGGER.warning(f"Image not found, skipping: {img_path}")
                continue
            img = Image.open(img_path).convert("RGB")
            variants, base_img = tta_variants(img)
            xs = torch.stack(variants, dim=0).to(device)
            
            density_tensor = None
            if use_density_global:
                density_val = compute_density(base_img)
                density_tensor = torch.tensor([density_val] * xs.size(0), dtype=torch.float32, device=device)

            # Ensemble: average probabilities over TTA and models
            with torch.no_grad():
                probs_agg = None
                for m in ensemble:
                    logits = m(xs)
                    pr = torch.softmax(logits, dim=1).mean(dim=0)  # average over TTA
                    probs_agg = pr if probs_agg is None else (probs_agg + pr)
                probs_agg = probs_agg / len(ensemble)
                pred_idx = int(torch.argmax(probs_agg).item())
                pred_prob = float(probs_agg[pred_idx].item())
                pred_name = classes[pred_idx] if classes and 0 <= pred_idx < len(classes) else str(pred_idx)

            # Refine Random vs Near-full
            pred_name, pred_prob = refine_prediction(base_img, pred_name, pred_prob)

            results.append([img_path, f"{score:.6f}", pred_name, f"{pred_prob:.6f}"])
            gt_label = infer_label(img_path)
            if gt_label:
                y_true.append(gt_label)
                y_pred.append(pred_name)
                confusion[gt_label][pred_name] += 1

    os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)
    with open(save_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(results)
    LOGGER.info(f"Saved classifications to {save_csv} with {len(results)-1} rows >= threshold {threshold}")

    if y_true:
        labels = list(classes) if classes else sorted(set(y_true) | set(y_pred))
        metrics_report = _compute_metrics(confusion, labels)
        _log_metrics(metrics_report)
        
        # Save detailed CSV report if requested
        if report_path:
            _save_detailed_report(metrics_report, confusion, labels, report_path)
            LOGGER.info(f"Detailed classification report saved to {report_path}")

        if metrics_path:
            os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics_report, f, indent=2)
            LOGGER.info(f"Metrics saved to {metrics_path}")
    else:
        LOGGER.warning("No ground-truth labels could be inferred from image paths; metrics skipped.")


def _save_detailed_report(report: Dict[str, object], confusion: Dict[str, Dict[str, int]], labels: List[str], output_path: str):
    """Generates a detailed CSV report matching the user's requested format."""
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        
        # 1. Summary Table
        writer.writerow(["Classes", "Accuracy", "Precision", "Recall", "F1", "Support"])
        
        per_class = report["per_class"]
        total_samples = report["total_samples"]
        
        # Calculate per-class accuracy (One-vs-Rest)
        # Acc = (TP + TN) / Total
        # TN = Total - (TP + FP + FN)
        # TP = stats['tp']
        # FP = stats['fp']
        # FN = stats['fn']
        
        avg_acc = 0.0
        avg_prec = 0.0
        avg_rec = 0.0
        avg_f1 = 0.0
        
        for cls in labels:
            stats = per_class.get(cls, {})
            tp = stats.get("tp", 0)
            fp = stats.get("fp", 0)
            fn = stats.get("fn", 0)
            tn = total_samples - (tp + fp + fn)
            
            cls_acc = (tp + tn) / total_samples if total_samples > 0 else 0.0
            prec = stats.get("precision", 0.0)
            rec = stats.get("recall", 0.0)
            f1 = stats.get("f1", 0.0)
            support = stats.get("support", 0)
            
            writer.writerow([cls, f"{cls_acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", support])
            
            avg_acc += cls_acc
            avg_prec += prec
            avg_rec += rec
            avg_f1 += f1
            
        # Average Row
        n = len(labels)
        if n > 0:
            writer.writerow(["Macro Avg", f"{avg_acc/n:.4f}", f"{avg_prec/n:.4f}", f"{avg_rec/n:.4f}", f"{avg_f1/n:.4f}", total_samples])
            
            # Weighted Average Row
            w_prec = report.get("weighted_precision", 0.0)
            w_rec = report.get("weighted_recall", 0.0)
            w_f1 = report.get("weighted_f1", 0.0)
            # Weighted Accuracy is the same as overall Accuracy
            w_acc = report.get("accuracy", 0.0)
            
            writer.writerow(["Weighted Avg", f"{w_acc:.4f}", f"{w_prec:.4f}", f"{w_rec:.4f}", f"{w_f1:.4f}", total_samples])

            # Micro Average Row
            m_prec = report.get("micro_precision", 0.0)
            m_rec = report.get("micro_recall", 0.0)
            m_f1 = report.get("micro_f1", 0.0)
            m_acc = report.get("accuracy", 0.0)

            writer.writerow(["Micro Avg", f"{m_acc:.4f}", f"{m_prec:.4f}", f"{m_rec:.4f}", f"{m_f1:.4f}", total_samples])
        
        writer.writerow([])
        writer.writerow([])

        # Global Metrics
        writer.writerow(["Global Metrics"])
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Accuracy", f"{report['accuracy']:.4f}"])
        writer.writerow(["Macro Precision", f"{report['macro_precision']:.4f}"])
        writer.writerow(["Macro Recall", f"{report['macro_recall']:.4f}"])
        writer.writerow(["Macro F1", f"{report['macro_f1']:.4f}"])
        writer.writerow(["Weighted Precision", f"{report['weighted_precision']:.4f}"])
        writer.writerow(["Weighted Recall", f"{report['weighted_recall']:.4f}"])
        writer.writerow(["Weighted F1", f"{report['weighted_f1']:.4f}"])
        writer.writerow(["Micro Precision", f"{report['micro_precision']:.4f}"])
        writer.writerow(["Micro Recall", f"{report['micro_recall']:.4f}"])
        writer.writerow(["Micro F1", f"{report['micro_f1']:.4f}"])
        writer.writerow([])
        writer.writerow([])

        # Helper to get count
        def get_cnt(r, c):
            return confusion.get(r, {}).get(c, 0)

        # 2. Recall Confusion Matrix (rows=true, cols=pred) -> Row Normalized
        writer.writerow(["Recall Confusion Matrix (rows=true, cols=pred)"])
        writer.writerow(["Label"] + labels)
        for r in labels:
            row_sum = sum(get_cnt(r, c) for c in labels)
            row_data = [r]
            for c in labels:
                cnt = get_cnt(r, c)
                pct = (cnt / row_sum * 100) if row_sum > 0 else 0.0
                row_data.append(f"{cnt} ({pct:.2f}%)")
            writer.writerow(row_data)
            
        writer.writerow([])
        writer.writerow([])

        # 3. Precision Confusion Matrix (rows=true, cols=pred) -> Col Normalized
        writer.writerow(["Precision Confusion Matrix (rows=true, cols=pred)"])
        writer.writerow(["Label"] + labels)
        
        # Pre-calculate col sums
        col_sums = {c: sum(get_cnt(r, c) for r in labels) for c in labels}
        
        for r in labels:
            row_data = [r]
            for c in labels:
                cnt = get_cnt(r, c)
                c_sum = col_sums[c]
                pct = (cnt / c_sum * 100) if c_sum > 0 else 0.0
                row_data.append(f"{cnt} ({pct:.2f}%)")
            writer.writerow(row_data)

        writer.writerow([])
        writer.writerow([])

        # 4. Accuracy Confusion Matrix (rows=true, cols=pred) -> Diagonal has Class Accuracy
        writer.writerow(["Accuracy Confusion Matrix (rows=true, cols=pred)"])
        writer.writerow(["Label"] + labels)
        
        for r in labels:
            row_data = [r]
            for c in labels:
                cnt = get_cnt(r, c)
                if r == c:
                    # Diagonal: Show Class Accuracy
                    # Re-calculate class accuracy
                    stats = per_class.get(r, {})
                    tp = stats.get("tp", 0)
                    fp = stats.get("fp", 0)
                    fn = stats.get("fn", 0)
                    tn = total_samples - (tp + fp + fn)
                    cls_acc = (tp + tn) / total_samples if total_samples > 0 else 0.0
                    row_data.append(f"{cnt} ({cls_acc*100:.2f}%)")
                else:
                    row_data.append(str(cnt))
            writer.writerow(row_data)


def _compute_metrics(confusion: Dict[str, Dict[str, int]], labels: List[str]) -> Dict[str, object]:
    total = 0
    correct = 0
    per_class = {}
    macro_precisions = []
    macro_recalls = []
    macro_f1s = []

    for true_label in labels:
        row = confusion.get(true_label, {})
        support = sum(row.values())
        tp = row.get(true_label, 0)
        fp = sum(confusion[other].get(true_label, 0) for other in labels if other != true_label)
        fn = support - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[true_label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

        total += support
        correct += tp
        macro_precisions.append(precision)
        macro_recalls.append(recall)
        macro_f1s.append(f1)

    accuracy = correct / total if total > 0 else 0.0
    macro_precision = sum(macro_precisions) / len(labels) if labels else 0.0
    macro_recall = sum(macro_recalls) / len(labels) if labels else 0.0
    macro_f1 = sum(macro_f1s) / len(labels) if labels else 0.0

    # Weighted metrics
    weighted_precision = sum(per_class[l]["precision"] * per_class[l]["support"] for l in labels) / total if total > 0 else 0.0
    weighted_recall = sum(per_class[l]["recall"] * per_class[l]["support"] for l in labels) / total if total > 0 else 0.0
    weighted_f1 = sum(per_class[l]["f1"] * per_class[l]["support"] for l in labels) / total if total > 0 else 0.0

    # Micro metrics (Accuracy)
    micro_precision = accuracy
    micro_recall = accuracy
    micro_f1 = accuracy

    # Convert defaultdict tree into regular dict for serialization
    confusion_matrix = {t: dict(preds) for t, preds in confusion.items()}

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "total_samples": total,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix,
    }


def _log_metrics(report: Dict[str, object]) -> None:
    LOGGER.info("Classification metrics on %s labelled samples", report["total_samples"])
    LOGGER.info("  Accuracy       : %.4f", report["accuracy"])
    LOGGER.info("  Macro Prec.    : %.4f", report["macro_precision"])
    LOGGER.info("  Macro Recall   : %.4f", report["macro_recall"])
    LOGGER.info("  Macro F1       : %.4f", report["macro_f1"])
    LOGGER.info("  Weighted Prec. : %.4f", report["weighted_precision"])
    LOGGER.info("  Weighted Recall: %.4f", report["weighted_recall"])
    LOGGER.info("  Weighted F1    : %.4f", report["weighted_f1"])
    LOGGER.info("  Micro Prec.    : %.4f", report["micro_precision"])
    LOGGER.info("  Micro Recall   : %.4f", report["micro_recall"])
    LOGGER.info("  Micro F1       : %.4f", report["micro_f1"])
    LOGGER.info("  Per-class breakdown:")
    for cls, stats in report["per_class"].items():
        LOGGER.info(
            "    %-10s P=%.4f R=%.4f F1=%.4f support=%d",
            cls,
            stats["precision"],
            stats["recall"],
            stats["f1"],
            stats["support"],
        )


if __name__ == "__main__":
    main()
