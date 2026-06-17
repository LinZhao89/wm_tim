import contextlib
import logging
import os
import sys
import csv

import click
import numpy as np
from scipy.ndimage import label, binary_erosion, generate_binary_structure
import torch
import json
import pandas as pd
from pathlib import PurePath
from torchvision import transforms
from PIL import Image

import patchcore.backbones
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.geometry_patchcore
import patchcore.sampler
import patchcore.utils
from patchcore.datasets.wm811k_cls import WM811KAnomalyClassDataset
from datetime import datetime

LOGGER = logging.getLogger(__name__)

_DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"],
        "wm811k":["patchcore.datasets.wm811k","wm811kDataset"],
        "synthetic_masks": [
            "patchcore.datasets.synthetic_masks",
            "SyntheticMaskPatchCoreDataset",
        ]}


def _infer_label_from_path(image_path, classes=None):
    parts = PurePath(image_path).parts
    label = None
    if "test" in parts:
        idx = parts.index("test")
        if idx + 1 < len(parts):
            label = parts[idx + 1]
    if label is None and len(parts) >= 2:
        label = parts[-2]
    if classes and label not in classes:
        for part in reversed(parts):
            if part in classes:
                label = part
                break
    return label


def _compute_classification_metrics(y_true, y_pred, labels):
    confusion = {lbl: {p: 0 for p in labels} for lbl in labels}
    for true, pred in zip(y_true, y_pred):
        if true in confusion and pred in confusion[true]:
            confusion[true][pred] += 1

    total = sum(sum(row.values()) for row in confusion.values())
    correct = sum(confusion[label][label] for label in labels)
    per_class = {}
    macro_precisions = []
    macro_recalls = []
    macro_f1s = []

    for label in labels:
        row = confusion[label]
        support = sum(row.values())
        tp = row[label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = support - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        macro_precisions.append(precision)
        macro_recalls.append(recall)
        macro_f1s.append(f1)

    accuracy = correct / total if total > 0 else 0.0
    macro_precision = sum(macro_precisions) / len(labels) if labels else 0.0
    macro_recall = sum(macro_recalls) / len(labels) if labels else 0.0
    macro_f1 = sum(macro_f1s) / len(labels) if labels else 0.0

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "total_samples": total,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def _log_classification_metrics(report):
    if not report["total_samples"]:
        LOGGER.info("Classification metrics: no labeled samples available.")
        return
    LOGGER.info("Classification metrics on %s samples", report["total_samples"])
    LOGGER.info("  Accuracy     : %.4f", report["accuracy"])
    LOGGER.info("  Macro Prec.  : %.4f", report["macro_precision"])
    LOGGER.info("  Macro Recall : %.4f", report["macro_recall"])
    LOGGER.info("  Macro F1     : %.4f", report["macro_f1"])
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


def _write_classification_report_csv(path, report, labels):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as rf:
        writer = csv.writer(rf)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for label in labels:
            stats = report["per_class"].get(label, {})
            writer.writerow([
                label,
                f"{stats.get('precision', 0.0):.4f}",
                f"{stats.get('recall', 0.0):.4f}",
                f"{stats.get('f1', 0.0):.4f}",
                int(stats.get("support", 0)),
            ])
        writer.writerow([])
        writer.writerow(["Confusion Matrix (rows=true, cols=pred)"])
        writer.writerow(["true\\pred"] + list(labels))
        for label in labels:
            row = [label]
            for pred in labels:
                row.append(report["confusion_matrix"].get(label, {}).get(pred, 0))
            writer.writerow(row)


@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--save_segmentation_images", is_flag=True, default=False, help="Set to true if need visualization")
@click.option("--save_patchcore_model", is_flag=True)
@click.option("--save_image_scores", is_flag=True)
@click.option("--image_scores_path", type=str, default="", help="Path to save per-image scores CSV (overrides default run folder)")
@click.option("--classify_anomalies", is_flag=True, help="Run anomaly classification after detection.")
@click.option("--classifier_weights", type=str, default="", help="Path to trained anomaly classifier weights (.pth)")
@click.option("--classification_threshold", type=float, default=-1.0, help="Score threshold for selecting images to classify; if <0, an automatic boundary is used.")
@click.option("--classification_save_csv", type=str, default="", help="Where to save anomaly classification CSV (defaults under run folder)")
@click.option("--classification_report_csv", type=str, default="", help="Optional: write per-class metrics (precision, recall, support) and confusion matrix.")
@click.option("--classification_metrics_path", type=str, default="", help="Optional JSON file to store classification metrics summary.")
@click.option("--classification_ensemble_weights", type=str, default="", help="Comma-separated list of additional classifier checkpoints for ensemble averaging.")
@click.option("--classification_tta", is_flag=True, help="Enable flip-based Test-Time Augmentation for classification stage.")
@click.option("--classification_resize", type=int, default=None, help="Resize before crop for classification stage (defaults to dataset resize).")
@click.option("--classification_imagesize", type=int, default=None, help="Final crop size for classification stage (defaults to dataset crop size).")
@click.option("--classification_filter_window_size", type=int, default=None, help="Override constrained mean filter window size for classifier inputs (defaults to dataset value).")
@click.option("--classification_filter_threshold", type=float, default=None, help="Override constrained mean filter threshold for classifier inputs (defaults to dataset value).")
@click.option("--classification_apply_filter/--no-classification_apply_filter", is_flag=True, default=True, show_default=True,
              help="Apply constrained_mean_filter to classifier inputs (must match training).")
def main(**kwargs):
    pass


@main.result_callback()
def run(
    methods,
    results_path,
    gpu,
    seed,
    log_group,
    log_project,
    save_segmentation_images,
    save_patchcore_model,
    save_image_scores,
    image_scores_path,
    classify_anomalies,
    classifier_weights,
    classification_threshold,
    classification_save_csv,
    classification_report_csv,
    classification_metrics_path,
    classification_ensemble_weights,
    classification_tta,
    classification_resize,
    classification_imagesize,
    classification_filter_window_size,
    classification_filter_threshold,
    classification_apply_filter,
):
    if classify_anomalies:
        raise click.UsageError(
            "Classification is archived in this repository; use detection/localization only."
        )
    ct = datetime.now()
    print("current time: ", ct)
    methods = {key: item for (key, item) in methods}
    print(methods)
    run_save_path = patchcore.utils.create_storage_folder(
        results_path, log_project, log_group, mode="iterate"
    )

    list_of_dataloaders = methods["get_dataloaders"](seed)

    device = patchcore.utils.set_torch_device(gpu)
    # Device context here is specifically set and used later
    # because there was GPU memory-bleeding which I could only fix with
    # context managers.
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []
    image_scores ={} # to save scores for each test image  
    
    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )

        patchcore.utils.fix_seeds(seed, device)

        dataset_name = dataloaders["training"].name

        with device_context:
            torch.cuda.empty_cache()
            imagesize = dataloaders["training"].dataset.imagesize
            sampler = methods["get_sampler"](
                device,
            )
            #print(f'what is sampler:{sampler}')
            print(f'what is imagesize:{imagesize}')
            
            PatchCore_list = methods["get_patchcore"](imagesize, sampler, device)
            print(f'what is len PatchCore_list:{len(PatchCore_list)}')
            if len(PatchCore_list) > 1:
                LOGGER.info(
                    "Utilizing PatchCore Ensemble (N={}).".format(len(PatchCore_list))
                )
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                if PatchCore.backbone.seed is not None:
                    patchcore.utils.fix_seeds(PatchCore.backbone.seed, device)
                LOGGER.info(
                    "Training models ({}/{})".format(i + 1, len(PatchCore_list))
                )
                torch.cuda.empty_cache()
                PatchCore.fit(dataloaders["training"])

            torch.cuda.empty_cache()
            aggregator = {"scores": [], "segmentations": []}
            
            image_names = [x[2] for x in dataloaders["testing"].dataset.data_to_iterate]

            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(PatchCore_list)
                    )
                )
                scores, segmentations, labels_gt, masks_gt = PatchCore.predict(
                    dataloaders["testing"]
                )
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)
             
            #print(f'run patchcore mask_gt: {len(masks_gt)}')
            #print(f'run patchcore label_gt: {len(labels_gt)}')
            scores = np.array(aggregator["scores"])
            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - min_scores) / (max_scores - min_scores)
           
            scores_mean = np.mean(scores, axis=0)
            #print("mean: ",scores_mean)
            #scores_max = np.max(scores, axis=0)
            #print("scores_max:", scores_max)
            #scores_min = np.min(scores, axis=0)
            #print("scores_min:", scores_min)
            #scores_median = np.median(scores, axis=0)
            #print("scores_median:", scores_median)
            """
            ## To output score value for each image, 
            for image_name, score in zip(image_names, scores):
                if image_name in image_scores: 
                    image_scores[image_name].append(str(score))
                else: 
                    image_scores[image_name] = [str(score)] 
            """
            scores_lst = list(scores_mean)
            #scores_lst1 = list(scores_min)
            #scores_lst2 = list(scores_median)
            #scores_lst3 = list(scores_max)
            #image_score_df = pd.DataFrame({'image':image_names, 'image_score_mean':scores_lst,'image_score_min':scores_lst1,'image_score_median':scores_lst2 ,'image_score_max':scores_lst3})
            image_score_df = pd.DataFrame({'image':image_names, 'image_score_mean':scores_lst})

            #print("image_score_df", image_score_df)
            #print(image_score_df.to_string())
            
            scores_lst.sort(reverse=True)

            boundary_idx = int(0.05*len(scores_lst))

            boundary_score = scores_lst[boundary_idx]
            print("boundary_score:", boundary_score)            

            segmentations = np.array(aggregator["segmentations"])
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            segmentations = (segmentations - min_scores) / (max_scores - min_scores)
            segmentations = np.mean(segmentations, axis=0)

            anomaly_labels = [
                x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
            ]

            # (Optional) Plot example images.
            if save_segmentation_images:
                #dataloaders["testing"].dataset.transform_std = [0.229, 0.224, 0.225]
                #dataloaders["testing"].dataset.transform_mean = [0.485, 0.456, 0.406]
                image_paths = [
                    x[2] for x in dataloaders["testing"].dataset.data_to_iterate
                ]
                mask_paths = [
                    x[3] for x in dataloaders["testing"].dataset.data_to_iterate
                ]

                def image_transform(image):
                    dataloaders["testing"].dataset.transform_std = [0.229, 0.224, 0.225]
                    dataloaders["testing"].dataset.transform_mean = [0.485, 0.456, 0.406]
                    in_std = np.array(dataloaders["testing"].dataset.transform_std ).reshape(-1, 1, 1)
                    in_mean = np.array(dataloaders["testing"].dataset.transform_mean ).reshape(-1, 1, 1)
                    image = dataloaders["testing"].dataset.transform_img(image)
                    return np.clip(
                        (image.numpy() * in_std + in_mean) * 255, 0, 255
                    ).astype(np.uint8)

                def mask_transform(mask):
                    return dataloaders["testing"].dataset.transform_mask(mask).numpy()

                image_save_path = os.path.join(
                    run_save_path, "segmentation_images", dataset_name
                )
                print(f'save_path:{image_save_path}')
                os.makedirs(image_save_path, exist_ok=True)
                patchcore.utils.plot_segmentation_images(
                    image_save_path,
                    image_paths,
                    segmentations,
                    scores_mean,
                    mask_paths,
                    image_transform=image_transform,
                    mask_transform=mask_transform,
                )

            LOGGER.info("Computing evaluation metrics.")
            auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
                scores_mean, anomaly_labels
            )["auroc"]

            # Compute PRO score & PW Auroc for all images
            print(f'run patchcore mask_gt: {len(masks_gt)}')

            # Safe compute helper
            def safe_compute_pixel_auroc(preds, truths):
                # Check provided ground truth for at least 2 classes (0 and 1)
                # Truths is a list of numpy arrays (masks)
                has_bg = False
                has_fg = False
                
                for t in truths:
                    t_max = np.max(t)
                    t_min = np.min(t)
                    if t_max > 0: has_fg = True
                    if t_min == 0: has_bg = True
                    
                    if has_fg and has_bg:
                        break
                        
                if not (has_fg and has_bg):
                    return {"auroc": 0.5} # Undefined, return default
                    
                return patchcore.metrics.compute_pixelwise_retrieval_metrics(preds, truths)

            pixel_scores = safe_compute_pixel_auroc(segmentations, masks_gt)
            full_pixel_auroc = pixel_scores["auroc"]
            
            # add in visualization to output images 
            # Save anomaly_segmentations
            #anomaly_img = Image.fromarray(segmentations)
            #anomaly_img.save("anomaly_segmentations1.png")

            # Save ground_truth_masks
            #ground_truth_img = Image.fromarray(masks_gt)
            #ground_truth_img.save("ground_truth_masks1.png")




            # Compute PRO score & PW Auroc only images with anomalies
            sel_idxs = []
            for i in range(len(masks_gt)):
                if np.sum(masks_gt[i]) > 0:
                    sel_idxs.append(i)
            
            if len(sel_idxs) > 0:
                pixel_scores = safe_compute_pixel_auroc(
                    [segmentations[i] for i in sel_idxs],
                    [masks_gt[i] for i in sel_idxs],
                )
                anomaly_pixel_auroc = pixel_scores["auroc"]
            else:
                anomaly_pixel_auroc = 0.5

            result_collect.append(
                {
                    "dataset_name": dataset_name,
                    "instance_auroc": auroc,
                    "full_pixel_auroc": full_pixel_auroc,
                    "anomaly_pixel_auroc": anomaly_pixel_auroc,
                }
            )

            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{0}: {1:3.3f}".format(key, item))

            # (Optional) Store PatchCore model for later re-use.
            # SAVE all patchcores only if mean_threshold is passed?
            if save_patchcore_model:
                patchcore_save_path = os.path.join(
                    run_save_path, "models", dataset_name
                )
                os.makedirs(patchcore_save_path, exist_ok=True)
                for i, PatchCore in enumerate(PatchCore_list):
                    prepend = (
                        "Ensemble-{}-{}_".format(i + 1, len(PatchCore_list))
                        if len(PatchCore_list) > 1
                        else ""
                    )
                    PatchCore.save_to_path(patchcore_save_path, prepend)
            # Save per-image scores CSV (if requested)
            if save_image_scores:
                if image_scores_path:
                    out_path = image_scores_path
                else:
                    out_path = os.path.join(run_save_path, f"image_scores_{dataset_name}.csv")
                os.makedirs(os.path.dirname(out_path) or run_save_path, exist_ok=True)
                image_score_df.to_csv(out_path, index=False)

            # Optionally run anomaly classification using trained classifier
            if classify_anomalies:
                if not classifier_weights or not os.path.isfile(classifier_weights):
                    LOGGER.error("--classify_anomalies set but --classifier_weights not provided or file missing.")
                else:
                    try:
                        from patchcore.classifier import AnomalyClassifier
                        from patchcore.datasets.wm811k import WM811K_MEAN, WM811K_STD

                        # Build ensemble list (primary weight + optional extra checkpoints)
                        weight_list = [classifier_weights]
                        if classification_ensemble_weights:
                            extra = [w.strip() for w in classification_ensemble_weights.split(",") if w.strip()]
                            weight_list.extend(extra)
                        ensemble = []
                        classes = None
                        use_density_global = False
                        for w in weight_list:
                            ckpt = torch.load(w, map_location=device)
                            model_name = ckpt.get("model_name", "tf_efficientnet_b3_ns")
                            num_classes = ckpt.get("num_classes", 8)
                            use_density = ckpt.get("use_density", False)
                            if len(ensemble) == 0:
                                use_density_global = use_density
                            elif use_density != use_density_global:
                                LOGGER.warning("Ensemble models have mixed use_density settings! This might fail.")
                            if classes is None:
                                classes = ckpt.get("classes", None)
                            clf = AnomalyClassifier(num_classes=num_classes, model_name=model_name, use_density=use_density)
                            clf.load_state_dict(ckpt["state_dict"])  # type: ignore[index]
                            ensemble.append(clf.to(device).eval())

                        # Use provided threshold or automatic boundary score
                        thresh = classification_threshold if classification_threshold >= 0 else boundary_score

                        dataset_obj = dataloaders["testing"].dataset
                        default_resize = getattr(dataset_obj, "resize", None)
                        default_crop = getattr(dataset_obj, "center_crop_size", None)
                        default_filter_window = getattr(dataset_obj, "filter_window_size", None)
                        default_filter_threshold = getattr(dataset_obj, "filter_threshold", None)
                        if default_crop is None:
                            img_size_tuple = getattr(dataset_obj, "imagesize", None)
                            if isinstance(img_size_tuple, (tuple, list)) and len(img_size_tuple) >= 2:
                                default_crop = img_size_tuple[1]

                        cls_resize = classification_resize if classification_resize is not None else (default_resize or default_crop or 224)
                        cls_crop = classification_imagesize if classification_imagesize is not None else (default_crop or cls_resize)
                        cls_filter_window = classification_filter_window_size if classification_filter_window_size is not None else (default_filter_window or 3)
                        cls_filter_threshold = classification_filter_threshold if classification_filter_threshold is not None else (default_filter_threshold or 1.25)
                        cls_transform = transforms.Compose([
                            transforms.Resize(cls_resize),
                            transforms.CenterCrop(cls_crop),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=WM811K_MEAN, std=WM811K_STD),
                        ])

                        # Select images above threshold
                        to_classify = []
                        to_scores = []
                        for p, s in zip(image_names, scores_mean):
                            if s >= thresh:
                                to_classify.append(p)
                                to_scores.append(float(s))

                        LOGGER.info(f"Classifying {len(to_classify)} images with score >= {thresh:.4f}")

                        def prepare_base_image(img: Image.Image) -> Image.Image:
                            if classification_apply_filter:
                                return WM811KAnomalyClassDataset.constrained_mean_filter(
                                    img,
                                    filter_window_size=cls_filter_window,
                                    threshold=cls_filter_threshold,
                                )
                            return img

                        def compute_density(img: Image.Image) -> np.ndarray:
                            gray = np.array(img.convert("L"))
                            anomaly_mask = gray > 200
                            anomaly_pixels = np.sum(anomaly_mask)
                            
                            # Edge Density (Perimeter / Area)
                            if anomaly_pixels > 0:
                                struct = generate_binary_structure(2, 1) # 4-connectivity
                                eroded = binary_erosion(anomaly_mask, structure=struct)
                                boundary = anomaly_mask ^ eroded 
                                perimeter = np.sum(boundary)
                                edge_ratio = float(perimeter) / float(anomaly_pixels)
                            else:
                                edge_ratio = 0.0
                            
                            return np.array([edge_ratio], dtype=np.float32)

                        def tta_flips(img: Image.Image):
                            base = prepare_base_image(img)
                            if not classification_tta:
                                return [cls_transform(base)], base
                            img_h = base.transpose(Image.FLIP_LEFT_RIGHT)
                            img_v = base.transpose(Image.FLIP_TOP_BOTTOM)
                            img_hv = img_h.transpose(Image.FLIP_TOP_BOTTOM)
                            return [cls_transform(base), cls_transform(img_h), cls_transform(img_v), cls_transform(img_hv)], base

                        batch_size_cls = 64
                        preds = []
                        probs = []
                        for i_b in range(0, len(to_classify), batch_size_cls):
                            batch_paths = to_classify[i_b:i_b+batch_size_cls]
                            all_imgs = []
                            all_densities = []
                            counts = []  # number of TTA variants per original image
                            for p in batch_paths:
                                try:
                                    img = Image.open(p).convert("RGB")
                                except Exception:
                                    rp = os.path.join(dataloaders["testing"].dataset.source, p)
                                    img = Image.open(rp).convert("RGB")
                                variants, base_img = tta_flips(img)
                                counts.append(len(variants))
                                all_imgs.extend(variants)
                                if use_density_global:
                                    d = compute_density(base_img)
                                    all_densities.extend([d] * len(variants))

                            if not all_imgs:
                                continue
                            x = torch.stack(all_imgs, dim=0).to(device)
                            
                            density_tensor = None
                            if use_density_global:
                                density_tensor = torch.tensor(all_densities, dtype=torch.float32, device=device)

                            with torch.no_grad():
                                # Accumulate probabilities per original image across ensemble + TTA
                                start = 0
                                probs_batch = []
                                for c in counts:
                                    # Slice TTA block
                                    tta_block = x[start:start+c]
                                    d_block = density_tensor[start:start+c] if density_tensor is not None else None
                                    start += c
                                    prob_acc = None
                                    for model_inst in ensemble:
                                        logits = model_inst(tta_block, d_block)
                                        pr = torch.softmax(logits, dim=1).mean(dim=0)  # average over TTA variants
                                        prob_acc = pr if prob_acc is None else (prob_acc + pr)
                                    prob_acc = prob_acc / len(ensemble)
                                    probs_batch.append(prob_acc)
                                for prob_vec in probs_batch:
                                    pr_max, pr_idx = torch.max(prob_vec, dim=0)
                                    preds.append(int(pr_idx.item()))
                                    probs.append(float(pr_max.item()))

                        # Save classification CSV
                        save_cls_csv = classification_save_csv or os.path.join(run_save_path, f"anomaly_classifications_{dataset_name}.csv")
                        os.makedirs(os.path.dirname(save_cls_csv) or run_save_path, exist_ok=True)
                        pred_label_names = [
                            classes[idx] if classes and 0 <= idx < len(classes) else str(idx)
                            for idx in preds
                        ]
                        with open(save_cls_csv, "w", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(["image", "score", "pred_class", "prob"])
                            for pth, sc, name, pr in zip(to_classify, to_scores, pred_label_names, probs):
                                writer.writerow([pth, f"{sc:.6f}", name, f"{pr:.6f}"])
                        LOGGER.info(f"Saved anomaly classifications to: {save_cls_csv}")

                        eval_true = []
                        eval_pred = []
                        for pth, pred_name in zip(to_classify, pred_label_names):
                            gt_name = _infer_label_from_path(pth, classes)
                            if gt_name is None or gt_name == "good":
                                continue
                            eval_true.append(gt_name)
                            eval_pred.append(pred_name)

                        if eval_true:
                            label_order = list(classes) if classes else sorted(set(eval_true) | set(eval_pred))
                            metrics_report = _compute_classification_metrics(eval_true, eval_pred, label_order)
                            _log_classification_metrics(metrics_report)

                            if classification_metrics_path:
                                metrics_file = classification_metrics_path
                                if not os.path.splitext(metrics_file)[1]:
                                    metrics_file = os.path.join(metrics_file, f"classification_metrics_{dataset_name}.json")
                            else:
                                metrics_file = os.path.join(run_save_path, f"classification_metrics_{dataset_name}.json")
                            os.makedirs(os.path.dirname(metrics_file) or run_save_path, exist_ok=True)
                            with open(metrics_file, "w", encoding="utf-8") as mf:
                                json.dump(metrics_report, mf, indent=2)
                            LOGGER.info(f"Classification metrics saved to: {metrics_file}")

                            if classification_report_csv:
                                rep_path = classification_report_csv
                                if not os.path.splitext(rep_path)[1]:
                                    rep_path = os.path.join(rep_path, f"classification_report_{dataset_name}.csv")
                                _write_classification_report_csv(rep_path, metrics_report, label_order)
                                LOGGER.info(f"Saved classification report to: {rep_path}")
                        else:
                            LOGGER.info("Classification metrics skipped (no labeled anomaly ground-truth inferred from paths).")
                    except Exception as e:
                        LOGGER.error(f"Failed to run anomaly classification: {e}")

        LOGGER.info("\n\n-----\n")
    ct = datetime.now()
    print("finish current time: ", ct)
    # Store all results and mean scores to a csv-file.
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]

    #with open('image_scores.json', 'w') as f:
    #    json.dump(image_scores, f)
    
    # per-dataset image score CSVs are written inside the loop when requested.

    patchcore.utils.compute_and_store_final_results(
        run_save_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )


@main.command("patch_core")
# Pretraining-specific parameters.
@click.option("--backbone_names", "-b", type=str, multiple=True, default=[])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[])
# Parameters for Glue-code (to merge different parts of the pipeline.
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--preprocessing", type=click.Choice(["mean", "conv"]), default="mean")
@click.option("--aggregation", type=click.Choice(["mean", "mlp"]), default="mean")
# Nearest-Neighbour Anomaly Scorer parameters.
@click.option("--anomaly_scorer_num_nn", type=int, default=5)
# Patch-parameters.
@click.option("--patchsize", type=int, default=3)
@click.option("--patchscore", type=str, default="max")
@click.option("--patchoverlap", type=float, default=0.0)
@click.option("--patchsize_aggregate", "-pa", type=int, multiple=True, default=[])
# NN on GPU.
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8)
@click.option("--use_geometry", is_flag=True, help="Enable polar geometry-aware PatchCore.")
@click.option("--cbam_checkpoint", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--embedding_adapter_path", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--radial_bins", type=int, default=4, show_default=True)
@click.option("--angular_bins", type=int, default=8, show_default=True)
@click.option("--min_wafer_coverage", type=float, default=0.5, show_default=True)
@click.option("--geometry_radial_neighbors", type=int, default=1, show_default=True)
@click.option("--geometry_angular_neighbors", type=int, default=1, show_default=True)
@click.option("--feature_adapter_path", type=click.Path(exists=True, dir_okay=False), default=None, help="Path to pre-trained feature adapter weights.")
@click.option("--use_fca", is_flag=True, help="Enable Frequency+Channel Attention on extracted feature maps")
@click.option("--fca_reduction", type=int, default=16, help="FCA channel reduction ratio (passed to FCA module)")
@click.option("--fca_freq_channels", type=int, default=16, help="FCA freq_conv intermediate channels")
@click.option("--use_cbam", is_flag=True, help="Enable CBAM (channel + spatial attention) on extracted feature maps")
@click.option("--cbam_reduction", type=int, default=16, help="CBAM channel reduction ratio (passed to CBAM module)")
@click.option("--cbam_spatial_kernel", type=int, default=7, help="CBAM spatial attention kernel size")
def patch_core(
    backbone_names,
    layers_to_extract_from,
    pretrain_embed_dimension,
    target_embed_dimension,
    preprocessing,
    aggregation,
    patchsize,
    patchscore,
    patchoverlap,
    anomaly_scorer_num_nn,
    patchsize_aggregate,
    faiss_on_gpu,
    faiss_num_workers,
    use_geometry,
    cbam_checkpoint,
    embedding_adapter_path,
    radial_bins,
    angular_bins,
    min_wafer_coverage,
    geometry_radial_neighbors,
    geometry_angular_neighbors,
    feature_adapter_path,
    use_fca,
    fca_reduction,
    fca_freq_channels,
    use_cbam,
    cbam_reduction,
    cbam_spatial_kernel,
):
    if use_fca or use_cbam:
        raise click.UsageError(
            "FCA and randomly initialized CBAM are archived. "
            "Use --cbam_checkpoint with --use_geometry."
        )
    if cbam_checkpoint and not use_geometry:
        raise click.UsageError("--cbam_checkpoint requires --use_geometry.")
    if embedding_adapter_path and not use_geometry:
        raise click.UsageError("--embedding_adapter_path requires --use_geometry.")
    backbone_names = list(backbone_names)
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
        for layer in layers_to_extract_from:
            idx = int(layer.split(".")[0])
            layer = ".".join(layer.split(".")[1:])
            layers_to_extract_from_coll[idx].append(layer)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    def get_patchcore(input_shape, sampler, device):
        loaded_patchcores = []
        for backbone_name, layers_to_extract_from in zip(
            backbone_names, layers_to_extract_from_coll
        ):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                    backbone_name.split("-")[-1]
                )
            backbone = patchcore.backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed

            nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)

            core_class = (
                patchcore.geometry_patchcore.PatchCore
                if use_geometry else patchcore.patchcore.PatchCore
            )
            patchcore_instance = core_class(device)
            # build FCA params dict if requested
            fca_params = {"reduction": fca_reduction, "freq_conv_channels": fca_freq_channels} if use_fca else None
            cbam_params = {"reduction": cbam_reduction, "spatial_kernel": cbam_spatial_kernel} if use_cbam else None

            patchcore_instance.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                featuresampler=sampler,
                anomaly_scorer_num_nn=anomaly_scorer_num_nn,
                nn_method=nn_method,
                cbam_checkpoint=cbam_checkpoint,
                embedding_adapter_path=embedding_adapter_path,
                radial_bins=radial_bins,
                angular_bins=angular_bins,
                min_wafer_coverage=min_wafer_coverage,
                geometry_radial_neighbors=geometry_radial_neighbors,
                geometry_angular_neighbors=geometry_angular_neighbors,
                feature_adapter_path=feature_adapter_path,
                use_fca=use_fca,
                fca_params=fca_params,
                use_cbam=use_cbam,
                cbam_params=cbam_params,
            )
            loaded_patchcores.append(patchcore_instance)
        return loaded_patchcores

    return ("get_patchcore", get_patchcore)


@main.command("sampler")
@click.argument("name", type=str)
@click.option("--percentage", "-p", type=float, default=0.1, show_default=True)
@click.option("--seed", type=int, default=None, help="Optional seed for deterministic samplers")
@click.option("--num_starting_points", type=int, default=10, help="(approx_greedy_coreset) number of starting points to use")
def sampler(name, percentage, seed, num_starting_points):
    def get_sampler(device):
        if name == "identity":
            return patchcore.sampler.IdentitySampler()
        elif name == "greedy_coreset":
            return patchcore.sampler.GreedyCoresetSampler(percentage, device)
        elif name == "approx_greedy_coreset":
            return patchcore.sampler.ApproximateGreedyCoresetSampler(
                percentage, device, number_of_starting_points=num_starting_points
            )
        elif name == "random":
            return patchcore.sampler.RandomSampler(percentage)
        elif name == "seeded_random":
            return patchcore.sampler.SeededRandomSampler(percentage, seed=seed)
        elif name == "geometry_coreset":
            return patchcore.sampler.GeometryAwareCoresetSampler(
                percentage,
                device,
                seed=0 if seed is None else seed,
                number_of_starting_points=num_starting_points,
            )
        raise click.BadParameter(f"Unknown sampler: {name}")

    return ("get_sampler", get_sampler)


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--train_val_split", type=float, default=1, show_default=True)
@click.option("--batch_size", default=2, type=int, show_default=True)
@click.option("--num_workers", default=8, type=int, show_default=True)
@click.option("--resize", default=256, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
@click.option("--augment", is_flag=True)
@click.option("--filter_window_size", type=int, default=3, show_default=True, help="Window size for constrained mean filter (wm811k dataset).")
@click.option("--filter_threshold", type=float, default=1.25, show_default=True, help="Threshold for constrained mean filter (wm811k dataset).")
@click.option(
    "--transform_mode",
    type=click.Choice(["resize_pad", "resize_only", "resize_crop"]),
    default="resize_pad",
    show_default=True,
    help="Wafer resize policy. resize_pad preserves full wafer content and never crops.",
)
def dataset(
    name,
    data_path,
    subdatasets,
    train_val_split,
    batch_size,
    resize,
    imagesize,
    num_workers,
    augment,
    filter_window_size,
    filter_threshold,
    transform_mode,
):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders(seed):
        dataloaders = []
        for subdataset in subdatasets:
            train_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                train_val_split=train_val_split,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TRAIN,
                seed=seed,
                augment=augment,
                filter_window_size=filter_window_size,
                filter_threshold=filter_threshold,
                transform_mode=transform_mode,
            )

            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
                filter_window_size=filter_window_size,
                filter_threshold=filter_threshold,
                transform_mode=transform_mode,
            )

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            train_dataloader.name = name
            if subdataset is not None:
                train_dataloader.name += "_" + subdataset

            if train_val_split < 1:
                val_dataset = dataset_library.__dict__[dataset_info[1]](
                    data_path,
                    classname=subdataset,
                    resize=resize,
                    train_val_split=train_val_split,
                    imagesize=imagesize,
                    split=dataset_library.DatasetSplit.VAL,
                    seed=seed,
                    filter_window_size=filter_window_size,
                    filter_threshold=filter_threshold,
                    transform_mode=transform_mode,
                )

                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            else:
                val_dataloader = None
            dataloader_dict = {
                "training": train_dataloader,
                "validation": val_dataloader,
                "testing": test_dataloader,
            }

            dataloaders.append(dataloader_dict)
        return dataloaders

    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
