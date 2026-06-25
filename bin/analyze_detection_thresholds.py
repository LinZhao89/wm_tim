import argparse
import csv
import os
from pathlib import PurePath

import numpy as np


def _category_from_path(path):
    parts = PurePath(path).parts
    if "test" in parts:
        idx = parts.index("test")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return parts[-2]


def _metrics(scores, labels, threshold):
    preds = scores >= threshold
    labels = labels.astype(bool)
    tp = int(np.sum(preds & labels))
    fp = int(np.sum(preds & ~labels))
    tn = int(np.sum(~preds & ~labels))
    fn = int(np.sum(~preds & labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) else 0.0
    balanced_accuracy = (recall + specificity) / 2
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall_anomaly_detection_rate": recall,
        "specificity_good_detection_rate": specificity,
        "f1": f1,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "predicted_anomaly_rate": float(np.mean(preds)),
    }


def _threshold_for_top_rate(scores, rate):
    sorted_scores = sorted(scores, reverse=True)
    idx = min(max(int(rate * len(sorted_scores)), 0), len(sorted_scores) - 1)
    return float(sorted_scores[idx])


def _threshold_for_recall(scores, labels, target_recall):
    anomaly_scores = np.sort(scores[labels.astype(bool)])
    if len(anomaly_scores) == 0:
        return float("nan")
    # To detect target_recall of anomalies, threshold must be at this anomaly-score percentile.
    idx = int(np.floor((1 - target_recall) * (len(anomaly_scores) - 1)))
    return float(anomaly_scores[max(idx, 0)])


def _best_thresholds(scores, labels):
    thresholds = np.unique(scores)
    best_f1 = None
    best_balanced = None
    best_youden = None
    for threshold in thresholds:
        row = _metrics(scores, labels, float(threshold))
        youden = row["recall_anomaly_detection_rate"] + row["specificity_good_detection_rate"] - 1
        if best_f1 is None or row["f1"] > best_f1["f1"]:
            best_f1 = row
        if best_balanced is None or row["balanced_accuracy"] > best_balanced["balanced_accuracy"]:
            best_balanced = row
        if best_youden is None or youden > best_youden["youden_j"]:
            best_youden = dict(row, youden_j=youden)
    return best_f1["threshold"], best_balanced["threshold"], best_youden["threshold"]


def _write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    records = []
    with open(args.scores_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            category = _category_from_path(row["image"])
            score = float(row["image_score_mean"])
            records.append({"image": row["image"], "category": category, "score": score})

    scores = np.asarray([row["score"] for row in records], dtype=np.float64)
    labels = np.asarray([row["category"] != "good" for row in records], dtype=np.int32)
    categories = sorted({row["category"] for row in records})

    best_f1, best_balanced, best_youden = _best_thresholds(scores, labels)
    threshold_specs = [
        ("current_top_5_percent", _threshold_for_top_rate(scores, 0.05)),
        ("top_10_percent", _threshold_for_top_rate(scores, 0.10)),
        ("top_20_percent", _threshold_for_top_rate(scores, 0.20)),
        ("top_33_percent", _threshold_for_top_rate(scores, 0.33)),
        ("max_f1", best_f1),
        ("max_balanced_accuracy", best_balanced),
        ("youden_j", best_youden),
        ("recall_90_percent", _threshold_for_recall(scores, labels, 0.90)),
        ("recall_95_percent", _threshold_for_recall(scores, labels, 0.95)),
    ]

    summary_rows = []
    category_rows = []
    confusion_rows = []
    for name, threshold in threshold_specs:
        metrics = _metrics(scores, labels, threshold)
        summary_rows.append({"threshold_name": name, **metrics})

        preds = scores >= threshold
        for category in categories:
            idxs = [i for i, row in enumerate(records) if row["category"] == category]
            pred_anomaly = int(np.sum(preds[idxs]))
            pred_good = len(idxs) - pred_anomaly
            detected = pred_good if category == "good" else pred_anomaly
            category_rows.append(
                {
                    "threshold_name": name,
                    "threshold": threshold,
                    "category": category,
                    "total": len(idxs),
                    "predicted_good": pred_good,
                    "predicted_anomaly": pred_anomaly,
                    "detected": detected,
                    "detected_rate": detected / len(idxs) if idxs else float("nan"),
                }
            )
            confusion_rows.append(
                {
                    "threshold_name": name,
                    "true_category": category,
                    "predicted_good": pred_good,
                    "predicted_anomaly": pred_anomaly,
                }
            )

    _write_csv(
        os.path.join(args.output_dir, "threshold_summary.csv"),
        summary_rows,
        [
            "threshold_name",
            "threshold",
            "tp",
            "fp",
            "tn",
            "fn",
            "precision",
            "recall_anomaly_detection_rate",
            "specificity_good_detection_rate",
            "f1",
            "accuracy",
            "balanced_accuracy",
            "predicted_anomaly_rate",
        ],
    )
    _write_csv(
        os.path.join(args.output_dir, "category_detection_by_threshold.csv"),
        category_rows,
        [
            "threshold_name",
            "threshold",
            "category",
            "total",
            "predicted_good",
            "predicted_anomaly",
            "detected",
            "detected_rate",
        ],
    )
    _write_csv(
        os.path.join(args.output_dir, "confusion_true_category_vs_binary_prediction.csv"),
        confusion_rows,
        ["threshold_name", "true_category", "predicted_good", "predicted_anomaly"],
    )


if __name__ == "__main__":
    main()
