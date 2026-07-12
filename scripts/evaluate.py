"""Evaluate image-level anomaly detection across all test defect folders."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tensorflow import keras

from anomaly_detection.data import load_images_only
from anomaly_detection.metadata import load_metadata
from anomaly_detection.scoring import reconstruction_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=os.getenv("MVTEC_DATASET_PATH"),
        required=os.getenv("MVTEC_DATASET_PATH") is None,
    )
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/best_autoencoder.keras"))
    parser.add_argument("--metadata-path", type=Path, default=Path("artifacts/metadata.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(args.metadata_path)
    image_size = tuple(int(value) for value in metadata["image_size"])
    threshold = float(metadata["threshold"])
    score_percentile = float(metadata["score_percentile"])

    model = keras.models.load_model(args.model_path, compile=False)
    test_root = args.dataset_path / args.category / "test"
    defect_dirs = sorted(path for path in test_root.iterdir() if path.is_dir())
    if not defect_dirs:
        raise ValueError(f"No test folders found below {test_root}")

    records: list[dict[str, object]] = []
    all_labels: list[int] = []
    all_scores: list[float] = []

    for defect_dir in defect_dirs:
        images, paths = load_images_only(
            defect_dir,
            image_size=image_size,
            batch_size=args.batch_size,
        )
        scores, _maps, _reconstructions = reconstruction_scores(
            images,
            model,
            percentile=score_percentile,
            batch_size=args.batch_size,
        )
        label = 0 if defect_dir.name == "good" else 1
        for path, score in zip(paths, scores, strict=True):
            records.append(
                {
                    "path": str(path),
                    "defect_type": defect_dir.name,
                    "label": label,
                    "score": float(score),
                    "prediction": int(score > threshold),
                }
            )
        all_labels.extend([label] * len(scores))
        all_scores.extend(float(score) for score in scores)

    results = pd.DataFrame.from_records(records)
    results.to_csv(args.output_dir / "image_scores.csv", index=False)

    labels = np.asarray(all_labels, dtype=int)
    scores = np.asarray(all_scores, dtype=float)
    predictions = (scores > threshold).astype(int)
    precision, recall, f1, _support = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics = {
        "threshold": threshold,
        "image_auroc": float(roc_auc_score(labels, scores)),
        "image_average_precision": float(average_precision_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(8, 4))
    normal = results.loc[results["label"] == 0, "score"]
    anomalous = results.loc[results["label"] == 1, "score"]
    axis.hist(normal, bins=20, alpha=0.65, label="Good")
    axis.hist(anomalous, bins=20, alpha=0.65, label="Defective")
    axis.axvline(threshold, linestyle="--", label=f"Threshold = {threshold:.5f}")
    axis.set(
        xlabel=f"p{score_percentile:g} pixel reconstruction error",
        ylabel="Images",
        title=f"{args.category.title()} anomaly-score distributions",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "score_distribution.png", dpi=160)
    plt.close(figure)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
