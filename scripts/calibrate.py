"""Calibrate metadata for an already-trained autoencoder without retraining it."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from tensorflow import keras

from anomaly_detection.data import create_train_val_datasets, dataset_to_numpy
from anomaly_detection.metadata import save_metadata
from anomaly_detection.scoring import calibrate_threshold, reconstruction_scores


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
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--score-percentile", type=float, default=99.9)
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = keras.models.load_model(args.model_path, compile=False)
    good_train_dir = args.dataset_path / args.category / "train" / "good"
    _train_ds, validation_ds = create_train_val_datasets(
        good_train_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        seed=args.seed,
    )
    validation_images = dataset_to_numpy(validation_ds)
    normal_scores, _error_maps, _reconstructions = reconstruction_scores(
        validation_images,
        model,
        percentile=args.score_percentile,
        batch_size=args.batch_size,
    )
    threshold = calibrate_threshold(normal_scores, percentile=args.threshold_percentile)
    latent_layer = model.get_layer("latent_vector")
    metadata = {
        "category": args.category,
        "image_size": [args.image_size, args.image_size],
        "channels": 3,
        "latent_dim": int(latent_layer.units),
        "score_percentile": args.score_percentile,
        "threshold_percentile": args.threshold_percentile,
        "threshold": threshold,
        "validation_score_mean": float(np.mean(normal_scores)),
        "validation_score_max": float(np.max(normal_scores)),
        "seed": args.seed,
    }
    save_metadata(args.metadata_path, metadata)
    print(json.dumps(metadata, indent=2))
    print(f"Saved metadata to: {args.metadata_path}")


if __name__ == "__main__":
    main()
