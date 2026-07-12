"""Train the convolutional autoencoder on defect-free MVTec AD images."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from anomaly_detection.data import create_train_val_datasets, dataset_to_numpy
from anomaly_detection.metadata import save_metadata
from anomaly_detection.model import build_conv_autoencoder
from anomaly_detection.scoring import calibrate_threshold, reconstruction_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=os.getenv("MVTEC_DATASET_PATH"),
        required=os.getenv("MVTEC_DATASET_PATH") is None,
        help="Path to the MVTec AD root directory.",
    )
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--score-percentile", type=float, default=99.9)
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def configure_runtime(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def plot_history(history: keras.callbacks.History, destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(history.history["loss"], label="Training loss")
    axis.plot(history.history["val_loss"], label="Validation loss")
    axis.set(xlabel="Epoch", ylabel="MSE loss", title="Training and validation loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    configure_runtime(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    good_train_dir = args.dataset_path / args.category / "train" / "good"
    train_ds, val_ds = create_train_val_datasets(
        good_train_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        seed=args.seed,
    )

    keras.backend.clear_session()
    gc.collect()
    model = build_conv_autoencoder(
        input_shape=(args.image_size, args.image_size, 3),
        latent_dim=args.latent_dim,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="mse",
        metrics=[keras.metrics.MeanSquaredError(name="mse")],
    )

    model_path = args.output_dir / "best_autoencoder.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
    )

    pd.DataFrame(history.history).to_csv(args.output_dir / "training_history.csv", index=False)
    plot_history(history, args.output_dir / "training_history.png")

    best_model = keras.models.load_model(model_path, compile=False)
    validation_images = dataset_to_numpy(val_ds)
    normal_scores, _error_maps, _reconstructions = reconstruction_scores(
        validation_images,
        best_model,
        percentile=args.score_percentile,
        batch_size=args.batch_size,
    )
    threshold = calibrate_threshold(normal_scores, percentile=args.threshold_percentile)

    metadata = {
        "category": args.category,
        "image_size": [args.image_size, args.image_size],
        "channels": 3,
        "latent_dim": args.latent_dim,
        "score_percentile": args.score_percentile,
        "threshold_percentile": args.threshold_percentile,
        "threshold": threshold,
        "validation_score_mean": float(np.mean(normal_scores)),
        "validation_score_max": float(np.max(normal_scores)),
        "seed": args.seed,
    }
    save_metadata(args.output_dir / "metadata.json", metadata)

    print(json.dumps(metadata, indent=2))
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
