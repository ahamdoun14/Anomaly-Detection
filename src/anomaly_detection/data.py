"""TensorFlow input pipelines for MVTec AD."""

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

SUPPORTED_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png"}


def create_train_val_datasets(
    good_train_dir: str | Path,
    image_size: tuple[int, int] = (128, 128),
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 42,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Create normalized autoencoder datasets from normal training images."""
    good_train_dir = Path(good_train_dir)
    if not good_train_dir.is_dir():
        raise FileNotFoundError(f"Training directory not found: {good_train_dir}")

    common = dict(
        directory=good_train_dir,
        labels=None,
        color_mode="rgb",
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split,
        seed=seed,
    )

    train = keras.utils.image_dataset_from_directory(
        subset="training",
        shuffle=True,
        **common,
    )
    validation = keras.utils.image_dataset_from_directory(
        subset="validation",
        shuffle=False,
        **common,
    )

    def normalize_for_autoencoder(batch: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        batch = tf.cast(batch, tf.float32) / 255.0
        return batch, batch

    train = train.map(normalize_for_autoencoder, num_parallel_calls=tf.data.AUTOTUNE)
    validation = validation.map(normalize_for_autoencoder, num_parallel_calls=tf.data.AUTOTUNE)

    return train.prefetch(tf.data.AUTOTUNE), validation.prefetch(tf.data.AUTOTUNE)


def image_paths(directory: str | Path) -> list[Path]:
    """Return sorted image paths below a directory."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No supported images found in {directory}")
    return paths


def load_images_only(
    directory: str | Path,
    image_size: tuple[int, int] = (128, 128),
    batch_size: int = 32,
) -> tuple[np.ndarray, list[Path]]:
    """Load a directory of images as a normalized NumPy array."""
    paths = image_paths(directory)
    dataset = tf.data.Dataset.from_tensor_slices([str(path) for path in paths])

    def decode(path: tf.Tensor) -> tf.Tensor:
        raw = tf.io.read_file(path)
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        image = tf.image.resize(image, image_size)
        return tf.cast(image, tf.float32) / 255.0

    dataset = dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    images = np.concatenate([batch.numpy() for batch in dataset], axis=0)
    return images, paths


def dataset_to_numpy(dataset: tf.data.Dataset) -> np.ndarray:
    """Extract the input image batches from an autoencoder dataset."""
    batches = [images.numpy() for images, _targets in dataset]
    if not batches:
        raise ValueError("Dataset contains no batches.")
    return np.concatenate(batches, axis=0)
