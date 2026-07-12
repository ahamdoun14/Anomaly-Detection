"""Pure NumPy anomaly-scoring utilities."""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


class PredictiveModel(Protocol):
    """Minimal interface required by :func:`reconstruction_scores`."""

    def predict(self, images: FloatArray, **kwargs: object) -> FloatArray: ...


def validate_image_batch(images: FloatArray) -> None:
    """Validate an image batch shaped ``(N, H, W, C)``."""
    if images.ndim != 4:
        raise ValueError(f"Expected a 4-D image batch, received shape {images.shape}.")
    if images.shape[-1] not in (1, 3, 4):
        raise ValueError("The final dimension must represent image channels.")
    if images.shape[0] == 0:
        raise ValueError("The image batch is empty.")


def compute_error_maps(images: FloatArray, reconstructions: FloatArray) -> FloatArray:
    """Return channel-averaged pixel-wise squared reconstruction errors."""
    images = np.asarray(images, dtype=np.float32)
    reconstructions = np.asarray(reconstructions, dtype=np.float32)
    validate_image_batch(images)
    validate_image_batch(reconstructions)
    if images.shape != reconstructions.shape:
        raise ValueError(
            "Images and reconstructions must have identical shapes; "
            f"received {images.shape} and {reconstructions.shape}."
        )
    return np.mean(np.square(images - reconstructions), axis=-1)


def image_scores_from_error_maps(
    error_maps: FloatArray,
    percentile: float = 99.9,
) -> FloatArray:
    """Aggregate each error map into one image-level anomaly score."""
    error_maps = np.asarray(error_maps, dtype=np.float32)
    if error_maps.ndim != 3:
        raise ValueError("Expected error maps shaped (N, H, W).")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100.")
    return np.percentile(error_maps, percentile, axis=(1, 2))


def reconstruction_scores(
    images: FloatArray,
    model: PredictiveModel,
    percentile: float = 99.9,
    batch_size: int = 32,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Reconstruct images and return scores, error maps, and reconstructions."""
    images = np.asarray(images, dtype=np.float32)
    validate_image_batch(images)
    reconstructions = np.asarray(
        model.predict(images, batch_size=batch_size, verbose=0),
        dtype=np.float32,
    )
    error_maps = compute_error_maps(images, reconstructions)
    scores = image_scores_from_error_maps(error_maps, percentile=percentile)
    return scores, error_maps, reconstructions


def calibrate_threshold(normal_scores: FloatArray, percentile: float = 99.0) -> float:
    """Calibrate a decision threshold from normal validation scores only."""
    normal_scores = np.asarray(normal_scores, dtype=np.float32)
    if normal_scores.ndim != 1 or normal_scores.size == 0:
        raise ValueError("normal_scores must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(normal_scores)):
        raise ValueError("normal_scores contains non-finite values.")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100.")
    return float(np.percentile(normal_scores, percentile))


def classify_scores(scores: FloatArray, threshold: float) -> NDArray[np.bool_]:
    """Return ``True`` for scores strictly above the anomaly threshold."""
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional.")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    return scores > threshold
