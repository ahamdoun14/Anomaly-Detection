import numpy as np
import pytest

from anomaly_detection.scoring import (
    calibrate_threshold,
    classify_scores,
    compute_error_maps,
    image_scores_from_error_maps,
    reconstruction_scores,
)


class IdentityModel:
    def predict(self, images, **kwargs):
        return images


class ZeroModel:
    def predict(self, images, **kwargs):
        return np.zeros_like(images)


def test_compute_error_maps_has_expected_shape_and_value():
    images = np.ones((2, 4, 5, 3), dtype=np.float32)
    reconstructions = np.zeros_like(images)

    error_maps = compute_error_maps(images, reconstructions)

    assert error_maps.shape == (2, 4, 5)
    np.testing.assert_allclose(error_maps, 1.0)


def test_compute_error_maps_rejects_mismatched_shapes():
    images = np.zeros((1, 4, 4, 3), dtype=np.float32)
    reconstructions = np.zeros((1, 8, 8, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="identical shapes"):
        compute_error_maps(images, reconstructions)


def test_image_scores_use_requested_percentile():
    error_maps = np.arange(16, dtype=np.float32).reshape(1, 4, 4)

    score = image_scores_from_error_maps(error_maps, percentile=100.0)

    np.testing.assert_allclose(score, [15.0])


def test_reconstruction_scores_are_zero_for_identity_model():
    images = np.random.default_rng(42).random((3, 8, 8, 3), dtype=np.float32)

    scores, error_maps, reconstructions = reconstruction_scores(images, IdentityModel())

    np.testing.assert_allclose(scores, 0.0)
    np.testing.assert_allclose(error_maps, 0.0)
    np.testing.assert_allclose(reconstructions, images)


def test_reconstruction_scores_for_zero_model():
    images = np.ones((2, 4, 4, 3), dtype=np.float32)

    scores, error_maps, _ = reconstruction_scores(images, ZeroModel(), percentile=99.9)

    np.testing.assert_allclose(error_maps, 1.0)
    np.testing.assert_allclose(scores, 1.0)


def test_threshold_and_classification():
    normal_scores = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float32)
    threshold = calibrate_threshold(normal_scores, percentile=75.0)
    predictions = classify_scores(np.array([0.01, 0.05]), threshold)

    assert threshold == pytest.approx(0.0325)
    assert predictions.tolist() == [False, True]
