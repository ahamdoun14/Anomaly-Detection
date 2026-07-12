import pytest

pytest.importorskip("tensorflow")


def test_autoencoder_preserves_image_shape():
    from anomaly_detection.model import build_conv_autoencoder

    model = build_conv_autoencoder(input_shape=(128, 128, 3), latent_dim=16)

    assert model.input_shape == (None, 128, 128, 3)
    assert model.output_shape == (None, 128, 128, 3)
    assert model.get_layer("latent_vector").units == 16


def test_autoencoder_rejects_invalid_spatial_size():
    from anomaly_detection.model import build_conv_autoencoder

    with pytest.raises(ValueError, match="divisible by 32"):
        build_conv_autoencoder(input_shape=(100, 128, 3))
