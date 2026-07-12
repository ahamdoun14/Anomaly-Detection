import numpy as np
from PIL import Image

from anomaly_detection.preprocessing import prepare_pil_image


def test_prepare_pil_image_returns_normalized_rgb_batch():
    image = Image.new("L", (20, 10), color=128)

    batch = prepare_pil_image(image, image_size=(16, 12))

    assert batch.shape == (1, 12, 16, 3)
    assert batch.dtype == np.float32
    assert 0.0 <= float(batch.min()) <= float(batch.max()) <= 1.0
