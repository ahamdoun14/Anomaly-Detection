"""Image preprocessing helpers shared by training and deployment."""

from pathlib import Path
from typing import BinaryIO

import numpy as np
from numpy.typing import NDArray
from PIL import Image


def prepare_pil_image(
    image: Image.Image,
    image_size: tuple[int, int] = (128, 128),
) -> NDArray[np.float32]:
    """Convert an image to normalized RGB and add a batch dimension."""
    if len(image_size) != 2 or min(image_size) <= 0:
        raise ValueError("image_size must contain two positive integers.")

    resized = image.convert("RGB").resize(image_size, Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def load_and_prepare_image(
    source: str | Path | BinaryIO,
    image_size: tuple[int, int] = (128, 128),
) -> NDArray[np.float32]:
    """Open an image path or binary stream and preprocess it."""
    with Image.open(source) as image:
        return prepare_pil_image(image, image_size=image_size)
