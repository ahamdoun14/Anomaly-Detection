"""Keras model architecture."""

from tensorflow import keras


def build_conv_autoencoder(
    input_shape: tuple[int, int, int] = (128, 128, 3),
    latent_dim: int = 64,
) -> keras.Model:
    """Build the convolutional autoencoder used in the project.

    The five stride-2 encoder blocks expect image height and width to be
    divisible by 32. The default 128 x 128 input becomes a 4 x 4 feature map.
    """
    height, width, channels = input_shape
    if height % 32 != 0 or width % 32 != 0:
        raise ValueError("Image height and width must be divisible by 32.")
    if channels != 3:
        raise ValueError("This architecture expects RGB images with 3 channels.")
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive.")

    reduced_height = height // 32
    reduced_width = width // 32

    inputs = keras.Input(shape=input_shape, name="image")

    x = inputs
    for filters in (32, 64, 128, 256, 512):
        x = keras.layers.Conv2D(
            filters,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu",
        )(x)

    x = keras.layers.Flatten()(x)
    latent = keras.layers.Dense(latent_dim, activation="relu", name="latent_vector")(x)

    x = keras.layers.Dense(reduced_height * reduced_width * 512, activation="relu")(latent)
    x = keras.layers.Reshape((reduced_height, reduced_width, 512))(x)

    for filters in (256, 128, 64, 32, 16):
        x = keras.layers.Conv2DTranspose(
            filters,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu",
        )(x)

    outputs = keras.layers.Conv2D(
        3,
        kernel_size=3,
        padding="same",
        activation="sigmoid",
        name="reconstruction",
    )(x)

    return keras.Model(inputs, outputs, name="compressed_autoencoder")
