# %%
import os
import gc
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
#--------------------------------------------
import tensorflow as tf

print("TensorFlow:", tf.__version__)

print("CPUs:")
print(tf.config.list_physical_devices("CPU"))

print("GPUs:")
print(tf.config.list_physical_devices("GPU"))

print("Done")
# ─── Config ───────────────────────────────────────────────────────────────────
DATASET_PATH = Path("/home/ayoub/jupyter/Anomaly_detection_project/mvtec_anomaly_detection")
CATEGORY     = "bottle"
IMG_SIZE     = (128, 128)
BATCH_SIZE   = 32
EPOCHS       = 50
LR           = 1e-4
SEED         = 42

# ─── Reproducibility / stability ──────────────────────────────────────────────
os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# Makes TensorFlow operations more deterministic
tf.config.experimental.enable_op_determinism()

# Optional: reduce GPU memory problems
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# ─── Data pipeline ────────────────────────────────────────────────────────────
def load_train_val_split(path, subset):
    ds = keras.utils.image_dataset_from_directory(
        path,
        labels=None,
        color_mode="rgb",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset=subset,
        seed=SEED,
        shuffle=(subset == "training"),
    )

    # Autoencoder: input = target
    ds = ds.map(
        lambda x: (x / 255.0, x / 255.0),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # During debugging, avoid cache()
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def load_images_only(path):
    ds = keras.utils.image_dataset_from_directory(
        path,
        labels=None,
        color_mode="rgb",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    x = np.concatenate(
        [batch.numpy() / 255.0 for batch in ds],
        axis=0
    )

    return x


train_path = DATASET_PATH / CATEGORY / "train" / "good"

train_ds = load_train_val_split(train_path, "training")
val_ds   = load_train_val_split(train_path, "validation")


# ─── Model ────────────────────────────────────────────────────────────────────
def build_conv_autoencoder(input_shape=(128, 128, 3), latent_dim=64):
    inputs = keras.Input(shape=input_shape)

    # Encoder
    x = keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)
    x = keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(512, 3, strides=2, padding="same", activation="relu")(x)

    x = keras.layers.Flatten()(x)
    latent = keras.layers.Dense(latent_dim, activation="relu", name="latent_vector")(x)

    # Decoder
    x = keras.layers.Dense(4 * 4 * 512, activation="relu")(latent)
    x = keras.layers.Reshape((4, 4, 512))(x)

    x = keras.layers.Conv2DTranspose(256, 3, strides=2, padding="same", activation="relu")(x)
    x = keras.layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
    x = keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    x = keras.layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu")(x)

    outputs = keras.layers.Conv2D(3, 3, padding="same", activation="sigmoid")(x)

    return keras.Model(inputs, outputs, name="compressed_autoencoder")


keras.backend.clear_session()
gc.collect()

model = build_conv_autoencoder(input_shape=(128, 128, 3), latent_dim=64)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    loss="mse",
    metrics=[keras.metrics.MeanSquaredError(name="mse")]
)

model.summary()


# ─── Callbacks ────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        "best_autoencoder.keras",
        monitor="val_loss",
        save_best_only=True
    ),
]


# ─── Training ─────────────────────────────────────────────────────────────────
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks,
)


# ─── Training curves ──────────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.tight_layout()
plt.show()


# ─── Scoring function ─────────────────────────────────────────────────────────
def reconstruction_scores(images, model, percentile=99.9):
    recons = model.predict(images, verbose=0)

    # Pixel-level MSE error map
    err_maps = np.mean((images - recons) ** 2, axis=-1)

    # Image-level anomaly score
    scores = np.percentile(err_maps, percentile, axis=(1, 2))

    return scores, err_maps, recons


# ─── Threshold from validation-good only ──────────────────────────────────────
x_val = np.concatenate(
    [x_batch.numpy() for x_batch, y_batch in val_ds],
    axis=0
)

normal_scores, normal_maps, normal_recons = reconstruction_scores(
    x_val,
    model,
    percentile=99.9
)

# More conservative than 95
threshold = np.percentile(normal_scores, 99)

print(f"Threshold from validation-good: {threshold:.6f}")


# ─── Load test/good and test/broken_large ─────────────────────────────────────
test_good_path = DATASET_PATH / CATEGORY / "test" / "good"
broken_path    = DATASET_PATH / CATEGORY / "test" / "broken_large"

x_good   = load_images_only(test_good_path)
x_broken = load_images_only(broken_path)

good_scores, good_maps, good_recons = reconstruction_scores(
    x_good,
    model,
    percentile=99.9
)

broken_scores, broken_maps, broken_recons = reconstruction_scores(
    x_broken,
    model,
    percentile=99.9
)


# ─── Print simple statistics ──────────────────────────────────────────────────
print("\nScore statistics:")
print(f"Validation-good mean: {normal_scores.mean():.6f}")
print(f"Validation-good max : {normal_scores.max():.6f}")
print(f"Test-good mean      : {good_scores.mean():.6f}")
print(f"Test-good max       : {good_scores.max():.6f}")
print(f"Broken mean         : {broken_scores.mean():.6f}")
print(f"Broken min          : {broken_scores.min():.6f}")
print(f"Broken max          : {broken_scores.max():.6f}")

print("\nClassification with threshold:")
print(f"Test-good predicted anomaly   : {(good_scores > threshold).sum()} / {len(good_scores)}")
print(f"Broken predicted anomaly      : {(broken_scores > threshold).sum()} / {len(broken_scores)}")


# ─── Histogram ────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.hist(good_scores, bins=20, alpha=0.6, label="Test Good")
plt.hist(broken_scores, bins=20, alpha=0.6, label="Broken Large")
plt.axvline(threshold, linestyle="--", label=f"Threshold={threshold:.4f}")
plt.xlabel("Anomaly score: p99.9 pixel MSE")
plt.ylabel("Number of images")
plt.title("Score Distribution: Good vs Broken Large")
plt.legend()
plt.tight_layout()
plt.show()


# ─── Visualization helper ─────────────────────────────────────────────────────
def show_reconstruction_example(images, recons, maps, scores, index=0, title="Example"):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(images[index])
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(recons[index])
    plt.title("Reconstruction")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(maps[index], cmap="hot")
    plt.title(f"Error map\nscore={scores[index]:.6f}")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# ─── Show examples ────────────────────────────────────────────────────────────
show_reconstruction_example(
    x_good,
    good_recons,
    good_maps,
    good_scores,
    index=0,
    title="Normal test image"
)

show_reconstruction_example(
    x_broken,
    broken_recons,
    broken_maps,
    broken_scores,
    index=0,
    title="Broken_large test image"
)