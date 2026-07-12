# Model Card: MVTec Bottle Reconstruction Autoencoder

## Summary

This project uses a convolutional autoencoder for unsupervised visual anomaly
detection. It is trained only on defect-free bottle images. At inference time,
the discrepancy between an input and its reconstruction is used as evidence of
an anomaly.

## Intended use

- Educational and portfolio demonstrations of industrial anomaly detection
- Baseline experiments on the MVTec AD bottle category
- Visual inspection of reconstruction-error maps

It is not intended to make autonomous production-quality decisions without
additional validation, calibration, monitoring, and safety controls.

## Inputs and outputs

- Input: RGB bottle image resized to 128 x 128 pixels by default
- Outputs: reconstructed image, pixel-level error map, image-level anomaly score,
  and binary decision based on a validation-derived threshold

## Training data

Only images in `bottle/train/good` are used for model fitting. A validation split
from the same normal-only set is used to calibrate the decision threshold.

## Scoring

1. Compute channel-averaged pixel-wise squared reconstruction error.
2. Use the 99.9th percentile of the error map as the image score.
3. Set the default threshold to the 99th percentile of normal validation scores.

Both percentiles are configurable.

## Limitations

- Autoencoders can sometimes reconstruct defects and therefore miss anomalies.
- Resizing to 128 x 128 may remove small defect details.
- The error map is an explanatory localization signal, not a calibrated
  segmentation mask.
- Performance can change under new lighting, cameras, backgrounds, rotations,
  and manufacturing domains.
- The threshold must be recalibrated when the acquisition process changes.

## Recommended extensions

- Compare against PatchCore, PaDiM, EfficientAD, or student-teacher methods.
- Add pixel-level AUROC, AUPRO, and mask-based evaluation.
- Add orientation and photometric robustness experiments.
- Export an optimized inference format and measure latency.
