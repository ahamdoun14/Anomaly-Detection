# Model artifacts

Running `scripts/train.py` creates:

- `best_autoencoder.keras`: trained Keras model
- `metadata.json`: input size, score percentile, and calibrated threshold
- `training_history.csv`: epoch-level losses
- `training_history.png`: training curve

The Streamlit application automatically reads `best_autoencoder.keras` and
`metadata.json` from this directory. It also supports uploading a `.keras`
model through the sidebar.
