# Publish this project on GitHub

## 1. Create the repository

Create an empty GitHub repository named:

```text
mvtec-bottle-anomaly-detection
```

Do not add another README, `.gitignore`, or license on GitHub because they are
already included here.

## 2. Initialize and push

```bash
cd mvtec-bottle-anomaly-detection

git init
git branch -M main
git add .
git commit -m "Build MVTec bottle anomaly-detection portfolio"
git remote add origin git@github.com:ahamdoun14/mvtec-bottle-anomaly-detection.git
git push -u origin main
```

HTTPS alternative:

```bash
git remote add origin https://github.com/ahamdoun14/mvtec-bottle-anomaly-detection.git
```

## 3. Add your trained model

Copy your existing checkpoint into the artifacts directory:

```bash
cp /path/to/best_autoencoder.keras artifacts/best_autoencoder.keras
```

The original experiment printed the threshold but did not save deployment
metadata. Generate it without retraining:

```bash
export MVTEC_DATASET_PATH=/home/ayoub/jupyter/Anomaly_detection_project/mvtec_anomaly_detection

python scripts/calibrate.py \
  --dataset-path "$MVTEC_DATASET_PATH" \
  --category bottle \
  --model-path artifacts/best_autoencoder.keras
```

Then commit the generated files that you want to publish:

```bash
git add artifacts/best_autoencoder.keras artifacts/metadata.json
git commit -m "Add trained autoencoder and calibrated threshold"
git push
```

For a large model, use Git LFS or external model storage rather than normal Git.

## 4. Confirm CI

Open the repository's **Actions** tab. The `CI` workflow should lint the project,
run pytest, and verify that the Streamlit entry point compiles.

## 5. Deploy Streamlit

In Streamlit Community Cloud:

1. Select the GitHub repository.
2. Set the main file path to `app.py`.
3. Deploy the app.

The deployed app needs access to `artifacts/best_autoencoder.keras` and
`artifacts/metadata.json`. The app also supports uploading a `.keras` model at
runtime for local demonstrations.
