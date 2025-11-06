# src/download_model.py
import os
import pathlib
import requests

# Expect a direct-download URL (S3, Dropbox ?dl=1, GitHub Release asset, etc.)
MODEL_URL = os.environ.get("MODEL_URL")
if not MODEL_URL:
    raise SystemExit("MODEL_URL env var is not set. Add it in Render → Environment.")

DEST = pathlib.Path("src/models/cnn_xray_classifier.keras")
DEST.parent.mkdir(parents=True, exist_ok=True)

print(f"[download_model] Downloading from: {MODEL_URL}")
with requests.get(MODEL_URL, stream=True, timeout=600) as r:
    r.raise_for_status()
    with open(DEST, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)
print(f"[download_model] Saved to: {DEST.resolve()}")
