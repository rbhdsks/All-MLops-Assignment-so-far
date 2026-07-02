"""
transform.py — Stage 2 of the DVC pipeline.

What this does:
  For SVM and MLP:
    Extracts HOG (Histogram of Oriented Gradients) features from every image
    in the train/val/test splits and saves them as numpy arrays.

  For CNN:
    The CNN uses on-the-fly augmentation inside the DataLoader, so there's
    nothing to precompute. We just save the augmentation config to disk so
    DVC tracks it as an output of this stage.

Why HOG for SVM/MLP and not raw pixels?
  Raw pixels: 224×224×3 = 150,528 features. That's:
    - Too many for SVM (slow, needs tons of RAM)
    - Terrible for MLP (no spatial structure preserved, just noise)
  HOG captures LOCAL edge directions in small cells, giving you a compact
  ~1764-dimensional descriptor that encodes the ridge/wrinkle patterns of
  a muzzle without the noise of raw pixel values.

  HOG intuition: divide the image into an 8×8 grid of cells. In each cell,
  compute a 9-bin histogram of edge directions. Normalize across 2×2 blocks.
  Result: a vector that's invariant to small lighting changes.

Outputs:
  - features/hog_train.npz  (X_train, y_train)
  - features/hog_val.npz
  - features/hog_test.npz
  - data_processed/v2_augmented/augmentation_config.yaml  (CNN marker)
"""

import os
import yaml
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog  # Assisted by Claude AI
from tqdm import tqdm

# ── load params ────────────────────────────────────────────────────────────────
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

IMG_SIZE    = params["base"]["image_size"]
HOG_ORI     = params["augmentation"]["randaugment_n"]   # reuse for consistency
HOG_ORI     = params["svm"]["hog_orientations"]
HOG_PPC     = params["svm"]["hog_pixels_per_cell"]
HOG_CPB     = params["svm"]["hog_cells_per_block"]
AUG_PARAMS  = params["augmentation"]

SPLITS_DIR   = "splits"
FEATURES_DIR = "features"
V2_DIR       = "data_processed/v2_augmented"

os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(V2_DIR, exist_ok=True)


# ── HOG feature extractor ──────────────────────────────────────────────────────
def extract_hog(img_path: str) -> np.ndarray:
    """
    Extracts a HOG feature vector from one image.

    Steps:
      1. Open image and convert to grayscale
         (HOG works on intensity gradients, color doesn't help here)
      2. Run skimage's hog() with our params
      3. Return a 1D float32 numpy array

    The feature_vector=True flag flattens the 3D output into 1D automatically.
    """
    img = Image.open(img_path).convert("L")  # "L" = grayscale
    img_array = np.array(img)

    features = hog(
        img_array,
        orientations=HOG_ORI,           # 9 direction bins (0°–180°)
        pixels_per_cell=(HOG_PPC, HOG_PPC),  # each cell = 8×8 pixels
        cells_per_block=(HOG_CPB, HOG_CPB),  # normalize over 2×2 cell blocks
        feature_vector=True             # flatten to 1D
    )
    return features.astype(np.float32)


# ── process each split ─────────────────────────────────────────────────────────
for split_name in ["train", "val", "test"]:
    csv_path = os.path.join(SPLITS_DIR, f"{split_name}.csv")
    df = pd.read_csv(csv_path)

    print(f"\n[transform] Extracting HOG features from {split_name} split "
          f"({len(df)} images)...")

    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=split_name):
        try:
            feat = extract_hog(row["filepath"])
            X.append(feat)
            y.append(row["label_idx"])
        except Exception as e:
            print(f"  WARNING: Could not process {row['filepath']}: {e}")

    X = np.array(X)
    y = np.array(y)

    out_path = os.path.join(FEATURES_DIR, f"hog_{split_name}.npz")
    np.savez_compressed(out_path, X=X, y=y)

    print(f"  Saved: {out_path}  |  shape: X={X.shape}, y={y.shape}")


# ── save augmentation config as v2 marker ─────────────────────────────────────
# This tells DVC that v2_augmented is a real output of this stage.
# The actual augmentation for CNN happens inside train_cnn.py's DataLoader.
aug_config_path = os.path.join(V2_DIR, "augmentation_config.yaml")
with open(aug_config_path, "w") as f:
    yaml.dump({"augmentation": AUG_PARAMS, "note": (
        "CNN augmentation is applied on-the-fly in train_cnn.py. "
        "This file records the config for DVC tracking and reproducibility."
    )}, f)

print(f"\n[transform] ✓ Saved augmentation config → {aug_config_path}")
print("[transform] Done.")