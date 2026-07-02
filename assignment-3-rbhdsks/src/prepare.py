"""
prepare.py — Stage 1 of the DVC pipeline.

What this does:
  1. Scans the raw data folder (one subfolder = one class/cow identity)
  2. Drops any class that has fewer than `min_images_per_class` images
     (can't train or evaluate meaningfully on 1-2 samples)
  3. Resizes every image to `image_size x image_size` and saves to
     data_processed/v1_resized/ — this is our first versioned dataset
  4. Performs a STRATIFIED split into train / val / test CSVs
     Stratified = each class keeps the same proportion in every split,
     which is critical when some classes have very few images

Why stratified?
  With 18+ classes and ~55 images each, a random split could accidentally
  put all images of one cow in test and none in train. Stratified split
  prevents this by guaranteeing every class appears in every split.

Outputs:
  - data_processed/v1_resized/<class>/<img>.jpg
  - splits/train.csv
  - splits/val.csv
  - splits/test.csv
"""

import os
import sys
import shutil
import yaml
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── load params ────────────────────────────────────────────────────────────────
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

SEED         = params["base"]["random_seed"]
IMG_SIZE     = params["base"]["image_size"]
MIN_IMGS     = params["base"]["min_images_per_class"]
TEST_SIZE    = params["split"]["test_size"]
VAL_SIZE     = params["split"]["val_size"]

RAW_DATA_DIR = "data/raw"                      # where you put your raw images
PROCESSED_DIR = "data_processed/v1_resized"
SPLITS_DIR   = "splits"

np.random.seed(SEED)

# ── helper: resize and save a single image ─────────────────────────────────────
def resize_and_save(src_path: str, dst_path: str, size: int):
    """
    Opens an image, resizes it to size×size using LANCZOS (highest quality
    downsampling filter), and saves as JPEG.
    LANCZOS is slower than BILINEAR but preserves fine muzzle ridge textures
    better, which matters for our task.
    """
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    img = Image.open(src_path).convert("RGB")   # force 3-channel (handles grayscale edge cases)
    img = img.resize((size, size), Image.LANCZOS)
    img.save(dst_path, "JPEG", quality=95)


# ── Step 1: scan raw data, filter small classes ────────────────────────────────
print(f"\n[prepare] Scanning {RAW_DATA_DIR}/ ...")

all_records = []   # list of dicts: {filepath, label}
skipped_classes = []

for class_name in sorted(os.listdir(RAW_DATA_DIR)):
    class_dir = os.path.join(RAW_DATA_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue

    images = [
        f for f in os.listdir(class_dir)
        if f.lower().endswith((".jpg", ".jpeg"))
    ]

    if len(images) < MIN_IMGS:
        skipped_classes.append((class_name, len(images)))
        continue

    for img_file in images:
        all_records.append({
            "raw_path": os.path.join(class_dir, img_file),
            "label": class_name
        })

print(f"  Found {len(set(r['label'] for r in all_records))} valid classes")
print(f"  Total images: {len(all_records)}")
if skipped_classes:
    print(f"  Skipped {len(skipped_classes)} classes (< {MIN_IMGS} images): "
          f"{[c[0] for c in skipped_classes]}")

df = pd.DataFrame(all_records)

# Create integer label encoding (SVM and MLP need integers, CNN uses class names)
classes = sorted(df["label"].unique())
label2idx = {c: i for i, c in enumerate(classes)}
df["label_idx"] = df["label"].map(label2idx)

# Save the label mapping so all stages use the same encoding
os.makedirs(SPLITS_DIR, exist_ok=True)
label_map_df = pd.DataFrame({"label": classes, "label_idx": range(len(classes))})
label_map_df.to_csv(os.path.join(SPLITS_DIR, "label_map.csv"), index=False)
print(f"  Saved label map: {len(classes)} classes → {SPLITS_DIR}/label_map.csv")


# ── Step 2: stratified 90/10 train+val / test split ───────────────────────────
# train_test_split with stratify= ensures each class is proportionally
# represented in both halves. This is CRITICAL for small datasets.

train_val_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df["label"],
    random_state=SEED
)

# Now split train_val into train and val
# val_size here is 20% of the 90% chunk = 18% of total
train_df, val_df = train_test_split(
    train_val_df,
    test_size=VAL_SIZE,
    stratify=train_val_df["label"],
    random_state=SEED
)

print(f"\n[prepare] Split sizes:")
print(f"  Train : {len(train_df)} images ({len(train_df)/len(df)*100:.1f}%)")
print(f"  Val   : {len(val_df)} images ({len(val_df)/len(df)*100:.1f}%)")
print(f"  Test  : {len(test_df)} images ({len(test_df)/len(df)*100:.1f}%)")


# ── Step 3: resize images and save to v1_resized ──────────────────────────────
print(f"\n[prepare] Resizing images to {IMG_SIZE}×{IMG_SIZE} → {PROCESSED_DIR}/")

if os.path.exists(PROCESSED_DIR):
    shutil.rmtree(PROCESSED_DIR)   # clean slate for reproducibility

for _, row in tqdm(df.iterrows(), total=len(df), desc="Resizing"):
    src = row["raw_path"]
    # Preserve folder structure: v1_resized/<class>/<filename>
    dst = os.path.join(PROCESSED_DIR, row["label"], os.path.basename(src))
    resize_and_save(src, dst, IMG_SIZE)


# ── Step 4: update paths in splits to point to processed dir ──────────────────
def update_path(raw_path, label):
    filename = os.path.basename(raw_path)
    return os.path.join(PROCESSED_DIR, label, filename)

for split_df in [train_df, val_df, test_df]:
    split_df["filepath"] = split_df.apply(
        lambda r: update_path(r["raw_path"], r["label"]), axis=1
    )

# Save splits
train_df[["filepath", "label", "label_idx"]].to_csv(
    os.path.join(SPLITS_DIR, "train.csv"), index=False)
val_df[["filepath", "label", "label_idx"]].to_csv(
    os.path.join(SPLITS_DIR, "val.csv"), index=False)
test_df[["filepath", "label", "label_idx"]].to_csv(
    os.path.join(SPLITS_DIR, "test.csv"), index=False)

print(f"\n[prepare] ✓ Splits saved to {SPLITS_DIR}/")
print(f"[prepare] ✓ Processed images saved to {PROCESSED_DIR}/")
print("[prepare] Done.")