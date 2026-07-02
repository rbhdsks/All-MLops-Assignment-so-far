"""
train_svm.py — Stage 3a: Traditional ML model.

Why the original had low F1 (0.05-0.06):
  1. 6084-dim HOG is still very high for SVM — distances become meaningless
     in high dimensions (curse of dimensionality)
  2. Massive class imbalance (Brindavan=267 vs Merithan Cross=4) means
     SVM decision boundary tilts toward majority classes
  3. C=10 was too soft for this many classes

Fixes applied here:
  1. PCA: reduces 6084 dims → 256 dims, keeping 95%+ variance
     This makes RBF kernel distances meaningful again
  2. class_weight='balanced': automatically upweights minority classes
     so a Merithan Cross sample counts as much as a Brindavan sample
  3. C=100: tighter margin, better for this many classes

Pipeline: HOG → StandardScaler → PCA(256) → SVM(RBF, C=100, balanced)
"""

import os
import yaml
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report
import json
import pandas as pd

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

SVM_PARAMS   = params["svm"]
SEED         = params["base"]["random_seed"]
FEATURES_DIR = "features"
MODELS_DIR   = "models"
SPLITS_DIR   = "splits"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── load HOG features ──────────────────────────────────────────────────────────
print("[train_svm] Loading HOG features...")
train_data = np.load(os.path.join(FEATURES_DIR, "hog_train.npz"))
val_data   = np.load(os.path.join(FEATURES_DIR, "hog_val.npz"))

X_train, y_train = train_data["X"], train_data["y"]
X_val,   y_val   = val_data["X"],   val_data["y"]

print(f"  Train: {X_train.shape}, Val: {X_val.shape}")

# Load class names for reporting
label_map  = pd.read_csv(os.path.join(SPLITS_DIR, "label_map.csv"))
class_names = label_map["label"].tolist()

# ── build pipeline ─────────────────────────────────────────────────────────────
# Step 1 — StandardScaler: zero mean, unit variance per HOG dimension
#   Without this, PCA and SVM both behave poorly because some HOG bins
#   naturally have larger values than others
#
# Step 2 — PCA(256): projects 6084 dims down to 256 principal components
#   These 256 components capture ~95% of the variance in the data.
#   This is the single biggest fix — RBF kernel distance is meaningful
#   in 256 dims but nearly useless in 6084 dims.
#   n_components=256 is a sweet spot: enough to preserve muzzle patterns,
#   small enough for SVM to compute distances reliably.
#
# Step 3 — SVC(rbf, C=100, balanced):
#   C=100 means we penalize misclassifications heavily → tighter fit
#   class_weight='balanced' scales each class's penalty by 1/frequency,
#   so the 4-image Merithan Cross class gets 267/4 = 66x more weight
#   than Brindavan. This directly fixes the macro F1 collapse.

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca",    PCA(n_components=256, random_state=SEED, whiten=True)),
    # whiten=True: makes PCA components unit variance → better for RBF
    ("svm",    SVC(
        kernel=SVM_PARAMS["kernel"],
        C=SVM_PARAMS["C"],
        gamma=SVM_PARAMS["gamma"],
        class_weight="balanced",
        probability=False,
        random_state=SEED
    ))
])

print(f"\n[train_svm] Training: HOG → Scaler → PCA(256) → SVM(rbf, C={SVM_PARAMS['C']})")
print("  (Should take 30-90 seconds with PCA reducing dimensions)")

pipeline.fit(X_train, y_train)

# ── validate ───────────────────────────────────────────────────────────────────
val_preds = pipeline.predict(X_val)
val_f1    = f1_score(y_val, val_preds, average="macro", zero_division=0)
print(f"\n[train_svm] Validation Macro F1: {val_f1:.4f}")
print("\nPer-class F1 on validation:")
print(classification_report(y_val, val_preds,
      target_names=class_names, zero_division=0))

# ── save ───────────────────────────────────────────────────────────────────────
model_path = os.path.join(MODELS_DIR, "svm_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(pipeline, f)

metrics = {"val_macro_f1": round(float(val_f1), 4)}
with open(os.path.join(MODELS_DIR, "svm_val_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n[train_svm] Model saved → {model_path}")
print("[train_svm] Done.")