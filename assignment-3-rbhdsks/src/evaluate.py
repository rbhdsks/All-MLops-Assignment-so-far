"""
evaluate.py — Stage 4: Final evaluation on the held-out test set.

THIS IS THE ONLY PLACE THE TEST SET IS USED.

Outputs:
  - metrics/metrics.json       <- dvc metrics show reads this
  - metrics/cnn_history.csv    <- dvc plots reads this
  - metrics/svm_results.csv
  - metrics/mlp_history.csv
  - metrics/all_models_f1.csv
"""

import os
import sys

# Make sure src/ is in path so we can import MLP class from train_mlp.py
# This works whether called as 'python src/evaluate.py' or 'PYTHONPATH=src python src/evaluate.py'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import json
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm
from sklearn.metrics import f1_score, classification_report

# ── load params ────────────────────────────────────────────────────────────────
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

SPLITS_DIR  = "splits"
MODELS_DIR  = "models"
METRICS_DIR = "metrics"
os.makedirs(METRICS_DIR, exist_ok=True)

# ── device ─────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"[evaluate] Using device: {DEVICE}")

# ── load test split ────────────────────────────────────────────────────────────
test_df     = pd.read_csv(os.path.join(SPLITS_DIR, "test.csv"))
label_map   = pd.read_csv(os.path.join(SPLITS_DIR, "label_map.csv"))
num_classes = len(label_map)
class_names = label_map["label"].tolist()

print(f"[evaluate] Test set: {len(test_df)} images, {num_classes} classes")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1: SVM
# ══════════════════════════════════════════════════════════════════════════════
print("\n[evaluate] ── SVM ──")
with open(os.path.join(MODELS_DIR, "svm_model.pkl"), "rb") as f:
    svm_pipeline = pickle.load(f)

test_hog           = np.load(os.path.join("features", "hog_test.npz"))
X_test_hog, y_test = test_hog["X"], test_hog["y"]

svm_preds = svm_pipeline.predict(X_test_hog)
svm_f1    = f1_score(y_test, svm_preds, average="macro", zero_division=0)
print(f"  Test Macro F1: {svm_f1:.4f}")
print(classification_report(y_test, svm_preds,
      zero_division=0))

svm_per_class = f1_score(y_test, svm_preds, average=None, zero_division=0, labels=list(range(num_classes)))
svm_plot_df   = pd.DataFrame({"class": class_names, "f1": svm_per_class, "model": "SVM"})
svm_plot_df.to_csv(os.path.join(METRICS_DIR, "svm_results.csv"), index=False)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2: MLP
# ══════════════════════════════════════════════════════════════════════════════
print("\n[evaluate] ── MLP ──")

with open(os.path.join(MODELS_DIR, "mlp_scaler.pkl"), "rb") as f:
    mlp_scaler = pickle.load(f)

X_test_scaled = mlp_scaler.transform(X_test_hog).astype(np.float32)

# Import MLP class — sys.path.insert above ensures this works
from train_mlp import MLP

ckpt = torch.load(os.path.join(MODELS_DIR, "mlp_model.pt"), map_location=DEVICE)
mlp_model = MLP(
    in_dim=ckpt["in_dim"],
    hidden_layers=ckpt["hidden_layers"],
    num_classes=ckpt["num_classes"],
    dropout=ckpt["dropout"]
).to(DEVICE)
mlp_model.load_state_dict(ckpt["model_state_dict"])
mlp_model.eval()

with torch.no_grad():
    X_tensor  = torch.tensor(X_test_scaled).to(DEVICE)
    mlp_preds = mlp_model(X_tensor).argmax(dim=1).cpu().numpy()

mlp_f1 = f1_score(y_test, mlp_preds, average="macro", zero_division=0)
print(f"  Test Macro F1: {mlp_f1:.4f}")
print(classification_report(y_test, mlp_preds,
      zero_division=0))

mlp_per_class = f1_score(y_test, mlp_preds, average=None, zero_division=0, labels=list(range(num_classes)))
mlp_plot_df   = pd.DataFrame({"class": class_names, "f1": mlp_per_class, "model": "MLP"})
mlp_plot_df.to_csv(os.path.join(METRICS_DIR, "mlp_results.csv"), index=False)

mlp_history = json.load(open(os.path.join(MODELS_DIR, "mlp_history.json")))
pd.DataFrame(mlp_history).to_csv(os.path.join(METRICS_DIR, "mlp_history.csv"), index=False)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3: EfficientNet CNN
# ══════════════════════════════════════════════════════════════════════════════
print("\n[evaluate] ── EfficientNet CNN ──")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

class MuzzleDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df        = pd.read_csv(csv_path)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(row["filepath"]).convert("RGB")
        label = int(row["label_idx"])
        if self.transform:
            img = self.transform(img)
        return img, label

test_dataset = MuzzleDataset(os.path.join(SPLITS_DIR, "test.csv"),
                              transform=val_transform)
test_loader  = DataLoader(test_dataset, batch_size=32,
                          shuffle=False, num_workers=0)

cnn_ckpt  = torch.load(os.path.join(MODELS_DIR, "cnn_best.pt"), map_location=DEVICE)
cnn_model = timm.create_model(
    cnn_ckpt["model_name"],
    pretrained=False,
    num_classes=cnn_ckpt["num_classes"]
).to(DEVICE)
cnn_model.load_state_dict(cnn_ckpt["model_state_dict"])
cnn_model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        preds  = cnn_model(images).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

cnn_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
print(f"  Test Macro F1: {cnn_f1:.4f}")
print(classification_report(all_labels, all_preds,
      zero_division=0))

cnn_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0, labels=list(range(num_classes)))
cnn_plot_df   = pd.DataFrame({"class": class_names, "f1": cnn_per_class,
                               "model": "EfficientNet"})
cnn_plot_df.to_csv(os.path.join(METRICS_DIR, "cnn_results.csv"), index=False)

cnn_history = json.load(open(os.path.join(MODELS_DIR, "cnn_history.json")))
if isinstance(cnn_history, list):
    pd.DataFrame(cnn_history).to_csv(os.path.join(METRICS_DIR, "cnn_history.csv"), index=False)
else:
    pd.DataFrame([cnn_history]).to_csv(os.path.join(METRICS_DIR, "cnn_history.csv"), index=False)


# ══════════════════════════════════════════════════════════════════════════════
# SAVE FINAL METRICS
# ══════════════════════════════════════════════════════════════════════════════
scores = {"svm": svm_f1, "mlp": mlp_f1, "efficientnet": cnn_f1}
metrics_out = {
    "svm":          {"test_macro_f1": round(float(svm_f1), 4)},
    "mlp":          {"test_macro_f1": round(float(mlp_f1), 4)},
    "efficientnet": {"test_macro_f1": round(float(cnn_f1), 4)},
    "best_model":   max(scores, key=scores.get)
}
with open(os.path.join(METRICS_DIR, "metrics.json"), "w") as f:
    json.dump(metrics_out, f, indent=2)

combined_df = pd.concat([svm_plot_df, mlp_plot_df, cnn_plot_df])
combined_df.to_csv(os.path.join(METRICS_DIR, "all_models_f1.csv"), index=False)

print("\n" + "="*60)
print("FINAL TEST RESULTS")
print("="*60)
print(f"  SVM          Macro F1: {svm_f1:.4f}")
print(f"  MLP          Macro F1: {mlp_f1:.4f}")
print(f"  EfficientNet Macro F1: {cnn_f1:.4f}")
print(f"\n  Best model: {metrics_out['best_model']}")
print("="*60)
print(f"\n[evaluate] Metrics saved -> {METRICS_DIR}/metrics.json")
print("[evaluate] Done.")