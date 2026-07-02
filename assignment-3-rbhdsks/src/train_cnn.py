"""
train_cnn.py — Stage 3c: Transfer Learning with EfficientNet-B0.

Key improvements for higher F1:
  1. WeightedRandomSampler — fixes class imbalance (biggest improvement)
  2. 60 epochs + patience 15 — more time to converge
  3. Stronger augmentation — ColorJitter + rotation on top of RandAugment
  4. Batch size 16 — more gradient updates per epoch for rare classes
  5. Saves model after EVERY improvement in val_f1 — never loses best weights
"""

import os
import yaml
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import timm  # Assisted by Claude AI
from sklearn.metrics import f1_score
from collections import Counter

# ── params ─────────────────────────────────────────────────────────────────────
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

CNN_P      = params["cnn"]
AUG_P      = params["augmentation"]
SEED       = params["base"]["random_seed"]
SPLITS_DIR = "splits"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── device ─────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("[train_cnn] Using Apple M1 MPS")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
    print("[train_cnn] Using CPU")

# ── dataset ────────────────────────────────────────────────────────────────────
class MuzzleDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df        = pd.read_csv(csv_path)
        self.transform = transform
        self.labels    = self.df["label_idx"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(row["filepath"]).convert("RGB")
        label = int(row["label_idx"])
        if self.transform:
            img = self.transform(img)
        return img, label

# ── transforms ─────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=float(AUG_P["horizontal_flip_prob"])),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.4, contrast=0.4,
                           saturation=0.3, hue=0.1),
    transforms.RandAugment(num_ops=int(AUG_P["randaugment_n"]),
                           magnitude=int(AUG_P["randaugment_m"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=float(AUG_P["random_erasing_prob"]),
                             scale=(0.02, 0.2), ratio=(0.3, 3.3))
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# ── datasets ───────────────────────────────────────────────────────────────────
train_dataset = MuzzleDataset(os.path.join(SPLITS_DIR, "train.csv"),
                               transform=train_transform)
val_dataset   = MuzzleDataset(os.path.join(SPLITS_DIR, "val.csv"),
                               transform=val_transform)
num_classes   = len(pd.read_csv(os.path.join(SPLITS_DIR, "label_map.csv")))

print(f"[train_cnn] {num_classes} classes | "
      f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

# ── weighted sampler ───────────────────────────────────────────────────────────
# Each class gets weight = 1/count so rare classes appear as often as common ones
# This is the single biggest fix for our imbalanced dataset
# Without this: a batch might have 20 Brindavan, 0 Merithan Cross
# With this: every class gets ~batch_size/num_classes samples per batch
labels_list   = train_dataset.labels.tolist()
class_counts  = Counter(labels_list)
class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
sample_weights = [class_weights[l] for l in labels_list]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(train_dataset,
                          batch_size=int(CNN_P["batch_size"]),
                          sampler=sampler,
                          num_workers=0,
                          pin_memory=False)
val_loader   = DataLoader(val_dataset,
                          batch_size=int(CNN_P["batch_size"]),
                          shuffle=False,
                          num_workers=0,
                          pin_memory=False)

# ── model ──────────────────────────────────────────────────────────────────────
print(f"\n[train_cnn] Loading {CNN_P['model_name']} pretrained=True ...")
model = timm.create_model(
    CNN_P["model_name"],
    pretrained=bool(CNN_P["pretrained"]),
    num_classes=num_classes
).to(DEVICE)

# ── helpers ────────────────────────────────────────────────────────────────────
def set_backbone_grad(model, requires_grad):
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = requires_grad

criterion = nn.CrossEntropyLoss(
    label_smoothing=float(CNN_P["label_smoothing"]))

def run_epoch(model, loader, optimizer, is_train):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if is_train:
                optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(images)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    f1       = f1_score(all_labels, all_preds,
                        average="macro", zero_division=0)
    return avg_loss, f1

# ── phase 1: head only ─────────────────────────────────────────────────────────
freeze_epochs = int(CNN_P["freeze_epochs"])
print(f"\n[train_cnn] Phase 1: head only for {freeze_epochs} epochs ...")
set_backbone_grad(model, requires_grad=False)

opt1 = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=float(CNN_P["learning_rate_head"]),
    weight_decay=float(CNN_P["weight_decay"])
)

history      = []
best_val_f1  = -1.0
best_weights = None

for epoch in range(1, freeze_epochs + 1):
    tr_loss, tr_f1 = run_epoch(model, train_loader, opt1, is_train=True)
    vl_loss, vl_f1 = run_epoch(model, val_loader,   None, is_train=False)
    print(f"  [Phase1] Epoch {epoch:2d} | "
          f"train_f1={tr_f1:.4f} | val_f1={vl_f1:.4f}")
    history.append({"phase": 1, "epoch": epoch,
                    "train_f1": round(tr_f1, 4),
                    "val_f1":   round(vl_f1, 4),
                    "val_loss": round(vl_loss, 4)})
    if vl_f1 > best_val_f1:
        best_val_f1  = vl_f1
        best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        # Save immediately — never lose best weights
        torch.save({"model_state_dict": best_weights,
                    "model_name": CNN_P["model_name"],
                    "num_classes": num_classes},
                   os.path.join(MODELS_DIR, "cnn_best.pt"))

# ── phase 2: full fine-tune ────────────────────────────────────────────────────
total_epochs   = int(CNN_P["total_epochs"])
remaining      = total_epochs - freeze_epochs
patience_limit = int(CNN_P["early_stopping_patience"])

print(f"\n[train_cnn] Phase 2: full fine-tune for up to {remaining} epochs ...")
set_backbone_grad(model, requires_grad=True)

opt2 = torch.optim.Adam([
    {"params": [p for n, p in model.named_parameters()
                if "classifier" not in n],
     "lr": float(CNN_P["learning_rate_backbone"])},
    {"params": [p for n, p in model.named_parameters()
                if "classifier" in n],
     "lr": float(CNN_P["learning_rate_head"])}
], weight_decay=float(CNN_P["weight_decay"]))

scheduler      = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt2, T_max=remaining, eta_min=1e-6)
patience_count = 0

for epoch in range(1, remaining + 1):
    tr_loss, tr_f1 = run_epoch(model, train_loader, opt2,  is_train=True)
    vl_loss, vl_f1 = run_epoch(model, val_loader,   None,  is_train=False)
    scheduler.step()
    lr_now = scheduler.get_last_lr()[0]

    print(f"  [Phase2] Epoch {epoch:2d}/{remaining} | "
          f"train_f1={tr_f1:.4f} | val_f1={vl_f1:.4f} | lr={lr_now:.6f}")
    history.append({"phase": 2, "epoch": epoch,
                    "train_f1": round(tr_f1, 4),
                    "val_f1":   round(vl_f1, 4),
                    "val_loss": round(vl_loss, 4),
                    "lr":       round(lr_now, 8)})

    if vl_f1 > best_val_f1:
        best_val_f1    = vl_f1
        patience_count = 0
        best_weights   = {k: v.clone() for k, v in model.state_dict().items()}
        # Save immediately on every improvement
        torch.save({"model_state_dict": best_weights,
                    "model_name": CNN_P["model_name"],
                    "num_classes": num_classes},
                   os.path.join(MODELS_DIR, "cnn_best.pt"))
        print(f"    --> New best! val_f1={best_val_f1:.4f} saved.")
    else:
        patience_count += 1
        if patience_count >= patience_limit:
            print(f"\n  Early stopping at epoch {epoch}")
            break

print(f"\n[train_cnn] Best Validation Macro F1: {best_val_f1:.4f}")

# ── save history and metrics ───────────────────────────────────────────────────
with open(os.path.join(MODELS_DIR, "cnn_history.json"), "w") as f:
    json.dump(history, f, indent=2)

with open(os.path.join(MODELS_DIR, "cnn_val_metrics.json"), "w") as f:
    json.dump({"val_macro_f1": round(float(best_val_f1), 4)}, f, indent=2)

print(f"[train_cnn] Model saved -> {MODELS_DIR}/cnn_best.pt")
print("[train_cnn] Done.")