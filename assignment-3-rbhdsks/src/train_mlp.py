"""
train_mlp.py — Stage 3b: Shallow Neural Network (MLP on HOG features).

What this does:
  Builds a Multi-Layer Perceptron on top of precomputed HOG features.
  NOT raw pixels — using HOG means the network starts with meaningful
  edge/texture descriptors instead of 150K noisy pixel values.

Architecture:
  HOG features (1764-dim)
    → Linear(1764, 512) → BatchNorm → ReLU → Dropout(0.4)
    → Linear(512, 256)  → BatchNorm → ReLU → Dropout(0.4)
    → Linear(256, 128)  → BatchNorm → ReLU → Dropout(0.4)
    → Linear(128, num_classes)
    → (CrossEntropyLoss handles the softmax internally)

Why BatchNorm?
  Normalizes activations within each mini-batch. This stabilizes training
  and acts as a mild regularizer — critical when we have limited data.

Why Dropout(0.4)?
  With ~700 training samples and 18 classes, the MLP will overfit easily.
  Dropout randomly zeros 40% of neurons per forward pass during training,
  forcing the network to learn redundant representations (robust features).

Why early stopping?
  We monitor validation loss. If it doesn't improve for `patience` epochs,
  we restore the best weights and stop. This prevents overfitting past the
  sweet spot automatically.

Outputs:
  - models/mlp_model.pt
"""

import os
import yaml
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pickle

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

MLP_P        = params["mlp"]
SEED         = params["base"]["random_seed"]
FEATURES_DIR = "features"
MODELS_DIR   = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── detect device (M1 Mac uses 'mps', NVIDIA uses 'cuda', else 'cpu') ──────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("[train_mlp] Using Apple M1 MPS (Metal Performance Shaders)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("[train_mlp] Using CUDA GPU")
else:
    DEVICE = torch.device("cpu")
    print("[train_mlp] Using CPU")


# ── load and scale HOG features ────────────────────────────────────────────────
print("[train_mlp] Loading HOG features...")
train_data = np.load(os.path.join(FEATURES_DIR, "hog_train.npz"))
val_data   = np.load(os.path.join(FEATURES_DIR, "hog_val.npz"))

X_train_raw, y_train = train_data["X"], train_data["y"]
X_val_raw,   y_val   = val_data["X"],   val_data["y"]

# Scale features (same reason as SVM — neural nets are sensitive to scale)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
X_val   = scaler.transform(X_val_raw).astype(np.float32)

# Save scaler — needed in evaluate.py to transform test features
with open(os.path.join(MODELS_DIR, "mlp_scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

num_features  = X_train.shape[1]
num_classes   = len(np.unique(y_train))
print(f"  Features: {num_features}, Classes: {num_classes}")

# ── convert to PyTorch tensors and DataLoaders ─────────────────────────────────
# TensorDataset wraps numpy arrays as a PyTorch dataset
train_dataset = TensorDataset(
    torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
val_dataset   = TensorDataset(
    torch.tensor(X_val),   torch.tensor(y_val,   dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=MLP_P["batch_size"],
                          shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=MLP_P["batch_size"],
                          shuffle=False)


# ── define MLP architecture ────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_layers, num_classes, dropout):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for h in hidden_layers:
            layers += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),   # normalize activations — stabilizes training
                nn.ReLU(),
                nn.Dropout(dropout)  # regularize — prevents overfitting
            ]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


model = MLP(
    in_dim=num_features,
    hidden_layers=MLP_P["hidden_layers"],
    num_classes=num_classes,
    dropout=MLP_P["dropout"]
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=MLP_P["learning_rate"],
                              weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()  # includes softmax — standard for multiclass


# ── training loop with early stopping ─────────────────────────────────────────
print(f"\n[train_mlp] Training for up to {MLP_P['epochs']} epochs "
      f"(patience={MLP_P['early_stopping_patience']})...")

best_val_loss  = float("inf")
patience_count = 0
best_weights   = None
history        = []

for epoch in range(1, MLP_P["epochs"] + 1):

    # ── train phase ──
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(X_batch)
    train_loss /= len(train_dataset)

    # ── val phase ──
    model.eval()
    val_loss  = 0.0
    all_preds = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            val_loss += loss.item() * len(X_batch)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
    val_loss /= len(val_dataset)

    from sklearn.metrics import f1_score
    val_f1 = f1_score(y_val, all_preds, average="macro", zero_division=0)
    history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                    "val_loss": round(val_loss, 4), "val_macro_f1": round(val_f1, 4)})

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_f1={val_f1:.4f}")

    # ── early stopping check ──
    if val_loss < best_val_loss:
        best_val_loss  = val_loss
        patience_count = 0
        best_weights   = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_count += 1
        if patience_count >= MLP_P["early_stopping_patience"]:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {patience_count} epochs)")
            break

# Restore the best weights (not the last — the last may have overfit)
model.load_state_dict(best_weights)
print(f"\n[train_mlp] Best val loss: {best_val_loss:.4f}")

# ── save model + metadata ──────────────────────────────────────────────────────
torch.save({
    "model_state_dict": model.state_dict(),
    "in_dim": num_features,
    "hidden_layers": MLP_P["hidden_layers"],
    "num_classes": num_classes,
    "dropout": MLP_P["dropout"]
}, os.path.join(MODELS_DIR, "mlp_model.pt"))

with open(os.path.join(MODELS_DIR, "mlp_history.json"), "w") as f:
    json.dump(history, f, indent=2)

metrics = {"val_macro_f1": round(float(max(h["val_macro_f1"] for h in history)), 4)}
with open(os.path.join(MODELS_DIR, "mlp_val_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(f"[train_mlp] ✓ Model saved → {MODELS_DIR}/mlp_model.pt")
print("[train_mlp] Done.")