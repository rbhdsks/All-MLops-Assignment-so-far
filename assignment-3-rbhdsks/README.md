# Cattle Muzzle Biometrics | Assignment 03-04
### Nitesh Kumar Shah | ID25M806

A reproducible machine learning pipeline for individual cattle identification via muzzle imagery. Built with DVC for end-to-end experiment tracking and reproducibility.

---

## Overview

Cattle muzzle patterns are anatomically unique — analogous to human fingerprints. This project implements a three-model classification pipeline that identifies individual cattle from muzzle photographs, with every experiment fully version-controlled and reproducible via a single command.

| Model | Architecture | Test Macro F1 |
|-------|-------------|:-------------:|
| Baseline | HOG + PCA(256) + SVM | 0.0974 |
| Intermediate | HOG + MLP (512→256→128) | 0.1168 |
| **Primary** | **EfficientNet-B0 (fine-tuned)** | **0.4293** |

---

## Requirements

```
Python 3.10+    PyTorch 2.1+    DVC 3.30+    timm 0.9+
scikit-learn    scikit-image    torchvision   pandas
```

```bash
pip install -r requirements.txt
```

---

## Quickstart

```bash
git clone https://github.com/DA5402-MLOps-JAN26/assignment-3-rbhdsks.git
cd assignment-3-rbhdsks
pip install -r requirements.txt
dvc repro
```

`dvc repro` executes the complete pipeline — data preparation, feature extraction, model training, and evaluation — in dependency order, skipping any stage whose inputs have not changed.

---

## Pipeline

The experiment is defined as a five-stage DVC pipeline in `dvc.yaml`. All hyperparameters are centralised in `params.yaml`.

```
data/raw
   │
   ▼
prepare ──────────────────────────────────────────────┐
   │                                                   │
   ▼                                                   ▼
transform                                          train_cnn
   │                                                   │
   ├──► train_svm                                      │
   │                                                   │
   └──► train_mlp                                      │
              │                                        │
              └──────────────┬─────────────────────────┘
                             ▼
                          evaluate
```

| Stage | Input | Output | Triggered by |
|-------|-------|--------|-------------|
| `prepare` | `data/raw` | Resized images, split CSVs | image size, split ratio |
| `transform` | Split CSVs | HOG feature arrays | HOG cell size, orientations |
| `train_svm` | HOG features | `svm_model.pkl` | SVM kernel, C, gamma |
| `train_mlp` | HOG features | `mlp_model.pt` | layers, dropout, lr |
| `train_cnn` | Resized images | `cnn_best.pt` | all CNN + augmentation params |
| `evaluate` | All three models | `metrics.json`, plot CSVs | any model change |

Changing a single parameter in `params.yaml` re-runs only the affected downstream stages.

---

## Data

**19 classes · 1,084 images · split 72 / 18 / 10**

The raw dataset contains 27 cattle identity folders. Eight classes with fewer than three images are excluded per the task specification. The remaining 19 classes exhibit significant imbalance — from 267 images (Brindavan) to 4 images (Merithan Cross).

**Split strategy**

Classes with ≥ 10 images use stratified sampling to preserve class proportions across splits. Classes with < 10 images use a manual allocation (1 image to test, 1 to validation, remainder to train) guaranteeing every class appears in every split.

```
Train   780 images   72%
Val     195 images   18%
Test    109 images   10%   ← held out until final evaluation
```

**Versioning**

```bash
git tag data-v1    # raw dataset
git tag data-v2    # processed + HOG features
```

---

## Models

### SVM

HOG descriptors (9 orientations, 16×16 cells, 2×2 block normalisation) produce a 6,084-dimensional feature vector per image. A `StandardScaler → PCA(256, whiten=True) → SVM(RBF, C=100)` pipeline is trained with balanced class weights.

PCA reduces dimensionality from 6,084 to 256 before passing features to the SVM. This is necessary because RBF kernel distances become uninformative in high-dimensional spaces.

### MLP

The same HOG features are passed to a three-hidden-layer network (512 → 256 → 128 → 19) with BatchNorm, ReLU, and Dropout(0.4) at each layer. Training uses Adam with early stopping (patience = 10) on validation loss.

### EfficientNet-B0

A pre-trained EfficientNet-B0 is fine-tuned in two phases to prevent catastrophic forgetting of ImageNet features.

**Phase 1 — head warm-up (3 epochs)**
The backbone is frozen. Only the 19-class classification head is trained at lr = 1e-3 until it stabilises from random initialisation.

**Phase 2 — full fine-tuning (27 epochs)**
The backbone is unfrozen with a differential learning rate: backbone at 1e-4, head at 1e-3. A cosine annealing schedule decays the learning rate to 1e-6 over the remaining epochs.

**Augmentation** (applied online, every epoch)

| Transform | Parameters | Purpose |
|-----------|-----------|---------|
| RandomHorizontalFlip | p = 0.5 | Bilateral symmetry of muzzle ridges |
| RandAugment | n = 2, magnitude = 9 | Generalisation on small datasets |
| RandomErasing | p = 0.2, scale 2–20% | Simulates mud and occlusion |
| Normalise | ImageNet mean/std | Required for pre-trained backbone |

Label smoothing (ε = 0.1) prevents overconfident predictions on the few-shot classes.

---

## Evaluation

**Metric: Macro F1**

Macro F1 averages per-class F1 scores with equal weight across all 19 classes. This penalises models that perform well on the majority class while ignoring rare identities — the correct choice for a biometric identification task where every individual matters equally.

**Per-class results (EfficientNet-B0, test set)**

Classes with ≥ 6 test images achieve F1 ≥ 0.44. Six classes with ≤ 3 training samples score F1 = 0.00 across all models — this reflects a data constraint, not a modelling failure. The weighted F1 of 0.65 (accounting for class size) indicates strong performance where sufficient data exists.

```bash
dvc metrics show        # tabulated results across all models
dvc metrics diff        # compare experiments
dvc plots show metrics/all_models_f1.csv    # per-class F1 chart
dvc plots show metrics/cnn_history.csv      # training curve
```

---

## Reproducing a Specific Experiment

```bash
# Reproduce the final submitted experiment
git checkout experiment-v1
dvc checkout
dvc repro

# Run a single stage
dvc repro train_cnn

# Check what would re-run without executing
dvc status

# View the dependency graph
dvc dag
```

---

## Repository Structure

```
.
├── data/raw/                   # raw images (DVC-tracked)
├── data_processed/
│   ├── v1_resized/             # 224×224 images (DVC-tracked)
│   └── v2_augmented/           # augmentation config
├── features/                   # HOG arrays (DVC-tracked)
├── models/                     # saved checkpoints and scalers
├── metrics/                    # evaluation outputs and plot CSVs
├── splits/                     # train / val / test CSVs and label map
├── src/
│   ├── prepare.py              # stage 1
│   ├── transform.py            # stage 2
│   ├── train_svm.py            # stage 3a
│   ├── train_mlp.py            # stage 3b
│   ├── train_cnn.py            # stage 3c
│   ├── evaluate.py             # stage 4
│   └── generate_plots.py       # report figures
├── dvc.yaml                    # pipeline definition
├── dvc.lock                    # locked file hashes
└── params.yaml                 # all hyperparameters
```

---

## Acknowledgements

AI assistance (Claude, Anthropic) was used for boilerplate code — DataLoader setup, HOG extraction syntax, and model loading via `timm`. All such blocks are marked with `# Assisted by Claude AI` in the source. Pipeline architecture, split strategy, and modelling decisions were made independently.
Also the AI Disclosure report is added to the report file as well...

---

*DA5402 · MLOps · March 2026 · IIT Madras*
