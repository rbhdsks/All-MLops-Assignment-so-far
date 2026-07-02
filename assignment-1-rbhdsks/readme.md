# Assignment 01
Manual MLOps Implementation  
Nitesh Kumar Shah | ID25M806
Demo video link: https://drive.google.com/file/d/1hNahSbfVpyo6naHOfCIEO-ovLoGgNsMS/view?usp=sharing
## Project Overview

This project implements a **complete Machine Learning lifecycle** using only primitive tools such as:

- Python scripts
- CSV files
- Git
- FastAPI
- YAML configuration
- Text-based logs

No MLOps automation tools (MLflow, DVC, Docker, Airflow, Kubernetes) were used.

The objective of this project is to manually simulate the responsibilities of an ML Engineer in a startup environment without infrastructure support.

---

# Business Problem

We simulate a startup building a **Predictive Maintenance System**.

Goal:
Predict whether a machine will fail based on sensor readings.

Output:
Binary classification  
- `0` → No failure  
- `1` → Failure  

Dataset:
AI4I 2020 Predictive Maintenance Dataset (10,000 chronological records)

---

# System Architecture (Manual MLOps Design)

```
Raw Data
   ↓
Data Processing (v1.csv)
   ↓
Model Training (model_v1.pkl)
   ↓
Manual Model Registry (model_metadata.log)
   ↓
API Deployment (FastAPI)
   ↓
Deployment Log (deployment_log.csv)
   ↓
Production Drift Simulation (day2.csv)
   ↓
Monitoring Script (monitor.py)
   ↓
Manual Retraining Trigger
```

---

# Directory Structure

```
manual_mlops_project/
│
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv
│   │
│   ├── processed/
│   │   ├── v1.csv
│   │   └── manifest.txt
│   │
│   └── production/
│       └── day2.csv
│
├── models/
│   ├── model_v1.pkl
│   └── model_metadata.log
│
├── src/
|.  ├── create_day2.py
│   ├── data_prep.py
│   ├── train.py
│   ├── inference.py
│   ├── test_api.py
│   └── monitor.py
│
├── config.yaml
├── deployment_log.csv
├── requirements.txt
└── README.md
```

---

# Phase A – Manual Data Versioning

## Raw Data
The original dataset is stored in:

```
data/raw/ai4i2020.csv
```

This file is immutable and never modified.

## Processed Data (v1)

Generated using:

```
src/data_prep.py
```

Transformations:
- Dropped identifier columns (`UDI`, `Product ID`)
- Label encoded `Type`
- Chronological split (first 7000 rows used for training)
- No shuffling applied

Saved as:

```
data/processed/v1.csv
```

## Manifest File

All dataset versions are recorded in:

```
data/processed/manifest.txt
```

Each entry includes:
- Version name
- Source file
- Script used
- Transformations applied
- Data shape
- Purpose

This ensures manual data lineage tracking.

---

# Phase B – Model Training & Manual Registry

## Training Script

```
src/train.py
```

Model:
RandomForestClassifier

Training Process:
- Loads processed dataset (v1.csv)
- Splits into train/validation
- Computes validation accuracy
- Saves model artifact

## Model Artifact

```
models/model_v1.pkl
```

## Manual Model Registry

```
models/model_metadata.log
```

Each training event logs:
- Model name
- Dataset used
- Accuracy
- Training timestamp

This simulates a manual MLflow-like registry.

---

# Phase C – Deployment

## API Framework

FastAPI is used to serve predictions.

Start server:

```
uvicorn src.inference:app --reload
```

Access API Docs:

```
http://127.0.0.1:8000/docs
```

## Endpoint

POST `/predict`

Returns:

```json
{
  "prediction": 0,
  "probability": 0.12
}
```

## Deployment Logging

Each server startup appends to:

```
deployment_log.csv
```

Logged fields:
- Timestamp
- Active model version

This simulates deployment tracking.

---

# Smoke Testing

Script:

```
src/test_api.py
```

Tests:
- API availability
- Correct JSON format
- Valid prediction output
- Probability range validation

Run:

```
python src/test_api.py
```

---

# Phase D – Drift Simulation

Script:

```
src/day2_generation.py
```

Process:
- Select rows 7000–8000 (future timeframe)
- Apply mild temperature shift (+1K)
- Apply slight torque scaling (×1.01)
- Add Gaussian noise
- Save as:

```
data/production/day2.csv
```

This simulates production drift.

Entry is recorded in `manifest.txt`.

---

# Monitoring & Retraining

Script:

```
src/monitor.py
```

Process:
- Loads production dataset
- Sends data to API
- Computes production error rate
- Compares with training performance
- Triggers retraining alert if threshold exceeded

Threshold defined in:

```
config.yaml
```

---

# Configuration Management

All hyperparameters and paths are stored in:

```
config.yaml
```

No hardcoded paths exist in scripts.

This ensures reproducibility and isolation of configuration from logic.

---

# Reproducibility Instructions

1. Clone repository
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Generate processed data:

```
python src/data_prep.py
```

4. Train model:

```
python src/train.py
```

5. Run API:

```
uvicorn src.inference:app --reload
```

6. Test API:

```
python src/test_api.py
```

---

# Environment Reproducibility

Dependencies are pinned in:

```
requirements.txt
```

This ensures consistent model behavior across machines.

---

# Manual MLOps Challenges Observed

1. Managing data versions manually is error-prone.
2. Tracking which model version is deployed requires discipline.
3. Feature mismatches can easily break inference.
4. Manual drift detection requires explicit scripting.
5. Deployment state must be logged explicitly.

---

# Why Automated MLOps Tools Matter

This project demonstrates the limitations of manual systems:

| Manual Process | Automated Tool Equivalent |
|---------------|--------------------------|
| manifest.txt | DVC |
| model_metadata.log | MLflow |
| deployment_log.csv | CI/CD + Model Registry |
| monitor.py | Production Monitoring Systems |
| manual retraining | Airflow / Kubeflow |

Automation reduces:
- Human error
- Cognitive load
- Deployment mistakes
- Version confusion

---



# Conclusion

This project successfully implements:

- Manual data versioning
- Manual model registry
- API deployment
- Deployment logging
- Drift simulation
- Monitoring and retraining trigger

All using primitive tools.

It demonstrates why automated MLOps systems are necessary in real-world production environments.

---





