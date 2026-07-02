# DA5402 A7: MLflow Sprint — MNIST Classifier
## Nitesh Kumar Shah 
## ID25M806

An end-to-end MLflow pipeline: training a ResNet18-based MNIST classifier, tracking experiments, packaging as an MLflow Project, serving the best model, and exposing it via a Streamlit web client.

## Project Structure

```
mlflow-sprint/
├── MLproject              # MLflow Project definition
├── python_env.yaml        # Python environment spec
├── requirements.txt       # Python dependencies
├── src/
│   └── train.py           # Training script with MLflow logging
├── app.py                 # Streamlit web client
├── screenshots/           # MLflow UI screenshots
└── README.md
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Train (via MLflow Projects)

```bash
mlflow run . -P learning_rate=0.005 -P batch_size=64 -P epochs=2 --env-manager=local
```

Configurable parameters: `learning_rate`, `batch_size`, `epochs`.

## View Experiments

```bash
mlflow ui --port 5555
```

Then open [http://localhost:5555](http://localhost:5555).

## Serve the Best Model

Pick the best run from the MLflow UI and copy its Run ID:

```bash
mlflow models serve -m "runs:/<RUN_ID>/model" --port 5001 --env-manager=local
```

## Run the Web Client

With the model server running on port 5001:

```bash
streamlit run app.py
```

Upload a digit image, click **Classify**, and see the prediction along with confidence scores and a probability bar chart.

## MLflow Logging Details

The training script (`src/train.py`) logs the following to MLflow:

- **Parameters**: `learning_rate`, `batch_size`, `epochs`, `architecture`, `optimizer`
- **Metrics** (per epoch): `train_loss`, `val_loss`, `val_accuracy`
- **Artifacts**: sample prediction plot (`sample_predictions.png`), trained PyTorch model with input signature

## How `runs:/` URIs Map to the Filesystem

MLflow resolves `runs:/<run_id>/model` by looking up the run in its tracking store (here, `mlflow.db` — a local SQLite file) and reading the `artifact_uri` column for that run. Artifacts are stored under `./mlruns/<experiment_id>/<run_id>/artifacts/`, so `runs:/<run_id>/model` maps to `./mlruns/0/<run_id>/artifacts/model/`, which contains the serialized PyTorch model, the `MLmodel` metadata file, `conda.yaml`, and `requirements.txt`.

## Experiment Results

| Run | Learning Rate | Batch Size | Epochs | Val Accuracy |
|-----|--------------|------------|--------|-------------|
| 1   | 0.01         | 64         | 1      | 0.8930      |
| 2   | 0.01         | 64         | 2      | 0.9490      |
| 3   | 0.001        | 32         | 2      | 0.9510      |
| 4   | 0.1          | 128        | 2      | 0.8680      |
| 5   | 0.005        | 64         | 2      | 0.9530      |
| 6   | 0.005        | 64         | 2      | 0.9500      |

Best run: `lr=0.005, bs=64, epochs=2` with **val_accuracy = 0.9530**.

High learning rate (0.1) significantly hurt performance, while moderate rates (0.001–0.01) converged well within 2 epochs.

## Environment

- macOS (Apple Silicon M1)
- Python 3.11
- MLflow 3.x
- PyTorch 2.x (MPS backend)
- Streamlit 1.x