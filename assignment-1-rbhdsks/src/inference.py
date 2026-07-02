import joblib
import yaml
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import os

# Load config
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()

# Load model
model_path = "models/model_v1.pkl"
model = joblib.load(model_path)

app = FastAPI()

# Log deployment when server starts
deployment_log = "deployment_log.csv"
if os.path.exists(deployment_log):
    with open(deployment_log, "a") as f:
        f.write(f"{datetime.now()},model_v1.pkl\n")

# Define input format (adjust fields if needed)
class InputData(BaseModel):
    Type: int
    Air_temperature_K: float
    Process_temperature_K: float
    Rotational_speed_rpm: float
    Torque_Nm: float
    Tool_wear_min: float


@app.post("/predict")
def predict(data: InputData):
    input_dict = data.dict()
    df = pd.DataFrame([input_dict])

    # Ensure all training columns exist
    training_columns = model.feature_names_in_

    for col in training_columns:
        if col not in df.columns:
            df[col] = 0  # fill missing columns with 0

    df = df[training_columns]  # correct column order

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }

@app.get("/")
def home():
    return {"message": "Manual MLOps API is running"}
