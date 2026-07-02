import os
import yaml
import pandas as pd
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    processed_path = config["data"]["processed_path"]
    model_params = config["model"]

    # Load processed data
    df = pd.read_csv(processed_path)

    print(f"Loaded processed data: {df.shape}")

    # Separate features and target
    target_column = "Machine failure"  # check exact column name
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split train/validation (simple split)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=model_params["random_state"]
    )

    # Train model
    model = RandomForestClassifier(
        n_estimators=model_params["n_estimators"],
        random_state=model_params["random_state"]
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    print(f"Validation Accuracy: {accuracy:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/model_v1.pkl"
    joblib.dump(model, model_path)

    print(f"Model saved at {model_path}")

    # Save metadata
    metadata_path = "models/model_metadata.log"

    with open(metadata_path, "a") as f:
        f.write("\n")
        f.write(f"Model: model_v1.pkl\n")
        f.write(f"Data Used: {processed_path}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Trained On: {datetime.now()}\n")
        f.write("-" * 40 + "\n")

    print("Model metadata logged.")


if __name__ == "__main__":
    main()

