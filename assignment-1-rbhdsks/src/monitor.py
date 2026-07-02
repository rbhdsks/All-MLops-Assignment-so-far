import pandas as pd
import joblib
import yaml
from sklearn.metrics import accuracy_score


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    model_path = "models/model_v1.pkl"
    production_data_path = "data/production/day2.csv"
    threshold = config["monitoring"]["error_threshold"]

    # Load model
    model = joblib.load(model_path)

    # Load production data
    df = pd.read_csv(production_data_path)

    print(f"Production data shape: {df.shape}")

    target_column = "Machine failure"

    # Separate features and target
    X_prod = df.drop(columns=[target_column])
    y_true = df[target_column]

    # Align feature columns with training
    training_columns = model.feature_names_in_

    for col in training_columns:
        if col not in X_prod.columns:
            X_prod[col] = 0

    X_prod = X_prod[training_columns]

    # Predict
    y_pred = model.predict(X_prod)

    # Calculate production accuracy
    prod_accuracy = accuracy_score(y_true, y_pred)
    prod_error = 1 - prod_accuracy

    print(f"Production Accuracy: {prod_accuracy:.4f}")
    print(f"Production Error Rate: {prod_error:.4f}")

    # Compare with threshold
    if prod_error > threshold:
        print("ALERT: Production error exceeded threshold.")
        print("Manual retraining required.")
    else:
        print(" Model performance within acceptable range.")


if __name__ == "__main__":
    main()
