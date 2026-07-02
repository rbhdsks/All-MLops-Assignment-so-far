import os
import yaml
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    raw_path = config["data"]["raw_path"]
    processed_path = config["data"]["processed_path"]
    train_rows = config["data"]["train_rows"]

    # Load raw data
    df = pd.read_csv(raw_path)

    print(f"Raw data shape: {df.shape}")

    # Drop ID columns if present
    columns_to_drop = ["UDI", "Product ID"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Encode categorical column 'Type' if present
    if "Type" in df.columns:
        le = LabelEncoder()
        df["Type"] = le.fit_transform(df["Type"])

    # Keep first N rows (chronological split)
    df_train = df.iloc[:train_rows]

    print(f"Processed training data shape: {df_train.shape}")

    # Ensure processed directory exists
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)

    # Save processed data
    df_train.to_csv(processed_path, index=False)

    print(f"Saved processed data to {processed_path}")

    # Create/append manifest log
    manifest_path = os.path.join(os.path.dirname(processed_path), "manifest.txt")

    with open(manifest_path, "a") as f:
        f.write("\n")
        f.write(f"Version: {os.path.basename(processed_path)}\n")
        f.write(f"Created on: {datetime.now()}\n")
        f.write("Changes:\n")
        f.write("- Dropped UDI and Product ID (if present)\n")
        f.write("- Label encoded Type column\n")
        f.write(f"- Selected first {train_rows} rows for training\n")
        f.write("-" * 40 + "\n")

    print("Manifest updated.")


if __name__ == "__main__":
    main()

