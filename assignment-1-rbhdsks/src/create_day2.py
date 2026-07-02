# import os
# import pandas as pd
# import yaml
# import numpy as np


# def load_config():
#     with open("config.yaml", "r") as f:
#         return yaml.safe_load(f)


# def main():
#     config = load_config()
#     raw_path = config["data"]["raw_path"]

#     # Load raw dataset
#     df = pd.read_csv(raw_path)

#     print(f"Raw dataset shape: {df.shape}")

#     # Drop same columns as training
#     columns_to_drop = ["UDI", "Product ID"]
#     df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

#     # Encode Type same way (simple label encoding logic)
#     if "Type" in df.columns:
#         df["Type"] = df["Type"].astype("category").cat.codes

#     # Take rows 7000–8000 (future simulation)
#     df_day2 = df.iloc[7000:8000].copy()

#     print(f"Day2 subset shape: {df_day2.shape}")

#     #  Simulate drift
#     if "Air temperature [K]" in df_day2.columns:
#         df_day2["Air temperature [K]"] += 5

#     if "Torque [Nm]" in df_day2.columns:
#         df_day2["Torque [Nm]"] *= 1.05

#     # Add small random noise
#     #numeric_cols = df_day2.select_dtypes(include=[np.number]).columns
#     #for col in numeric_cols:
#         #df_day2[col] += np.random.normal(0, 0.5, size=len(df_day2))
#     numeric_cols = df_day2.select_dtypes(include=[np.number]).columns

#     for col in numeric_cols:
#         if col != "Machine failure":   #  do NOT modify target
#             df_day2[col] += np.random.normal(0, 0.5, size=len(df_day2))

#     # Ensure production folder exists
#     os.makedirs("data/production", exist_ok=True)

#     # Save day2 data
#     output_path = "data/production/day2.csv"
#     df_day2.to_csv(output_path, index=False)

#     print(f"Day2 data saved to {output_path}")


# if __name__ == "__main__":
    # main()


import os
import pandas as pd
import yaml
import numpy as np
from datetime import datetime


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    raw_path = config["data"]["raw_path"]

    # Load raw dataset
    df = pd.read_csv(raw_path)
    print(f"Raw dataset shape: {df.shape}")

    # Drop same columns as training
    columns_to_drop = ["UDI", "Product ID"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Encode Type same way as training
    if "Type" in df.columns:
        df["Type"] = df["Type"].astype("category").cat.codes

    # Take rows 7000–8000 (simulate future data)
    df_day2 = df.iloc[7000:8000].copy()
    print(f"Day2 subset shape: {df_day2.shape}")

    # -------------------------
    # Mild Drift Simulation
    # -------------------------

    if "Air temperature [K]" in df_day2.columns:
        df_day2["Air temperature [K]"] += 1

    if "Torque [Nm]" in df_day2.columns:
        df_day2["Torque [Nm]"] *= 1.01

    numeric_cols = df_day2.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col != "Machine failure":
            df_day2[col] += np.random.normal(0, 0.05, size=len(df_day2))

    # Ensure production folder exists
    os.makedirs("data/production", exist_ok=True)

    # Save production dataset
    output_path = "data/production/day2.csv"
    df_day2.to_csv(output_path, index=False)

    print(f"Day2 data saved to {output_path}")

    # -------------------------
    # Manifest Logging
    # -------------------------

    manifest_path = "data/processed/manifest.txt"

    with open(manifest_path, "a") as f:
        f.write("\n")
        f.write("========================================\n")
        f.write("DATA VERSION: day2.csv (Production Simulation)\n")
        f.write(f"CREATED ON: {datetime.now()}\n")
        f.write(f"SOURCE FILE: {raw_path}\n")
        f.write("SCRIPT USED: src/day2_generation.py\n\n")

        f.write("TRANSFORMATIONS:\n")
        f.write("- Selected rows 7000–8000 (future timeframe simulation)\n")
        f.write("- Increased Air temperature by +1K\n")
        f.write("- Increased Torque by 1%\n")
        f.write("- Added small Gaussian noise (mean=0, std=0.05) to numeric features\n")
        f.write("- No shuffling applied\n\n")

        f.write("DATA SHAPE:\n")
        f.write(f"- Rows: {df_day2.shape[0]}\n")
        f.write(f"- Columns: {df_day2.shape[1]}\n\n")

        f.write("PURPOSE:\n")
        f.write("Simulated production dataset for drift monitoring and evaluation.\n")
        f.write("----------------------------------------\n")

    print("Manifest updated with production dataset entry.")


if __name__ == "__main__":
    main()
