"""Ray pipeline: ingest, clean, join, UDF, export.

Note: NYC TLC parquet files have schema drift (VendorID is INT in some files,
BIGINT in others; airport_fee vs Airport_fee). pyarrow's reader is more
permissive than Spark's, but we still normalize to be safe.
"""
import argparse, time, json, os
import ray
import pandas as pd
import numpy as np


# Canonical column types — same logical schema as Spark
CANONICAL_TYPES = {
    "VendorID": "int64",
    "passenger_count": "float64",
    "trip_distance": "float64",
    "RatecodeID": "float64",
    "PULocationID": "int64",
    "DOLocationID": "int64",
    "payment_type": "int64",
    "fare_amount": "float64",
    "total_amount": "float64",
}

KEEP_COLS = ["VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
             "passenger_count", "trip_distance", "RatecodeID",
             "PULocationID", "DOLocationID", "payment_type",
             "fare_amount", "total_amount"]


def normalize_batch(batch: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to canonical types, drop unused columns."""
    # Handle Airport_fee vs airport_fee
    for c in list(batch.columns):
        if c.lower() == "airport_fee" and c != "airport_fee":
            batch = batch.rename(columns={c: "airport_fee"})
    # Keep only what we need
    cols = [c for c in KEEP_COLS if c in batch.columns]
    batch = batch[cols].copy()
    # Cast numeric columns
    for col, dtype in CANONICAL_TYPES.items():
        if col in batch.columns:
            batch[col] = pd.to_numeric(batch[col], errors="coerce")
            if dtype == "int64":
                batch[col] = batch[col].astype("Int64")  # nullable int
    # Cast timestamps
    batch["tpep_pickup_datetime"] = pd.to_datetime(batch["tpep_pickup_datetime"])
    batch["tpep_dropoff_datetime"] = pd.to_datetime(batch["tpep_dropoff_datetime"])
    return batch


def avg_speed_mph_batch(batch: pd.DataFrame) -> pd.DataFrame:
    duration_sec = (batch["tpep_dropoff_datetime"] -
                    batch["tpep_pickup_datetime"]).dt.total_seconds()
    duration_hr = duration_sec / 3600.0
    speed = np.where(
        (duration_hr > 0) & (batch["trip_distance"] > 0),
        batch["trip_distance"] / duration_hr.replace(0, np.nan),
        np.nan)
    batch["avg_speed_mph"] = speed
    return batch


def run(args):
    timings = {}
    t_total = time.perf_counter()
    if args.address == "local":
        ray.init(ignore_reinit_error=True)
    else:
        ray.init(address=args.address, ignore_reinit_error=True)
    print(f"[Ray] Resources: {ray.cluster_resources()}")

    # 1. INGESTION
    t = time.perf_counter()
    trips_raw = ray.data.read_parquet(args.input_trips)
    trips = trips_raw.map_batches(normalize_batch, batch_format="pandas")
    zones = ray.data.read_csv(args.input_zones)
    initial = trips.count()
    timings["1_ingestion_sec"] = round(time.perf_counter() - t, 3)
    print(f"[Ray] Ingested {initial:,}")

    # 2. CLEANSING
    t = time.perf_counter()
    def cleanse(b):
        b = b.dropna(subset=["tpep_pickup_datetime","tpep_dropoff_datetime",
                             "passenger_count","trip_distance","PULocationID"])
        b = b.drop_duplicates(subset=["VendorID","tpep_pickup_datetime",
                                      "tpep_dropoff_datetime","PULocationID",
                                      "DOLocationID","trip_distance"])
        b = b[b["trip_distance"] > 0]
        b = b[b["passenger_count"] > 0]
        return b
    cleaned = trips.map_batches(cleanse, batch_format="pandas")
    cleaned_count = cleaned.count()
    timings["2_cleansing_sec"] = round(time.perf_counter() - t, 3)
    print(f"[Ray] Cleaned: {cleaned_count:,}")

    # 3a. JOIN (broadcast small zones table — same strategy as Spark)
    t = time.perf_counter()
    zones_pd = zones.to_pandas().rename(columns={
        "LocationID":"PULocationID","Borough":"PU_Borough",
        "Zone":"PU_Zone","service_zone":"PU_service_zone"})
    zones_pd["PULocationID"] = zones_pd["PULocationID"].astype("int64")
    def join(b):
        b["PULocationID"] = b["PULocationID"].astype("int64")
        return b.merge(zones_pd, on="PULocationID", how="left")
    joined = cleaned.map_batches(join, batch_format="pandas")
    j_count = joined.count()
    timings["3a_heavy_join_sec"] = round(time.perf_counter() - t, 3)
    print(f"[Ray] Joined: {j_count:,}")

    # 3b. PYTHON UDF — runs natively in Python, no JVM hop
    t = time.perf_counter()
    transformed = joined.map_batches(avg_speed_mph_batch, batch_format="pandas")
    def keep_valid(b):
        return b[b["avg_speed_mph"].notna()]
    transformed = transformed.map_batches(keep_valid, batch_format="pandas")
    udf_count = transformed.count()
    timings["3b_python_udf_sec"] = round(time.perf_counter() - t, 3)
    print(f"[Ray] After UDF: {udf_count:,}")

    # 4. EXPORT
    t = time.perf_counter()
    transformed.write_parquet(args.output)
    timings["4_export_sec"] = round(time.perf_counter() - t, 3)
    print(f"[Ray] Wrote to {args.output}")

    timings["total_sec"] = round(time.perf_counter() - t_total, 3)
    timings["framework"] = "ray"
    timings["address"] = args.address
    timings["initial_rows"] = initial
    timings["final_rows"] = udf_count

    os.makedirs("logs", exist_ok=True)
    with open("logs/ray_metrics.json","w") as f:
        json.dump(timings, f, indent=2)
    print("\n=== Ray Timings ===")
    print(json.dumps(timings, indent=2))
    ray.shutdown()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--address", default="local")
    p.add_argument("--input-trips", default="data/raw/")
    p.add_argument("--input-zones", default="data/zones/taxi_zone_lookup.csv")
    p.add_argument("--output", default=os.path.expanduser("~/Documents/spark-vs-ray/output/ray_out"))
    run(p.parse_args())
