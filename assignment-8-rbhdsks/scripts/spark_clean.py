"""Spark pipeline: ingest, clean, join, UDF, export.

Optimized for limited-RAM machines:
- Drops unused columns early (less data to shuffle)
- Uses dropDuplicates on key columns only (not whole row)
- Uses pandas_udf for vectorized Python execution
"""
import argparse, time, json, os, glob
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (StructType, StructField, LongType, DoubleType,
                                StringType, TimestampType)
from functools import reduce
import pandas as pd


def build_spark(master_url):
    return (SparkSession.builder
        .appName("SparkClean_NYC_Taxi")
        .master(master_url)
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.driver.memory", "3g")
        .config("spark.executor.memory", "3g")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate())


TAXI_SCHEMA = StructType([
    StructField("VendorID", LongType(), True),
    StructField("tpep_pickup_datetime", TimestampType(), True),
    StructField("tpep_dropoff_datetime", TimestampType(), True),
    StructField("passenger_count", DoubleType(), True),
    StructField("trip_distance", DoubleType(), True),
    StructField("RatecodeID", DoubleType(), True),
    StructField("store_and_fwd_flag", StringType(), True),
    StructField("PULocationID", LongType(), True),
    StructField("DOLocationID", LongType(), True),
    StructField("payment_type", LongType(), True),
    StructField("fare_amount", DoubleType(), True),
    StructField("total_amount", DoubleType(), True),
])


def normalize_columns(df):
    """Cast to canonical schema, dropping columns we don't need."""
    select_exprs = []
    existing = set(df.columns)
    for field in TAXI_SCHEMA.fields:
        if field.name in existing:
            select_exprs.append(F.col(field.name).cast(field.dataType).alias(field.name))
        else:
            select_exprs.append(F.lit(None).cast(field.dataType).alias(field.name))
    return df.select(*select_exprs)


# Vectorized pandas UDF — way faster than row-by-row Python UDF
@F.pandas_udf(DoubleType())
def avg_speed_mph_udf(distance: pd.Series, pickup: pd.Series, dropoff: pd.Series) -> pd.Series:
    duration_hr = (dropoff - pickup).dt.total_seconds() / 3600.0
    speed = distance / duration_hr
    speed = speed.where((duration_hr > 0) & (distance > 0))
    return speed


def run(args):
    timings = {}
    t_total = time.perf_counter()
    spark = build_spark(args.master)
    spark.sparkContext.setLogLevel("WARN")

    # 1. INGESTION
    t = time.perf_counter()
    parquet_files = sorted(glob.glob(args.input_trips))
    print(f"[Spark] Found {len(parquet_files)} parquet files")
    dfs = [normalize_columns(spark.read.parquet(p)) for p in parquet_files]
    trips = reduce(lambda a, b: a.unionByName(b), dfs)
    zones = spark.read.option("header", True).csv(args.input_zones)
    initial = trips.count()
    timings["1_ingestion_sec"] = round(time.perf_counter() - t, 3)
    print(f"[Spark] Ingested {initial:,} trips")

    # 2. CLEANSING (deduplicate on KEY columns only — much cheaper)
    t = time.perf_counter()
    cleaned = (trips
        .dropna(subset=["tpep_pickup_datetime","tpep_dropoff_datetime",
                        "passenger_count","trip_distance","PULocationID"])
        .dropDuplicates(["VendorID","tpep_pickup_datetime","tpep_dropoff_datetime",
                         "PULocationID","DOLocationID","trip_distance"])
        .filter(F.col("trip_distance") > 0)
        .filter(F.col("passenger_count") > 0))
    cleaned_count = cleaned.count()
    timings["2_cleansing_sec"] = round(time.perf_counter() - t, 3)
    print(f"[Spark] Cleaned: {cleaned_count:,}")

    # 3a. HEAVY JOIN (broadcast small zones table — much faster than shuffle join)
    t = time.perf_counter()
    z = (zones.withColumnRenamed("LocationID","PULocationID")
              .withColumnRenamed("Borough","PU_Borough")
              .withColumnRenamed("Zone","PU_Zone")
              .withColumnRenamed("service_zone","PU_service_zone"))
    z = z.withColumn("PULocationID", F.col("PULocationID").cast("long"))
    joined = cleaned.join(F.broadcast(z), on="PULocationID", how="left")
    j_count = joined.count()
    timings["3a_heavy_join_sec"] = round(time.perf_counter() - t, 3)
    print(f"[Spark] Joined: {j_count:,}")

    # 3b. PANDAS UDF (vectorized — 10-100x faster than row UDF)
    t = time.perf_counter()
    transformed = (joined.withColumn("avg_speed_mph",
            avg_speed_mph_udf(F.col("trip_distance"),
                              F.col("tpep_pickup_datetime"),
                              F.col("tpep_dropoff_datetime")))
        .filter(F.col("avg_speed_mph").isNotNull()))
    udf_count = transformed.count()
    timings["3b_python_udf_sec"] = round(time.perf_counter() - t, 3)
    print(f"[Spark] After UDF: {udf_count:,}")

    # 4. EXPORT
    t = time.perf_counter()
    transformed.write.mode("overwrite").parquet(args.output)
    timings["4_export_sec"] = round(time.perf_counter() - t, 3)
    print(f"[Spark] Wrote to {args.output}")

    timings["total_sec"] = round(time.perf_counter() - t_total, 3)
    timings["framework"] = "spark"
    timings["master"] = args.master
    timings["initial_rows"] = initial
    timings["final_rows"] = udf_count

    os.makedirs("logs", exist_ok=True)
    with open("logs/spark_metrics.json","w") as f:
        json.dump(timings, f, indent=2)
    print("\n=== Spark Timings ===")
    print(json.dumps(timings, indent=2))
    spark.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--master", default="local[*]")
    p.add_argument("--input-trips", default="data/raw/*.parquet")
    p.add_argument("--input-zones", default="data/zones/taxi_zone_lookup.csv")
    p.add_argument("--output", default="output/spark_out")
    run(p.parse_args())
