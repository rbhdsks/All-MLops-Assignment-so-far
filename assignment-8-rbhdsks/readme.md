# Spark vs Ray: The Data Engineering Duel

**Course:** DA5402 — MLOps (Jan 2026), Assignment 8
**Author:** Nitesh Kumar Shah (ID25M806)
**Hardware Partner:** Nikesh Kumar Mandal (ID25M805)

This project benchmarks Apache Spark against Ray on an identical 5-stage data engineering pipeline, deployed across a real 2-node cluster (two MacBook Airs over a Wi-Fi LAN).

## Headline Result

| Configuration | Total Time | vs Spark Single-Node |
|---|---|---|
| Spark single-node | 196.16 s | baseline |
| **Spark 2-node distributed** | **116.91 s** | **40% faster** |
| Ray single-node | 258.99 s | 32% slower |
| Ray 2-node distributed | 439.08 s | 124% slower |

**TL;DR:** Spark wins every configuration. Spark scaled with distribution; Ray got worse with distribution. See the full report (`report.pdf`) for the architectural reasons.

## Project Structure

    spark-vs-ray/
    ├── scripts/
    │   ├── spark_clean.py          # Spark pipeline (5 stages)
    │   ├── ray_clean.py            # Ray pipeline (5 stages)
    │   ├── verify_parity.py        # Cross-framework output verification
    │   ├── plot.py                 # Generate comparison charts
    │   ├── download_data.sh        # Fetch NYC TLC dataset (~2 GB)
    │   └── monitor.sh              # Capture top output during runs
    ├── setup/
    │   ├── start_spark_master.sh
    │   ├── start_spark_worker.sh
    │   ├── start_ray_head.sh
    │   └── start_ray_worker.sh
    ├── screenshots/                # Cluster orchestration evidence
    │   ├── spark_master_2workers.png
    │   ├── Spark Executors tab showing both IPs.png
    │   ├── Spark master UI showing completed application.png
    │   └── ray dashboard 2 nodes.png
    ├── logs/                       # Metrics + comparison charts
    │   ├── spark_metrics.json
    │   ├── spark_distributed_metrics.json
    │   ├── ray_metrics.json
    │   ├── ray_distributed_metrics.json
    │   ├── comparison.png
    │   ├── comparison_distributed.png
    │   ├── scaling_comparison.png
    │   └── total_comparison.png
    ├── data/                       # Not in git (see .gitignore)
    ├── output/                     # Pipeline outputs (not in git)
    ├── requirements.txt
    ├── report.pdf                  # 8-page benchmark report
    └── README.md

## Setup

### Prerequisites

Both machines need:
- macOS (Apple Silicon)
- Homebrew
- Java 17 (`brew install openjdk@17`)
- Python 3.11 (`brew install python@3.11`)

### Installation

~~~bash
# 1. Clone
git clone <repo-url>
cd spark-vs-ray

# 2. Create venv
python3.11 -m venv venv
source venv/bin/activate

# 3. Install Python deps
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install full Spark distribution (the pip pyspark package omits sbin/)
curl -L -o spark-3.5.1-bin-hadoop3.tgz https://archive.apache.org/dist/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz
tar -xzf spark-3.5.1-bin-hadoop3.tgz && rm spark-3.5.1-bin-hadoop3.tgz

# 5. Download dataset (~2 GB)
chmod +x scripts/*.sh setup/*.sh
./scripts/download_data.sh
~~~

## Running

### Single-Node (Local Testing)

~~~bash
# Spark
python scripts/spark_clean.py --master "local[*]"

# Ray
python scripts/ray_clean.py --address local

# Verify parity
python scripts/verify_parity.py

# Generate charts
python scripts/plot.py
~~~

### 2-Node Distributed (the Real Benchmark)

**On Master (M1):**

~~~bash
export SPARK_HOME=$(pwd)/spark-3.5.1-bin-hadoop3
export SPARK_LOCAL_IP=<M1_IP>
$SPARK_HOME/sbin/start-master.sh --host <M1_IP>
$SPARK_HOME/sbin/start-worker.sh spark://<M1_IP>:7077 --memory 4g --cores 4 --host <M1_IP>
~~~

**On Worker (M4):**

~~~bash
export SPARK_HOME=$(pwd)/spark-3.5.1-bin-hadoop3
export SPARK_LOCAL_IP=<M4_IP>
$SPARK_HOME/sbin/start-worker.sh spark://<M1_IP>:7077 --memory 4g --cores 4 --host <M4_IP>
~~~

**Submit job (on M1):**

~~~bash
python scripts/spark_clean.py --master spark://<M1_IP>:7077
~~~

**For Ray (similar flow):**

~~~bash
# On M1 (Head)
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1
ray start --head --node-ip-address=<M1_IP> --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --object-store-memory=1500000000

# On M4 (Worker)
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1
ray start --address='<M1_IP>:6379' --node-ip-address=<M4_IP> --object-store-memory=1500000000

# Submit (on M1)
python scripts/ray_clean.py --address auto
~~~

## Pipeline Stages

| Stage | Operation |
|---|---|
| 1. Ingest | Read 6 monthly parquet files + zone CSV (with schema normalization) |
| 2. Cleanse | Drop nulls, deduplicate, filter zero-distance / zero-passenger trips |
| 3a. Join | Broadcast-join trips with `taxi_zones` lookup |
| 3b. UDF | Apply Python `avg_speed_mph(distance, pickup, dropoff)` per row |
| 4. Export | Write enriched dataset back as parquet |

## Cluster Topology

| | Master/Head | Worker |
|---|---|---|
| Machine | MacBook Air M1 | MacBook Air M4 |
| Cores | 8 (4P + 4E) | 8 |
| RAM | 8 GB | 16 GB |
| LAN IP | 192.168.0.134 | 192.168.0.209 |

## Key Findings

1. **Spark beat Ray on every configuration.** Single-node and distributed.
2. **Spark scales with distribution; Ray gets worse with distribution.** Ray's actor-model overhead exceeds parallelism gains at this dataset scale on a Wi-Fi LAN.
3. **`pandas_udf` neutralises most of Ray's "Python-native advantage."** By using Arrow as the Spark UDF wire format, the JVM-Python tax shrinks dramatically.
4. **Catalyst optimiser fuses operators** that Ray Data treats as independent stages — a structural advantage for Spark on ETL workloads.
5. **For an AI-first project, Ray still wins on integration** (Ray Train, Ray Tune, Ray Serve, no JVM boundary). For BI/ETL, Spark wins on raw performance.

See `report.pdf` for the full analysis.

## Reproducing the Results

All metrics are stored in `logs/*_metrics.json`. Charts can be regenerated with:

~~~bash
python scripts/plot.py
~~~

The `screenshots/` directory contains live UI captures from the Spark master, Spark Executors tab, and Ray Dashboard during the actual distributed runs.

## Tech Stack

- Apache Spark 3.5.1 (PySpark)
- Ray 2.9.3 (with Ray Data)
- Python 3.11.15
- OpenJDK 17.0.14
- pandas 2.1.4, pyarrow 14.0.2, numpy 1.26.3
- matplotlib 3.8.2

