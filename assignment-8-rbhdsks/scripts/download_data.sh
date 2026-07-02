#!/bin/bash
set -e
DATA_DIR="$(dirname "$0")/../data/raw"
ZONE_DIR="$(dirname "$0")/../data/zones"
mkdir -p "$DATA_DIR" "$ZONE_DIR"
BASE="https://d37ci6vzurychx.cloudfront.net/trip-data"

echo "Downloading 6 months of yellow taxi data..."
for month in 01 02 03 04 05 06; do
    FILE="yellow_tripdata_2023-${month}.parquet"
    if [ ! -f "$DATA_DIR/$FILE" ]; then
        echo "  Fetching $FILE..."
        curl -# -o "$DATA_DIR/$FILE" "$BASE/$FILE"
    else
        echo "  $FILE exists, skip."
    fi
done

echo "Downloading taxi zone lookup..."
curl -s -o "$ZONE_DIR/taxi_zone_lookup.csv" \
    "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

echo ""
echo "Summary:"
du -sh "$DATA_DIR"
ls -lh "$DATA_DIR"
ls -lh "$ZONE_DIR"
