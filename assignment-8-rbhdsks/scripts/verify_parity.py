"""Verify Spark and Ray outputs match."""
import pandas as pd, glob, sys

def load(d):
    files = glob.glob(f"{d}/**/*.parquet", recursive=True)
    if not files:
        raise FileNotFoundError(f"No parquet in {d}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

s = load("output/spark_out")
r = load("output/ray_out")
print(f"Spark rows: {len(s):,}")
print(f"Ray rows:   {len(r):,}")
print(f"Diff:       {abs(len(s)-len(r)):,} ({100*abs(len(s)-len(r))/len(s):.3f}%)")
print(f"\nAvg speed — Spark: {s['avg_speed_mph'].mean():.3f}  Ray: {r['avg_speed_mph'].mean():.3f}")
print(f"Distance sum — Spark: {s['trip_distance'].sum():,.2f}  Ray: {r['trip_distance'].sum():,.2f}")

diff = abs(s['avg_speed_mph'].mean() - r['avg_speed_mph'].mean()) / s['avg_speed_mph'].mean()
if diff < 0.01:
    print("\n✅ PARITY PASSED")
else:
    print("\n❌ PARITY FAILED")
    sys.exit(1)
