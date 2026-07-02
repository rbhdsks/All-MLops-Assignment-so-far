"""Plot Spark vs Ray comparison."""
import json, numpy as np, matplotlib.pyplot as plt

s = json.load(open("logs/spark_metrics.json"))
r = json.load(open("logs/ray_metrics.json"))

stages = ["1_ingestion_sec","2_cleansing_sec","3a_heavy_join_sec",
          "3b_python_udf_sec","4_export_sec"]
labels = ["Ingest","Cleanse","Join","UDF","Export"]
sv = [s[k] for k in stages]
rv = [r[k] for k in stages]
x = np.arange(len(labels)); w = 0.35

plt.figure(figsize=(10,5))
plt.bar(x-w/2, sv, w, label="Spark", color="#E25A1C")
plt.bar(x+w/2, rv, w, label="Ray",   color="#028CF3")
plt.xticks(x, labels)
plt.ylabel("Seconds")
plt.title(f"Spark ({s['total_sec']}s)  vs  Ray ({r['total_sec']}s)")
plt.legend(); plt.tight_layout()
plt.savefig("logs/comparison.png", dpi=150)
print("Saved logs/comparison.png")
