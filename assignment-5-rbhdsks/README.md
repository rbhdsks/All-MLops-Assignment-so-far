# DA5402 A5 — Application Monitoring with Prometheus & Grafana

**Name:** Nitesh Kumar Shah | **Roll:** ID25M806 | **Course:** DA5402 MLOps 

---
#### Screencast: https://drive.google.com/file/d/1EPzGynIw6D-QDKVXtyl6e5C7L-BayhyF/view?usp=sharing 


## Stack

| Component | Tool | Version |
|---|---|---|
| Web app | Streamlit | latest |
| AI model | BLIP (blip-image-captioning-base) | HuggingFace |
| Instrumentation | prometheus_client | latest |
| Metrics scraping | Prometheus | 3.10.0 |
| System metrics | node_exporter | 1.10.2 |
| Alerting | AlertManager | 0.27.0 |
| Email testing | Mailtrap sandbox | — |
| Dashboard | Grafana | 12.4.1 |
| Platform | MacBook Air M1 (darwin/arm64) | — |

---

## Project Structure

```
assignment-5-rbhdsks/
├── app.py                  # Streamlit app with all Prometheus metrics
├── prometheus.yml          # Scrape config — both targets
├── alert_rules.yml         # 7 alert rules + 3 recording rules
├── alertmanager.yml        # Email routing via Mailtrap SMTP
├── grafana-dashboard.json  # Exportable 9-panel Grafana dashboard
├── start.sh                # One-command launcher for all services
├── requirements.txt        # Python dependencies
└── README.md
```

---

## How to Run

### Prerequisites

Install tools via Homebrew (M1 Mac):
```bash
brew install prometheus node_exporter grafana
```

AlertManager (ARM64 binary — not in Homebrew):
```bash
curl -LO https://github.com/prometheus/alertmanager/releases/download/v0.27.0/alertmanager-0.27.0.darwin-arm64.tar.gz
tar xvf alertmanager-0.27.0.darwin-arm64.tar.gz
sudo mv alertmanager-0.27.0.darwin-arm64/alertmanager /usr/local/bin/
sudo xattr -rd com.apple.quarantine /usr/local/bin/alertmanager
```

### Python environment
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Start everything
```bash
chmod +x start.sh
./start.sh
```

This launches node_exporter, AlertManager, Prometheus, Grafana, and the Streamlit app in one command.

### Verify all services are up

| URL | Expected |
|---|---|
| http://localhost:8501 | Streamlit app |
| http://localhost:8001/metrics | Raw Prometheus metrics |
| http://localhost:9090/targets | Both targets green UP |
| http://localhost:9093 | AlertManager UI |
| http://localhost:3000 | Grafana (admin/admin123) |

---

## Metrics Implemented

The app exposes 8 custom metrics on port 8001 covering all 4 Prometheus metric types.

### Counters
| Metric | Labels | What it tracks |
|---|---|---|
| `captioning_images_processed_total` | `mode` (single/bulk) | Total images captioned |
| `captioning_errors_total` | `error_type` | Total errors by exception type |
| `captioning_requests_by_source_total` | `source_ip`, `mode` | Requests by client IP |

### Gauges
| Metric | What it tracks |
|---|---|
| `captioning_active_requests` | Images currently being processed |
| `captioning_model_memory_mb` | Estimated BLIP model memory (~900MB) |

### Histograms
| Metric | Labels | Buckets |
|---|---|---|
| `captioning_inference_latency_seconds` | `mode` | 0.5, 1, 2, 5, 10, 30s |
| `captioning_bulk_batch_size` | — | 1, 5, 10, 20, 50, 100 |

### Summary
| Metric | What it tracks |
|---|---|
| `captioning_caption_word_count` | Word count of generated captions (gives `_sum` and `_count`) |

---

## Alert Rules

7 alert rules are defined in `alert_rules.yml`:

| Alert | Condition | For | Severity |
|---|---|---|---|
| `AppDown` | `up{job="captioning_app"} == 0` | 30s | critical |
| `HighErrorRate` | error rate > 0.05/s | 1m | critical |
| `SlowInference` | P95 latency > 10s | 2m | warning |
| `TooManyActiveRequests` | active > 5 | 1m | warning |
| `HighCPUUsage` | CPU > 80% | 2m | warning |
| `HighMemoryUsage` | RAM > 85% | 2m | warning |
| `HighDiskUsage` | Disk > 85% | 5m | warning |

### Recording Rules
Three recording rules pre-compute expensive queries for dashboard performance:
- `job:captioning_inference_latency:p95` — P95 inference latency
- `job:captioning_inference_latency:p50` — P50 inference latency  
- `job:cpu_usage:avg` — Average CPU usage percentage

### Inhibition Rules
`SlowInference` is suppressed when `AppDown` is firing — if the app is down, slow inference is redundant noise.

---

## Grafana Dashboard

The dashboard **AI Captioning Observability** has 9 panels with 5-second auto-refresh.

### Panel Layout

**Row 1 — Application Health (Stat panels)**
- App Status — UP/DOWN with green/red color mapping
- Active Requests — images in-flight right now
- Model Memory — 900MB confirms model is loaded

**Row 2 — Performance (Time series)**
- Throughput — images/second split by single vs bulk mode
- Inference Latency P95 — 95th percentile captioning time
- Error Rate — errors/second (zero = healthy)

**Row 3 — System Resources**
- CPU Usage — time series with % scale
- RAM Usage — gauge with yellow (70%) and red (85%) thresholds
- Disk Usage — gauge with same thresholds

### Import the Dashboard
1. Go to `http://localhost:3000`
2. Dashboards → Import
3. Upload `grafana-dashboard.json`
4. Select Prometheus as the data source
5. Click Import

### 7 Commandments Compliance

| Commandment | Implementation |
|---|---|
| Color-blind inclusive | Green/yellow/red with numeric values as fallback |
| Markers work across media | Solid lines + numeric displays readable in print |
| Axes named | Time (X), Percent/Seconds/count (Y) on all panels |
| Scale and units explicit | Each panel has units set: percent, seconds, short, decmbytes |
| Legend for multiple variables | Throughput shows single/bulk legend |
| Title and subtitle | All panels have title + description |
| Self-explanatory story | Each panel description explains what it tells you |

---

## Email Alerting Setup

1. Sign up at [mailtrap.io](https://mailtrap.io) (free)
2. Go to **Sandboxes → My Sandbox → SMTP** tab
3. Copy your Username and Password
4. Edit `alertmanager.yml`:
```yaml
global:
  smtp_smarthost: "sandbox.smtp.mailtrap.io:587"
  smtp_auth_username: "YOUR_USERNAME"
  smtp_auth_password: "YOUR_PASSWORD"
```
5. Restart AlertManager
6. Stop the Streamlit app → wait 35 seconds → check Mailtrap inbox for `[FIRING] AppDown` email

### Creating a Silence (Maintenance Window)
1. Go to `http://localhost:9093`
2. Click **New Silence**
3. Set matcher: `alertname=AppDown`
4. Set duration and add creator/comment
5. Click **Create**

---

## M1 Mac Specific Notes

### PyTorch MPS Segfault Fix
The BLIP model causes a segmentation fault on MPS (Apple GPU). The app forces CPU mode:
```python
DEVICE = torch.device("cpu")  # MPS causes segfault with BLIP on M1
```

### Duplicate Registry Fix
Streamlit reruns the script on every interaction. Metrics are wrapped in try/except to handle re-registration gracefully:
```python
def _counter(name, desc, labels=[]):
    try:
        return prom.Counter(name, desc, labels)
    except ValueError:
        return prom.REGISTRY._names_to_collectors.get(name + "_total")
```

### Versions Tested
```
torch:        2.10.0
transformers: 4.36.2  (downgraded from 5.3.0 — segfault fix)
streamlit:    latest
prometheus:   3.10.0
alertmanager: 0.27.0 (ARM64 binary)
grafana:      12.4.1
node_exporter: 1.10.2
```

---

## Stopping All Services

```bash
pkill -f streamlit
pkill -f node_exporter
pkill -f alertmanager
pkill -f prometheus
brew services stop grafana
deactivate
```



## AI Disclosure

Per the assignment Code of Conduct:

- **PromQL queries** in Grafana were generated with AI assistance and verified manually in the Prometheus query explorer. Each complex query is commented in `grafana-dashboard.json` with the prompt used.
- **AlertManager YAML structure** was generated with AI; threshold values were independently justified based on observed M1 CPU behavior during bulk uploads.
- **Debugging** — AI was used to diagnose MPS segfault, duplicate registry error, YAML encoding issue, and AlertManager binary issue.
- **Core design** (metric naming, label choices, alert logic, dashboard layout, threshold selection) was independently designed.
