#!/bin/bash
PROJECT="$(cd "$(dirname "$0")" && pwd)"
echo "Project: $PROJECT"

# Kill any leftovers from previous runs
pkill -f node_exporter  2>/dev/null
pkill -f alertmanager   2>/dev/null
pkill -f prometheus     2>/dev/null
sleep 1

echo "Starting node_exporter..."
node_exporter > /tmp/node_exporter.log 2>&1 &

echo "Starting alertmanager..."
alertmanager \
  --config.file="$PROJECT/alertmanager.yml" \
  --storage.path="$PROJECT/alertmanager_data" \
  > /tmp/alertmanager.log 2>&1 &

echo "Starting prometheus..."
prometheus \
  --config.file="$PROJECT/prometheus.yml" \
  --storage.tsdb.path="$PROJECT/prometheus_data" \
  > /tmp/prometheus.log 2>&1 &

echo "Starting grafana..."
brew services start grafana

sleep 3
echo ""
echo "All services running. Open these URLs:"
echo "  Streamlit   -> http://localhost:8501"
echo "  Metrics     -> http://localhost:8001/metrics"
echo "  Prometheus  -> http://localhost:9090/targets"
echo "  Alertmanager-> http://localhost:9093"
echo "  Grafana     -> http://localhost:3000"
echo ""

source "$PROJECT/venv/bin/activate"
streamlit run "$PROJECT/app.py"