# DA5402 Assignment 6 — Orchestrated Web Scraper Pipeline

**name:** Nitesh Kumar Shah  
**Roll No:** ID25M806  
**Course:** DA5402 — MLOps  
**Assignment:** A6 — Orchestrated Web Scraper Pipeline

An Airflow-based "Web-to-DB" pipeline that watches for incoming CSV target lists, scrapes the listed URLs concurrently (throttled by an Airflow Pool), persists results in SQLite, and sends email alerts for dry pipelines, broken links, and batch collection statistics.

---



| Task | Type | Purpose |
|---|---|---|
| `sense_csv_file` | FileSensor | Watches `data/input/targets.csv` (reschedule mode, 5-min timeout for testing) |
| `dry_pipeline_alert` | PythonOperator | Sends "Dry Pipeline" email — runs only if sensor fails (`trigger_rule="all_failed"`) |
| `init_database` | PythonOperator | Creates SQLite table if missing |
| `read_urls` | PythonOperator | Parses CSV, pushes URL list to XCom |
| `scrape_urls` | PythonOperator | Scrapes each URL, extracts images, writes to DB. Uses `scraper_pool` (3 slots) |
| `batch_notification` | PythonOperator | Sends "Collection Statistics" email when DB has ≥10 pages |

---

## 2. Tech Stack

- **Apache Airflow 2.9.2** (CeleryExecutor)
- **PostgreSQL 13** — Airflow metadata DB
- **Redis 7.2** — Celery broker
- **SQLite** — scraped data store (`data/scraper.db`)
- **Python 3.11**, `requests`, `beautifulsoup4`
- **Docker Compose** — full stack orchestration
- **SMTP (Gmail)** — email alerts

---

## 3. Project Structure

```
airflow_docker/
├── config/
├── dags/
│   ├── __pycache__/
│   ├── scraper_utils.py       # DB + scraping helpers
│   └── web_scraper_dag.py     # DAG definition
├── data/
│   ├── input/
│   │   └── targets.csv        # URL list (sensed by FileSensor)
│   ├── scraped/               # Saved HTML files
│   └── scraper.db             # SQLite store for scraped pages
├── logs/                      # Airflow task logs
├── plugins/
├── screenshots/
│   ├── alert-dry.jpeg
│   ├── broken link alert.jpeg
│   ├── collection stats.jpeg
│   ├── connection.jpeg
│   ├── dry pipeline.jpeg
│   ├── mail.jpeg
│   ├── pool.jpeg
│   └── successful run.jpeg
├── .env                       # AIRFLOW_UID for Docker
├── docker                     # helper script
├── docker-compose.yaml        # Airflow + Postgres + Redis stack
├── Dockerfile                 # Custom image with scraper deps
├── requirements.txt           # Python deps (requests, beautifulsoup4)
└── README.md
```
---

## 4. Setup Instructions

### Prerequisites
- Docker Desktop (Apple Silicon or Intel)
- ≥4 GB RAM allocated to Docker
- A Gmail account with an [App Password](https://myaccount.google.com/apppasswords)

### Steps

```bash
# 1. Clone or unzip the project
cd ~/Documents/airflow_docker

# 2. Set the user ID (Linux/Mac)
echo "AIRFLOW_UID=$(id -u)" > .env

# 3. Initialize the Airflow metadata DB (one-time)
docker compose up airflow-init

# 4. Start everything
docker compose up -d

# 5. Wait ~60 seconds, then check
docker compose ps
```

All services should report `(healthy)`. Open the UI at **http://localhost:8080** with `airflow` / `airflow`.

### Configure Airflow

**Create the worker pool:**
```bash
docker compose exec airflow-webserver airflow pools set scraper_pool 3 "Scraper concurrency pool"
```

**Add SMTP connection** (UI → Admin → Connections → +):
- Connection Id: `smtp_default`
- Connection Type: `SMTP`
- Host: `smtp.gmail.com`
- Login: your Gmail address
- Password: your 16-char Gmail App Password
- Port: `587`
- From email: your Gmail address

---

## 5. Running the Pipeline

### Test 1 — Successful Run
```bash
cat > data/input/targets.csv << 'EOF'
https://example.com
https://www.python.org
https://www.wikipedia.org
https://www.github.com
https://news.ycombinator.com
https://www.djangoproject.com
https://flask.palletsprojects.com
https://www.postgresql.org
https://www.mozilla.org
https://www.gnu.org
https://www.kernel.org
https://this-does-not-exist-12345.com
EOF
```
In the UI: toggle `web_scraper_pipeline` ON, click ▶ to trigger.

**Expected:** All tasks green except `dry_pipeline_alert` (skipped — by design). 11 pages saved to DB. One broken-link email sent for the bad URL. Once DB has ≥10 rows, the Collection Statistics email also fires.

### Test 2 — Broken Link Alert
The CSV above already includes one bad URL, so this is exercised in Test 1. Email subject: **`[Airflow] Broken Link Alert`**.

### Test 3 — Dry Pipeline Alert
```bash
rm data/input/targets.csv
```
Edit `dags/web_scraper_dag.py` and lower the FileSensor `timeout` from `300` to `60`. Trigger the DAG. After ~1 minute the sensor fails, `dry_pipeline_alert` runs, and an email is sent. Email subject: **`[Airflow] Dry Pipeline Alert`**.

Restore `timeout=300` and put `targets.csv` back when done.

---

## 6. Verifying the Database

```bash
docker compose exec airflow-scheduler sqlite3 /opt/airflow/data/scraper.db \
  "SELECT id, url, status, scraped_at FROM scraped_pages;"
```
`url` is `UNIQUE`, and inserts use `INSERT OR IGNORE`, guaranteeing no duplicates across runs.

---

## 7. Concurrency & Robustness

- **Pool throttling:** `scrape_urls` is assigned `pool="scraper_pool"` with 3 slots — at most 3 scrapes run in parallel even with 100 URLs.
- **Sensor mode:** `mode="reschedule"` releases the worker between pokes (avoids occupying a slot for the full 5 minutes).
- **Retries:** `retries=3`, `retry_delay=30s`, `retry_exponential_backoff=True`, `max_retry_delay=5min`.
- **Malformed CSV handling:** rows that don't start with `http` are skipped silently in `read_urls`.
- **Trigger rules:**
  - `dry_pipeline_alert`: `all_failed` — fires only when the sensor times out
  - `init_database`: `all_success` — skipped on dry pipeline path

---

## 8. Test Report

| Test | Result | Evidence |
|---|---|---|
| Successful 12-URL run | ✅ All scrape tasks pooled & completed | `screenshots/successful_run.jpeg` |
| Broken link email | ✅ Sent for `this-does-not-exist-12345.com` | `screenshots/broken_link_alert.jpeg` |
| Dry pipeline email | ✅ Sensor failed, alert fired | `screenshots/dry_pipeline_run.jpeg` |
| Pool configured | ✅ `scraper_pool` with 3 slots | `screenshots/pool.jpeg` |
| SMTP configured | ✅ `smtp_default` | `screenshots/connection.jpeg` |
| All emails received | ✅ Inbox screenshot | `screenshots/mail.jpeg` |

---

## 9. Useful Commands

```bash
docker compose ps                                  # service status
docker compose logs -f airflow-scheduler           # tail scheduler
docker compose logs -f airflow-worker              # tail worker
docker compose exec airflow-webserver bash         # shell into a container
docker compose down                                # stop everything
docker compose down -v                             # stop AND wipe metadata DB
```

---

## 10. AI Disclosure Appendix

Per the course's "Fair Use of AI" Code of Conduct, this section discloses where and how AI assistance was used in this assignment. The DAG structure, task dependencies, pool configuration, trigger-rule design, and overall orchestration logic were authored manually. AI was used only for boilerplate scaffolding, debugging environment issues, and proofreading the report — never for designing the pipeline logic itself.



### Where AI was used

| Area | Specific use | Status per Code of Conduct |
|---|---|---|
| Scraping helper function | Asked AI to draft a `requests` + `BeautifulSoup` function that fetches a URL with timeout and extracts `<script src>` and `<img src>` tags | ✅ Allowed (boilerplate) |
| Mac M1 SIGSEGV debugging | Asked AI for help diagnosing gunicorn worker crashes on Apple Silicon, leading to the switch from local venv → Docker Compose | ✅ Allowed (debugging) |
| Docker Compose setup | Asked AI to explain how to extend the official `docker-compose.yaml` with a custom `Dockerfile` and a mounted `data/` volume | ✅ Allowed (boilerplate) |
| README structure | Asked AI to proofread and help structure this README into clean sections | ✅ Allowed (report proofreading) |
| **DAG logic, dependencies, trigger rules, pool design** | **Authored manually** | — |
| **Database schema and `INSERT OR IGNORE` deduplication strategy** | **Authored manually** | — |
| **Email branching (dry-pipe / broken-link / batch stats)** | **Authored manually** | — |


---

## 11. Known Limitations

- SMTP password is stored in Airflow's connection store (not encrypted unless `AIRFLOW__CORE__FERNET_KEY` is set).
- Scraped HTML is dumped to disk uncompressed; for production use, consider blob storage or compression.
- The batch notification fires every run once the threshold is met — a real system would want a "since last notification" check.