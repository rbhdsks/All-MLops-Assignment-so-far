import csv
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

from scraper_utils import (
    init_db, save_to_db, get_page_count,
    send_email, scrape_url
)

# ── Config ────────────────────────────────────────────────
AIRFLOW_HOME = os.environ.get("AIRFLOW_HOME", os.path.expanduser("~/Documents/airflow_a6"))
INPUT_DIR = "/opt/airflow/data/input"
CSV_FILE  = "/opt/airflow/data/input/targets.csv"
BATCH_LIMIT  = 2

# ── Default Args ──────────────────────────────────────────
default_args = {
    "owner": "student",
    "retries": 3,
    "retry_delay": timedelta(seconds=30),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

# ── DAG ───────────────────────────────────────────────────
with DAG(
    dag_id="web_scraper_pipeline",
    default_args=default_args,
    start_date=days_ago(1),
    schedule_interval=timedelta(hours=12),
    catchup=False,
    tags=["a6", "scraper"],
) as dag:

    # Task 1: Sense CSV — reschedule mode + soft_fail for dry-pipe branch
    sense_file = FileSensor(
        task_id="sense_csv_file",
        filepath=CSV_FILE,
        poke_interval=30,
        timeout=300,            # 5 min for testing; 43200 for 12h
        mode="reschedule",      # CRITICAL: releases worker between pokes
        soft_fail=True,         # on timeout → failed (not skipped)
    )

    # Task 2: Dry pipeline alert — only runs if sensor failed
    def dry_pipeline_alert():
        send_email(
            subject="[Airflow] Dry Pipeline Alert",
            body="No CSV file detected in the input directory within the timeout window."
        )

    dry_alert = PythonOperator(
        task_id="dry_pipeline_alert",
        python_callable=dry_pipeline_alert,
        trigger_rule="all_failed",   # only runs if upstream sensor failed
    )

    # Task 3: Init DB
    def init_database():
        init_db()
        print("Database initialized.")

    init_db_task = PythonOperator(
        task_id="init_database",
        python_callable=init_database,
        trigger_rule="all_success",  # only runs if sensor succeeded
    )

    # Task 4: Read URLs
    def read_urls(**context):
        urls = []
        with open(CSV_FILE, "r") as f:
            for row in csv.reader(f):
                if row and row[0].startswith("http"):
                    urls.append(row[0].strip())
        context["ti"].xcom_push(key="urls", value=urls)
        print(f"Found {len(urls)} URLs: {urls}")

    read_urls_task = PythonOperator(
        task_id="read_urls",
        python_callable=read_urls,
    )

    # Task 5: Scrape (uses pool)
    def scrape_and_store(**context):
        urls = context["ti"].xcom_pull(task_ids="read_urls", key="urls")
        if not urls:
            print("No URLs found.")
            return
        for url in urls:
            try:
                html_path, images = scrape_url(url)
                save_to_db(url, html_path, images, "success")
                print(f"SUCCESS: {url}")
            except Exception as e:
                save_to_db(url, "", [], "failed")
                send_email(
                    subject="[Airflow] Broken Link Alert",
                    body=f"Failed to scrape: {url}\n\nError: {str(e)}"
                )
                print(f"FAILED: {url} — {e}")

    scrape_task = PythonOperator(
        task_id="scrape_urls",
        python_callable=scrape_and_store,
        pool="scraper_pool",     # matches the pool you created in UI
    )

    # Task 6: Batch notification
    def check_and_notify():
        count = get_page_count()
        print(f"Total pages in DB: {count}")
        if count >= BATCH_LIMIT:
            send_email(
                subject="[Airflow] Collection Statistics",
                body=f"Batch threshold reached!\n\nTotal pages scraped: {count}"
            )

    notify_task = PythonOperator(
        task_id="batch_notification",
        python_callable=check_and_notify,
    )

    # Dependencies
    sense_file >> dry_alert
    sense_file >> init_db_task >> read_urls_task >> scrape_task >> notify_task