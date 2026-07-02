import requests
import smtplib
import sqlite3
from email.mime.text import MIMEText
from bs4 import BeautifulSoup
from datetime import datetime
import os
AIRFLOW_HOME = "/opt/airflow"
DB_PATH = "/opt/airflow/data/scraper.db"
# # ── Database ──────────────────────────────────────────────
# DB_PATH = "/Users/zsres/airflow_a6/data/scraper.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS scraped_pages (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            url       TEXT UNIQUE,
            html_path TEXT,
            images    TEXT,
            status    TEXT,
            scraped_at TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_to_db(url, html_path, images, status):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute('''
            INSERT OR IGNORE INTO scraped_pages
            (url, html_path, images, status, scraped_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (url, html_path, ",".join(images), status, datetime.now().isoformat()))
        conn.commit()
    finally:
        conn.close()

def get_page_count():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM scraped_pages WHERE status='success'")
    count = c.fetchone()[0]
    conn.close()
    return count

# ── Email ─────────────────────────────────────────────────
SMTP_HOST   = "smtp.gmail.com"
SMTP_PORT   = 587
SMTP_USER   = "sirali.nitesh@gmail.com"
SMTP_PASS   = "zengncnghuuowjsn"
EMAIL_TO    = "sirali.nitesh@gmail.com"

def send_email(subject, body):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"]    = SMTP_USER
    msg["To"]      = EMAIL_TO
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, EMAIL_TO, msg.as_string())

# ── Scraper ───────────────────────────────────────────────
SCRAPED_DIR = "/opt/airflow/data/scraped"

def scrape_url(url):
    """Scrape a URL. Returns (html_path, image_links) or raises on failure."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()  # raises on 404, 5xx etc.

    soup = BeautifulSoup(response.text, "html.parser")

    # Save HTML
    safe_name = url.replace("https://", "").replace("http://", "").replace("/", "_")
    html_path = f"{SCRAPED_DIR}/{safe_name}.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(response.text)

    # Extract image links
    images = [
        img["src"] for img in soup.find_all("img") if img.get("src")
    ]

    return html_path, images