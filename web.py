import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from datetime import datetime
import pandas as pd

app = Flask(__name__)


def load_env_file():
    env_path = Path(".env")
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load_env_file()

# =========================
# Load CSV
# =========================
df = pd.read_csv("crime_dataw.csv")
df["day"] = df["day"].str.lower()

# =========================
# Time Helpers
# =========================
def time_to_min(t):
    h, m = map(int, t.split(":"))
    return h * 60 + m

def in_range(user_min, start_min, end_min):
    # normal: 6 - 11:59
    if start_min <= end_min:
        return start_min <= user_min <= end_min
    # overnight: 20 - 5:59    php artisan make:model Post -m
    else:
        return user_min >= start_min or user_min <= end_min

# =========================
@app.route("/")
def home():
    return render_template("map.html")

# =========================
@app.route("/get_hotspots")
def get_hotspots():
    date = request.args.get("date")
    time = request.args.get("time")

    if date and time:
        dt = datetime.strptime(date + " " + time, "%Y-%m-%d %H:%M")
    else:
        dt = datetime.now()

    day = dt.strftime("%A").lower()
    user_min = time_to_min(dt.strftime("%H:%M"))

    data = []

    for _, row in df.iterrows():
        if row["day"] != day:
            continue

        # example: "6 - 11:59"
        start, end = row["hour"].split("-")
        start = start.strip()
        end = end.strip()

        # "6" → "06:00"
        if ":" not in start:
            start += ":00"

        start_min = time_to_min(start)
        end_min   = time_to_min(end)

        if in_range(user_min, start_min, end_min):
            data.append({
                "lat": float(row["lat"]),
                "lng": float(row["long"]),   
                "weight": float(row["weight"]),
                "crime": row["crime"],
                "time_range": row["hour"]
            })

    return jsonify(data)

# =========================
@app.route("/send_sos_email", methods=["POST"])
def send_sos_email():
    payload = request.get_json(silent=True) or {}

    my_email = (payload.get("my_email") or "").strip()
    to_email = (payload.get("to_email") or "").strip()
    body = (payload.get("body") or "").strip()
    stationary_minutes = payload.get("stationary_minutes")
    countdown_minutes = payload.get("countdown_minutes")
    location = payload.get("location") or {}

    if not my_email or not to_email or not body:
        return jsonify({"ok": False, "error": "Missing required email fields."}), 400

    smtp_host = os.getenv("SMTP_HOST") or "smtp.gmail.com"
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    smtp_from = os.getenv("SMTP_FROM") or smtp_username or my_email
    smtp_use_tls = os.getenv("SMTP_USE_TLS", "true").lower() != "false"

    if not smtp_host or not smtp_username or not smtp_password:
        return jsonify({"ok": False, "error": "SMTP is not configured on the server."}), 500

    location_text = "Unavailable"
    if location.get("lat") is not None and location.get("lng") is not None:
        location_text = f'{location["lat"]}, {location["lng"]}'

    message = EmailMessage()
    message["Subject"] = "SOS Rescue Alert"
    message["From"] = smtp_from
    message["To"] = to_email
    message["Reply-To"] = my_email
    message.set_content(
        f"SOS rescue mail triggered.\n\n"
        f"My Email: {my_email}\n"
        f"To Mail: {to_email}\n"
        f"Stationary alert after: {stationary_minutes} minute(s)\n"
        f"Notification countdown: {countdown_minutes} minute(s)\n"
        f"Last known location: {location_text}\n\n"
        f"Message:\n{body}\n"
    )

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            if smtp_use_tls:
                server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(message)
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Failed to send email: {exc}"}), 500

    return jsonify({"ok": True})

# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
