import os
import smtplib
import json
import re
from email.message import EmailMessage
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import pandas as pd
from hotspot_prediction import RouteHotspotPredictor

app = Flask(__name__)
NOMINATIM_BASE_URL = "https://nominatim.openstreetmap.org/search"
THANA_AREAS_FILE = Path("thana_with_areas (2).xlsx")
THANA_ADMIN_DATA_FILE = Path("thana_admin_data.json")
ROUTE_HOTSPOT_PREDICTOR = RouteHotspotPredictor()
POLICE_STATION_META = {
    "Kotwali Thana": {"lat": 23.707703986257986, "lng": 90.40946942982121, "phone": "0123456789", "email": "kotwali@police.gov.bd"},
    "Ramna Thana": {"lat": 23.7410, "lng": 90.4040, "phone": "0123456789", "email": "ramna@police.gov.bd"},
    "Gulshan Thana": {"lat": 23.7920, "lng": 90.4120, "phone": "0123456789", "email": "gulshan@police.gov.bd"},
    "Mirpur Thana": {"lat": 23.8280, "lng": 90.3570, "phone": "0123456789", "email": "mirpur@police.gov.bd"},
    "Badda Thana": {"lat": 23.7800, "lng": 90.4250, "phone": "0123456789", "email": "badda@police.gov.bd"},
    "Dhanmondi Thana": {"lat": 23.7460, "lng": 90.3740, "phone": "0123456789", "email": "dhanmondi@police.gov.bd"},
    "Mohammadpur Thana": {"lat": 23.7590, "lng": 90.3660, "phone": "0123456789", "email": "mohammadpur@police.gov.bd"},
    "Motijheel Thana": {"lat": 23.7280, "lng": 90.4110, "phone": "0123456789", "email": "motijheel@police.gov.bd"},
    "Paltan Thana": {"lat": 23.7350, "lng": 90.4150, "phone": "0123456789", "email": "paltan@police.gov.bd"},
    "Tejgaon Thana": {"lat": 23.7680, "lng": 90.4050, "phone": "0123456789", "email": "tejgaon@police.gov.bd"},
    "Keraniganj Thana": {"lat": 23.6960, "lng": 90.3650, "phone": "0123456789", "email": "keraniganj@police.gov.bd"},
    "Kamrangirchar Thana": {"lat": 23.7290, "lng": 90.4200, "phone": "0123456789", "email": "kamrangirchar@police.gov.bd"},
    "Hazaribagh Thana": {"lat": 23.7340, "lng": 90.4040, "phone": "0123456789", "email": "hazaribagh@police.gov.bd"},
    "Shahbagh Thana": {"lat": 23.7360, "lng": 90.3930, "phone": "0123456789", "email": "shahbagh@police.gov.bd"},
    "Lalbagh Thana": {"lat": 23.7160, "lng": 90.3990, "phone": "", "email": "lalbagh@police.gov.bd"},
    "Khilgaon Thana": {"lat": 23.7420, "lng": 90.4440, "phone": "0123456789", "email": "khilgaon@police.gov.bd"},
    "Khilkhet Thana": {"lat": 23.8680, "lng": 90.4220, "phone": "0123456789", "email": "khilkhet@police.gov.bd"},
    "Uttara Thana": {"lat": 23.8750, "lng": 90.3980, "phone": "0123456789", "email": "uttara@police.gov.bd"},
    "Airport Thana": {"lat": 23.8478, "lng": 90.4106, "phone": "0123456789", "email": "airport@police.gov.bd"},
    "Malibagh Thana": {"lat": 23.7500, "lng": 90.4180, "phone": "0123456789", "email": "malibagh@police.gov.bd"},
    "Kafrul Thana": {"lat": 23.8140, "lng": 90.3550, "phone": "0123456789", "email": "kafrul@police.gov.bd"},
}


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
# Load hotspot dataset
# =========================
HOTSPOT_DATA_FILE = Path("crime_data_cleaned_weather_metrics.xlsx")
CRIME_RISK_POINTS = {
    "murder": 10,
    "rape": 9,
    "stab+assault": 8,
    "robbery": 7,
    "body found": 6,
    "harassment": 5,
}


def crime_risk_points(crime_name):
    normalized_name = str(crime_name or "").strip().lower()
    return CRIME_RISK_POINTS.get(normalized_name, 4)


def normalize_hotspot_dataset(dataset_path):
    empty_columns = [
        "time_minutes",
        "hour",
        "crime",
        "weight",
        "day",
        "lat",
        "long",
    ]
    try:
        frame = pd.read_excel(dataset_path)
    except (FileNotFoundError, PermissionError, ValueError):
        return pd.DataFrame(columns=empty_columns)

    frame = frame.copy()

    frame["Day"] = frame["Day"].astype(str).str.strip().str.lower()
    frame["Time_dt"] = pd.to_datetime(frame["Time"], format="%H:%M", errors="coerce")
    frame = frame.dropna(subset=["Day", "Time_dt", "Latitude", "Longitude", "Crime type"]).copy()

    frame["time_minutes"] = frame["Time_dt"].dt.hour * 60 + frame["Time_dt"].dt.minute
    frame["hour"] = frame["Time_dt"].dt.strftime("%H:%M")
    frame["crime"] = frame["Crime type"].astype(str).str.strip()
    frame["weight"] = frame["crime"].map(crime_risk_points).astype(float)
    frame["day"] = frame["Day"]
    frame["lat"] = frame["Latitude"].astype(float)
    frame["long"] = frame["Longitude"].astype(float)
    return frame


df = normalize_hotspot_dataset(HOTSPOT_DATA_FILE)


def parse_lat_lng_pairs(raw_value):
    if pd.isna(raw_value):
        return []

    matches = re.findall(
        r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)",
        str(raw_value),
    )
    points = [[float(lat), float(lng)] for lat, lng in matches]
    if points and points[0] != points[-1]:
        points.append(points[0])
    return points


def calculate_polygon_center(points):
    if not points:
        return {"lat": None, "lng": None}

    unique_points = points[:-1] if len(points) > 1 and points[0] == points[-1] else points
    lat = sum(point[0] for point in unique_points) / len(unique_points)
    lng = sum(point[1] for point in unique_points) / len(unique_points)
    return {"lat": round(lat, 8), "lng": round(lng, 8)}


def load_thana_area_data():
    if not THANA_AREAS_FILE.exists():
        return {}

    sheet = pd.read_excel(THANA_AREAS_FILE)
    thana_data = {}

    for _, row in sheet.iterrows():
        thana_name = str(row.get("Thana") or "").strip()
        area_name = str(row.get("Area") or "").strip()
        polygon = parse_lat_lng_pairs(row.get("(lat, long)"))

        if not thana_name or not area_name or not polygon:
            continue

        area_entry = {
            "name": area_name,
            "polygon": polygon,
            "center": calculate_polygon_center(polygon),
        }
        thana_data.setdefault(thana_name, {"areas": []})["areas"].append(area_entry)

    for thana_name, payload in thana_data.items():
        area_centers = [
            area["center"]
            for area in payload["areas"]
            if area["center"]["lat"] is not None and area["center"]["lng"] is not None
        ]
        if area_centers:
            payload["center"] = {
                "lat": round(sum(item["lat"] for item in area_centers) / len(area_centers), 8),
                "lng": round(sum(item["lng"] for item in area_centers) / len(area_centers), 8),
            }
        else:
            payload["center"] = {"lat": None, "lng": None}

    return thana_data


THANA_AREA_DATA = load_thana_area_data()


def load_admin_data():
    if not THANA_ADMIN_DATA_FILE.exists():
        return {"thana_meta": {}, "areas": {}, "deleted_areas": {}}

    try:
        payload = json.loads(THANA_ADMIN_DATA_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"thana_meta": {}, "areas": {}, "deleted_areas": {}}

    return {
        "thana_meta": payload.get("thana_meta") or {},
        "areas": payload.get("areas") or {},
        "deleted_areas": payload.get("deleted_areas") or {},
    }


def save_admin_data(payload):
    THANA_ADMIN_DATA_FILE.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def merge_admin_data(base_data, admin_data):
    merged = json.loads(json.dumps(base_data))

    for thana_name, deleted_names in (admin_data.get("deleted_areas") or {}).items():
        if not isinstance(deleted_names, list):
            continue

        thana_entry = merged.setdefault(thana_name, {"areas": [], "center": {"lat": None, "lng": None}})
        deleted_lookup = {str(name).strip().lower() for name in deleted_names if str(name).strip()}
        thana_entry["areas"] = [
            area for area in thana_entry.get("areas", [])
            if str(area.get("name") or "").strip().lower() not in deleted_lookup
        ]

    for thana_name, areas in (admin_data.get("areas") or {}).items():
        if not isinstance(areas, list):
            continue

        thana_entry = merged.setdefault(thana_name, {"areas": [], "center": {"lat": None, "lng": None}})
        area_map = {area["name"]: area for area in thana_entry.get("areas", []) if area.get("name")}

        for area in areas:
            area_name = str(area.get("name") or "").strip()
            polygon = area.get("polygon") or []
            if not area_name or len(polygon) < 4:
                continue
            normalized_polygon = [[float(point[0]), float(point[1])] for point in polygon]
            if normalized_polygon[0] != normalized_polygon[-1]:
                normalized_polygon.append(normalized_polygon[0])

            area_map[area_name] = {
                "name": area_name,
                "polygon": normalized_polygon,
                "center": calculate_polygon_center(normalized_polygon),
            }

        thana_entry["areas"] = sorted(area_map.values(), key=lambda item: item["name"].lower())

    for thana_entry in merged.values():
        area_centers = [
            area["center"]
            for area in thana_entry.get("areas", [])
            if area["center"]["lat"] is not None and area["center"]["lng"] is not None
        ]
        if area_centers:
            thana_entry["center"] = {
                "lat": round(sum(item["lat"] for item in area_centers) / len(area_centers), 8),
                "lng": round(sum(item["lng"] for item in area_centers) / len(area_centers), 8),
            }
        else:
            thana_entry["center"] = {"lat": None, "lng": None}

    return merged


ADMIN_DATA = load_admin_data()
THANA_AREA_DATA = merge_admin_data(THANA_AREA_DATA, ADMIN_DATA)


def build_police_stations(thana_data):
    stations = []
    for thana_name in sorted(thana_data):
        center = thana_data[thana_name].get("center") or {}
        meta = POLICE_STATION_META.get(thana_name, {})
        admin_meta = (ADMIN_DATA.get("thana_meta") or {}).get(thana_name, {})
        stations.append(
            {
                "name": thana_name,
                "lat": admin_meta.get("lat", meta.get("lat", center.get("lat"))),
                "lng": admin_meta.get("lng", meta.get("lng", center.get("lng"))),
                "phone": admin_meta.get("phone", meta.get("phone", "")),
                "email": admin_meta.get("email", meta.get("email", "")),
            }
        )
    return stations


POLICE_STATIONS = build_police_stations(THANA_AREA_DATA)


def refresh_thana_runtime_data():
    global ADMIN_DATA, THANA_AREA_DATA, POLICE_STATIONS
    ADMIN_DATA = load_admin_data()
    THANA_AREA_DATA = merge_admin_data(load_thana_area_data(), ADMIN_DATA)
    POLICE_STATIONS = build_police_stations(THANA_AREA_DATA)

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


def nominatim_search(query, *, polygon=False, limit=1):
    params = {
        "format": "jsonv2",
        "limit": str(limit),
        "q": query,
    }
    if polygon:
        params["polygon_geojson"] = "1"

    url = f"{NOMINATIM_BASE_URL}?{urlencode(params)}"
    req = Request(
        url,
        headers={
            "User-Agent": "DhakaRouteFinder/1.0",
            "Accept": "application/json",
        },
    )
    with urlopen(req, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_point_payload(point):
    if not isinstance(point, dict):
        return None

    try:
        lat = float(point.get("lat"))
        lng = float(point.get("lng"))
    except (TypeError, ValueError):
        return None

    payload = {"lat": lat, "lng": lng}
    if point.get("name"):
        payload["name"] = str(point.get("name")).strip()
    return payload


def parse_route_points(points):
    if not isinstance(points, list):
        return []

    parsed = []
    for point in points:
        normalized = parse_point_payload(point)
        if normalized:
            parsed.append(normalized)
    return parsed

# =========================
@app.route("/")
def home():
    return render_template(
        "map.html",
        thana_geo_data=THANA_AREA_DATA,
        police_stations=POLICE_STATIONS,
    )


@app.route("/admin/thana")
def admin_thana():
    return render_template(
        "thana_admin.html",
        thana_geo_data=THANA_AREA_DATA,
        police_stations=POLICE_STATIONS,
    )


@app.route("/admin/thana-data")
def admin_thana_data():
    return jsonify(
        {
            "ok": True,
            "thana_geo_data": THANA_AREA_DATA,
            "police_stations": POLICE_STATIONS,
        }
    )


@app.route("/admin/update-thana-location", methods=["POST"])
def admin_update_thana_location():
    payload = request.get_json(silent=True) or {}
    thana_name = str(payload.get("thana_name") or "").strip()

    try:
        lat = float(payload.get("lat"))
        lng = float(payload.get("lng"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Valid lat/lng are required."}), 400

    if not thana_name:
        return jsonify({"ok": False, "error": "Thana name is required."}), 400

    admin_data = load_admin_data()
    admin_data.setdefault("thana_meta", {}).setdefault(thana_name, {})
    admin_data["thana_meta"][thana_name]["lat"] = lat
    admin_data["thana_meta"][thana_name]["lng"] = lng
    save_admin_data(admin_data)
    refresh_thana_runtime_data()

    station = next((item for item in POLICE_STATIONS if item["name"] == thana_name), None)
    return jsonify({"ok": True, "station": station})


@app.route("/admin/add-area", methods=["POST"])
def admin_add_area():
    payload = request.get_json(silent=True) or {}
    thana_name = str(payload.get("thana_name") or "").strip()
    area_name = str(payload.get("area_name") or "").strip()
    original_area_name = str(payload.get("original_area_name") or "").strip()
    polygon = payload.get("polygon") or []

    if not thana_name or not area_name:
        return jsonify({"ok": False, "error": "Thana name and area name are required."}), 400
    if not isinstance(polygon, list) or len(polygon) < 3:
        return jsonify({"ok": False, "error": "At least 3 boundary points are required."}), 400

    normalized_polygon = []
    for point in polygon:
        if not isinstance(point, list) or len(point) != 2:
            return jsonify({"ok": False, "error": "Boundary points must be [lat, lng]."}), 400
        try:
            normalized_polygon.append([float(point[0]), float(point[1])])
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "Boundary points must be numeric."}), 400

    if normalized_polygon[0] != normalized_polygon[-1]:
        normalized_polygon.append(normalized_polygon[0])

    admin_data = load_admin_data()
    area_store = admin_data.setdefault("areas", {}).setdefault(thana_name, [])
    deleted_area_store = admin_data.setdefault("deleted_areas", {}).setdefault(thana_name, [])

    if (
        original_area_name
        and original_area_name.lower() != area_name.lower()
        and any(
            str(existing.get("name") or "").strip().lower() == area_name.lower()
            for existing in area_store
        )
    ):
        return jsonify({"ok": False, "error": "Another area already uses this name."}), 400

    match_name = (original_area_name or area_name).lower()
    replaced = False
    for index, existing in enumerate(area_store):
        if str(existing.get("name") or "").strip().lower() == match_name:
            area_store[index] = {"name": area_name, "polygon": normalized_polygon}
            replaced = True
            break
    if not replaced:
        area_store.append({"name": area_name, "polygon": normalized_polygon})

    admin_data["deleted_areas"][thana_name] = [
        existing_name
        for existing_name in deleted_area_store
        if str(existing_name).strip().lower() not in {area_name.lower(), original_area_name.lower()}
    ]

    save_admin_data(admin_data)
    refresh_thana_runtime_data()

    thana_payload = THANA_AREA_DATA.get(thana_name, {"areas": []})
    return jsonify({"ok": True, "thana": thana_payload})


@app.route("/admin/delete-area", methods=["POST"])
def admin_delete_area():
    payload = request.get_json(silent=True) or {}
    thana_name = str(payload.get("thana_name") or "").strip()
    area_name = str(payload.get("area_name") or "").strip()

    if not thana_name or not area_name:
        return jsonify({"ok": False, "error": "Thana name and area name are required."}), 400

    admin_data = load_admin_data()
    area_store = admin_data.setdefault("areas", {}).setdefault(thana_name, [])
    original_count = len(area_store)
    filtered_areas = [
        existing
        for existing in area_store
        if str(existing.get("name") or "").strip().lower() != area_name.lower()
    ]
    deleted_area_store = admin_data.setdefault("deleted_areas", {}).setdefault(thana_name, [])
    deleted_lookup = {str(name).strip().lower() for name in deleted_area_store if str(name).strip()}
    area_found = len(filtered_areas) != original_count

    base_thana = load_thana_area_data().get(thana_name, {"areas": []})
    if any(str(area.get("name") or "").strip().lower() == area_name.lower() for area in base_thana.get("areas", [])):
        area_found = True
        if area_name.lower() not in deleted_lookup:
            deleted_area_store.append(area_name)

    if not area_found:
        return jsonify({"ok": False, "error": "Selected area was not found."}), 404

    admin_data["areas"][thana_name] = filtered_areas
    save_admin_data(admin_data)
    refresh_thana_runtime_data()

    thana_payload = THANA_AREA_DATA.get(thana_name, {"areas": []})
    return jsonify({"ok": True, "thana": thana_payload})

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

    time_window_minutes = 120
    day_matches = df[df["day"] == day].copy()

    if day_matches.empty:
        return jsonify([])

    day_matches["minute_gap"] = (day_matches["time_minutes"] - user_min).abs()
    filtered = day_matches[day_matches["minute_gap"] <= time_window_minutes]
    if filtered.empty:
        filtered = day_matches.nsmallest(25, "minute_gap")

    data = []
    for _, row in filtered.iterrows():
        data.append({
            "lat": float(row["lat"]),
            "lng": float(row["long"]),
            "weight": float(row["weight"]),
            "crime": row["crime"],
            "time_range": row["hour"]
        })

    return jsonify(data)


@app.route("/predict_hotspots", methods=["GET"])
def predict_hotspots():
    date = str(request.args.get("date") or "").strip() or None
    time = str(request.args.get("time") or "").strip() or None
    try:
        limit = max(1, min(int(request.args.get("limit", "10")), 20))
    except ValueError:
        limit = 10

    result = ROUTE_HOTSPOT_PREDICTOR.predict_hotspots(
        date_text=date,
        time_text=time,
        limit=limit,
    )

    status_code = 200 if result.get("ok") else 400
    if not ROUTE_HOTSPOT_PREDICTOR.ready and not result.get("ok"):
        status_code = 503
    return jsonify(result), status_code


@app.route("/search_area_boundary")
def search_area_boundary():
    query = (request.args.get("q") or "").strip()
    polygon = request.args.get("polygon") == "1"
    try:
        limit = max(1, min(int(request.args.get("limit", "5")), 10))
    except ValueError:
        limit = 5

    if not query:
        return jsonify({"ok": False, "error": "Query is required."}), 400

    try:
        results = nominatim_search(query, polygon=polygon, limit=limit)
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Boundary lookup failed: {exc}"}), 502

    return jsonify({"ok": True, "results": results})

# =========================
@app.route("/send_sos_email", methods=["POST"])
def send_sos_email():
    payload = request.get_json(silent=True) or {}

    my_contact_no = (payload.get("my_contact_no") or "").strip()
    to_email = (payload.get("to_email") or "").strip()
    body = (payload.get("body") or "").strip()
    stationary_minutes = payload.get("stationary_minutes")
    countdown_minutes = payload.get("countdown_minutes")
    location = payload.get("location") or {}

    if not my_contact_no or not to_email or not body:
        return jsonify({"ok": False, "error": "Missing required contact or email fields."}), 400

    if location.get("lat") is None or location.get("lng") is None:
        return jsonify({"ok": False, "error": "Last known location is required."}), 400

    smtp_host = os.getenv("SMTP_HOST") or "smtp.gmail.com"
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    smtp_from = os.getenv("SMTP_FROM") or smtp_username
    smtp_use_tls = os.getenv("SMTP_USE_TLS", "true").lower() != "false"

    if not smtp_host or not smtp_username or not smtp_password:
        return jsonify({"ok": False, "error": "SMTP is not configured on the server."}), 500

    location_text = f'{location["lat"]}, {location["lng"]}'

    message = EmailMessage()
    message["Subject"] = "SOS Rescue Alert"
    message["From"] = smtp_from
    message["To"] = to_email
    message.set_content(
        f"SOS rescue mail triggered.\n\n"
        f"My Contact No.: {my_contact_no}\n"
        f"To Mail: {to_email}\n"
        f"My Last known location: {location_text}\n\n"
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
