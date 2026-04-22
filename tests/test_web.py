import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import web


def test_get_hotspots_endpoint(client=None):
    test_client = web.app.test_client()
    response = test_client.get("/get_hotspots")
    assert response.status_code == 200
    assert isinstance(response.get_json(), list)


def test_send_sos_email_requires_fields():
    test_client = web.app.test_client()
    response = test_client.post("/send_sos_email", json={})
    assert response.status_code == 400
    assert response.get_json()["ok"] is False


def test_send_sos_email_requires_location():
    test_client = web.app.test_client()
    response = test_client.post(
        "/send_sos_email",
        json={
            "my_contact_no": "01700111222",
            "to_email": "rescue@example.com",
            "body": "Please help me.",
        },
    )
    assert response.status_code == 400
    assert response.get_json()["ok"] is False


def test_send_sos_email_success(monkeypatch):
    sent_messages = []

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            self.host = host
            self.port = port
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def starttls(self):
            return None

        def login(self, username, password):
            self.username = username
            self.password = password

        def send_message(self, message):
            sent_messages.append(message)

    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USERNAME", "server@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "secret")
    monkeypatch.setattr(web.smtplib, "SMTP", FakeSMTP)

    test_client = web.app.test_client()
    response = test_client.post(
        "/send_sos_email",
        json={
            "my_contact_no": "01700111222",
            "to_email": "rescue@example.com",
            "body": "Please help me.",
            "stationary_minutes": 3,
            "countdown_minutes": 2,
            "location": {"lat": 23.8, "lng": 90.4},
        },
    )

    assert response.status_code == 200
    assert response.get_json()["ok"] is True
    assert len(sent_messages) == 1
    assert sent_messages[0]["To"] == "rescue@example.com"


def test_search_area_boundary_requires_query():
    test_client = web.app.test_client()
    response = test_client.get("/search_area_boundary")
    assert response.status_code == 400
    assert response.get_json()["ok"] is False


def test_search_area_boundary_success(monkeypatch):
    def fake_nominatim_search(query, *, polygon=False, limit=1):
        assert query == "Motijheel, Dhaka, Bangladesh"
        assert polygon is True
        assert limit == 5
        return [{"display_name": query, "geojson": {"type": "Polygon", "coordinates": []}}]

    monkeypatch.setattr(web, "nominatim_search", fake_nominatim_search)

    test_client = web.app.test_client()
    response = test_client.get("/search_area_boundary?q=Motijheel, Dhaka, Bangladesh&polygon=1")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert len(payload["results"]) == 1


def test_search_area_boundary_honors_limit(monkeypatch):
    def fake_nominatim_search(query, *, polygon=False, limit=1):
        assert query == "Dilkusha, Dhaka, Bangladesh"
        assert polygon is False
        assert limit == 3
        return [{"display_name": query}]

    monkeypatch.setattr(web, "nominatim_search", fake_nominatim_search)

    test_client = web.app.test_client()
    response = test_client.get("/search_area_boundary?q=Dilkusha, Dhaka, Bangladesh&limit=3")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert len(payload["results"]) == 1


def test_predict_hotspots_endpoint(monkeypatch):
    captured = {}

    class FakePredictor:
        ready = True

        def predict_hotspots(self, **kwargs):
            captured.update(kwargs)
            return {
                "ok": True,
                "prediction": {
                    "overall_risk_score": 0.63,
                    "overall_risk_percent": 63.0,
                    "overall_risk_level": "Medium",
                    "day_of_week": "Wednesday",
                    "date": "2025-07-23",
                    "time": "22:00",
                    "flagged_hotspot_count": 2,
                    "candidate_place_count": 4,
                    "hotspot_candidates": [
                        {
                            "place": "Newmarket",
                            "lat": 23.734,
                            "lng": 90.385,
                            "risk_score": 0.72,
                            "risk_percent": 72.0,
                            "risk_level": "High",
                            "nearest_place_distance_km": 0.31,
                        }
                    ],
                    "model": {"name": "RandomForestClassifier", "benchmark": {"f1": 0.71}},
                },
            }

    monkeypatch.setattr(web, "ROUTE_HOTSPOT_PREDICTOR", FakePredictor())

    test_client = web.app.test_client()
    response = test_client.get("/predict_hotspots?date=2025-07-23&time=22:00&limit=7")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["prediction"]["overall_risk_level"] == "Medium"
    assert payload["prediction"]["hotspot_candidates"][0]["place"] == "Newmarket"
    assert captured["date_text"] == "2025-07-23"
    assert captured["time_text"] == "22:00"
    assert captured["limit"] == 7


def test_admin_update_thana_location_persists(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "THANA_ADMIN_DATA_FILE", tmp_path / "thana_admin_data.json")
    web.refresh_thana_runtime_data()

    test_client = web.app.test_client()
    response = test_client.post(
        "/admin/update-thana-location",
        json={"thana_name": "Gulshan Thana", "lat": 23.7999, "lng": 90.4333},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["station"]["lat"] == 23.7999
    assert payload["station"]["lng"] == 90.4333

    saved = web.load_admin_data()
    assert saved["thana_meta"]["Gulshan Thana"]["lat"] == 23.7999
    assert saved["thana_meta"]["Gulshan Thana"]["lng"] == 90.4333


def test_admin_add_area_persists(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "THANA_ADMIN_DATA_FILE", tmp_path / "thana_admin_data.json")
    web.refresh_thana_runtime_data()

    test_client = web.app.test_client()
    response = test_client.post(
        "/admin/add-area",
        json={
            "thana_name": "Gulshan Thana",
            "area_name": "Test Area",
            "polygon": [
                [23.8, 90.4],
                [23.8005, 90.401],
                [23.799, 90.402],
            ],
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert any(area["name"] == "Test Area" for area in payload["thana"]["areas"])

    saved = web.load_admin_data()
    assert saved["areas"]["Gulshan Thana"][0]["name"] == "Test Area"
    assert len(saved["areas"]["Gulshan Thana"][0]["polygon"]) == 4


def test_admin_add_area_can_rename_existing_area(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "THANA_ADMIN_DATA_FILE", tmp_path / "thana_admin_data.json")
    web.save_admin_data(
        {
            "thana_meta": {},
            "areas": {
                "Gulshan Thana": [
                    {
                        "name": "Old Area",
                        "polygon": [
                            [23.8, 90.4],
                            [23.8005, 90.401],
                            [23.799, 90.402],
                            [23.8, 90.4],
                        ],
                    }
                ]
            },
        }
    )
    web.refresh_thana_runtime_data()

    test_client = web.app.test_client()
    response = test_client.post(
        "/admin/add-area",
        json={
            "thana_name": "Gulshan Thana",
            "original_area_name": "Old Area",
            "area_name": "Renamed Area",
            "polygon": [
                [23.81, 90.41],
                [23.8105, 90.411],
                [23.809, 90.412],
            ],
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert any(area["name"] == "Renamed Area" for area in payload["thana"]["areas"])
    assert not any(area["name"] == "Old Area" for area in payload["thana"]["areas"])

    saved = web.load_admin_data()
    assert saved["areas"]["Gulshan Thana"][0]["name"] == "Renamed Area"


def test_admin_add_area_rejects_rename_to_existing_name(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "THANA_ADMIN_DATA_FILE", tmp_path / "thana_admin_data.json")
    web.save_admin_data(
        {
            "thana_meta": {},
            "areas": {
                "Gulshan Thana": [
                    {
                        "name": "Area One",
                        "polygon": [
                            [23.8, 90.4],
                            [23.8005, 90.401],
                            [23.799, 90.402],
                            [23.8, 90.4],
                        ],
                    },
                    {
                        "name": "Area Two",
                        "polygon": [
                            [23.81, 90.41],
                            [23.8105, 90.411],
                            [23.809, 90.412],
                            [23.81, 90.41],
                        ],
                    },
                ]
            },
        }
    )
    web.refresh_thana_runtime_data()

    test_client = web.app.test_client()
    response = test_client.post(
        "/admin/add-area",
        json={
            "thana_name": "Gulshan Thana",
            "original_area_name": "Area One",
            "area_name": "Area Two",
            "polygon": [
                [23.82, 90.42],
                [23.8205, 90.421],
                [23.819, 90.422],
            ],
        },
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["ok"] is False
    assert payload["error"] == "Another area already uses this name."


def test_admin_delete_area_removes_existing_area(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "THANA_ADMIN_DATA_FILE", tmp_path / "thana_admin_data.json")
    web.save_admin_data(
        {
            "thana_meta": {},
            "areas": {
                "Gulshan Thana": [
                    {
                        "name": "Area One",
                        "polygon": [
                            [23.8, 90.4],
                            [23.8005, 90.401],
                            [23.799, 90.402],
                            [23.8, 90.4],
                        ],
                    },
                    {
                        "name": "Area Two",
                        "polygon": [
                            [23.81, 90.41],
                            [23.8105, 90.411],
                            [23.809, 90.412],
                            [23.81, 90.41],
                        ],
                    },
                ]
            },
        }
    )
    web.refresh_thana_runtime_data()

    test_client = web.app.test_client()
    response = test_client.post(
        "/admin/delete-area",
        json={
            "thana_name": "Gulshan Thana",
            "area_name": "Area One",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert not any(area["name"] == "Area One" for area in payload["thana"]["areas"])
    assert any(area["name"] == "Area Two" for area in payload["thana"]["areas"])

    saved = web.load_admin_data()
    assert len(saved["areas"]["Gulshan Thana"]) == 1
    assert saved["areas"]["Gulshan Thana"][0]["name"] == "Area Two"


def test_admin_delete_area_returns_404_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "THANA_ADMIN_DATA_FILE", tmp_path / "thana_admin_data.json")
    web.save_admin_data(
        {
            "thana_meta": {},
            "areas": {
                "Gulshan Thana": [
                    {
                        "name": "Area One",
                        "polygon": [
                            [23.8, 90.4],
                            [23.8005, 90.401],
                            [23.799, 90.402],
                            [23.8, 90.4],
                        ],
                    }
                ]
            },
        }
    )
    web.refresh_thana_runtime_data()

    test_client = web.app.test_client()
    response = test_client.post(
        "/admin/delete-area",
        json={
            "thana_name": "Gulshan Thana",
            "area_name": "Missing Area",
        },
    )

    assert response.status_code == 404
    payload = response.get_json()
    assert payload["ok"] is False
    assert payload["error"] == "Selected area was not found."


def test_admin_delete_area_hides_base_area(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "THANA_ADMIN_DATA_FILE", tmp_path / "thana_admin_data.json")

    def fake_load_thana_area_data():
        return {
            "Gulshan Thana": {
                "areas": [
                    {
                        "name": "Base Area",
                        "polygon": [
                            [23.8, 90.4],
                            [23.8005, 90.401],
                            [23.799, 90.402],
                            [23.8, 90.4],
                        ],
                        "center": {"lat": 23.79983, "lng": 90.401},
                    }
                ],
                "center": {"lat": 23.79983, "lng": 90.401},
            }
        }

    monkeypatch.setattr(web, "load_thana_area_data", fake_load_thana_area_data)
    web.refresh_thana_runtime_data()

    test_client = web.app.test_client()
    response = test_client.post(
        "/admin/delete-area",
        json={
            "thana_name": "Gulshan Thana",
            "area_name": "Base Area",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert not any(area["name"] == "Base Area" for area in payload["thana"]["areas"])

    saved = web.load_admin_data()
    assert saved["deleted_areas"]["Gulshan Thana"] == ["Base Area"]
