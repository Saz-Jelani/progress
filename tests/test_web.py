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
            "my_email": "me@example.com",
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
