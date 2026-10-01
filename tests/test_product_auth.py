# -*- coding: utf-8 -*-
import os

from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightrag.product_accounts import (
    content_matches_extension,
    create_auth_router,
    production_startup_error,
)


def _client(monkeypatch, enabled: str = "1"):
    monkeypatch.setenv("PRODUCT_AUTH", enabled)
    monkeypatch.setenv("TOKEN_SECRET", "unit-test-secret")
    store: dict = {"users": [], "device_sessions": []}

    async def read_data():
        return store

    async def write_data(data):
        if data is not store:
            store.clear()
            store.update(data)

    app = FastAPI()
    app.include_router(create_auth_router(read_data, write_data), prefix="/api/v1")
    return TestClient(app), store


def test_auth_routes_are_hidden_when_disabled(monkeypatch):
    client, _store = _client(monkeypatch, "0")
    assert client.get("/api/v1/auth/status").json()["enabled"] is False
    assert client.post("/api/v1/auth/register", json={"email": "ada@example.com", "password": "correct-horse"}).status_code == 404


def test_register_login_refresh_and_logout(monkeypatch):
    client, store = _client(monkeypatch, "1")
    csrf = {"X-KB-Request": "1"}
    missing_header = client.post("/api/v1/auth/register", json={"email": "Ada@Example.com", "password": "correct-horse"})
    assert missing_header.status_code == 403
    created = client.post(
        "/api/v1/auth/register",
        json={"email": "Ada@Example.com", "password": "correct-horse"},
        headers=csrf,
    )
    assert created.status_code == 200
    body = created.json()
    assert body["user"]["email"] == "ada@example.com"
    assert "password_hash" not in body
    assert "refresh_token" not in body
    assert "scrypt$" not in created.text
    assert created.cookies.get("kb_refresh")
    first_cookie = created.cookies.get("kb_refresh")

    wrong = client.post("/api/v1/auth/login", json={"email": "missing@example.com", "password": "correct-horse"}, headers=csrf)
    mismatch = client.post("/api/v1/auth/login", json={"email": "ada@example.com", "password": "wrong-password"}, headers=csrf)
    assert wrong.status_code == mismatch.status_code == 401
    assert wrong.json()["detail"] == mismatch.json()["detail"] == "邮箱或密码错误"

    me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert me.status_code == 200
    assert me.json()["user_id"] == body["user"]["user_id"]
    assert client.get("/api/v1/auth/me").status_code == 401

    refreshed = client.post("/api/v1/auth/refresh", headers=csrf)
    assert refreshed.status_code == 200
    assert "refresh_token" not in refreshed.json()
    replay = client.post("/api/v1/auth/refresh", headers=csrf, cookies={"kb_refresh": first_cookie})
    assert replay.status_code == 401

    logout = client.post(
        "/api/v1/auth/logout",
        headers={**csrf, "Authorization": f"Bearer {refreshed.json()['access_token']}"},
    )
    assert logout.status_code == 401 or all(item.get("revoked_at") for item in store["device_sessions"])


def test_production_refuses_to_start_without_real_accounts(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("PRODUCT_AUTH", raising=False)
    monkeypatch.delenv("TOKEN_SECRET", raising=False)
    assert production_startup_error() is None

    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("PRODUCT_AUTH", "0")
    assert "PRODUCT_AUTH" in (production_startup_error() or "")

    monkeypatch.setenv("PRODUCT_AUTH", "1")
    monkeypatch.setenv("TOKEN_SECRET", "lightrag-jwt-default-secret")
    assert "TOKEN_SECRET" in (production_startup_error() or "")

    monkeypatch.setenv("TOKEN_SECRET", "unit-test-secret")
    assert "CORS" in (production_startup_error() or "")
    monkeypatch.setenv("CORS_ORIGINS", "http://127.0.0.1:9621")
    assert production_startup_error() is None
    monkeypatch.delenv("RETRIEVAL_PROFILE", raising=False)
    monkeypatch.setenv("EMBEDDING_MODEL", "unknown-embed")
    from lightrag.retrieval_admission import retrieval_profile_startup_error

    assert "没有匹配的检索画像" in (retrieval_profile_startup_error() or "")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    assert retrieval_profile_startup_error() is None
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("EMBEDDING_MODEL", "unknown-embed")
    assert retrieval_profile_startup_error() is None
    assert "sk-" not in (production_startup_error() or "")
    assert os.getenv("TOKEN_SECRET") == "unit-test-secret"


def test_upload_rejects_mismatched_content():
    assert content_matches_extension("note.pdf", b"%PDF-1.4")
    assert not content_matches_extension("note.pdf", b"MZ fake")
    assert not content_matches_extension("note.txt", b"MZ")
