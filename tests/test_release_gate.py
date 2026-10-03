# -*- coding: utf-8 -*-
import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightrag.product_accounts import (
    inspect_upload,
    issue_access_token,
    register_user,
    transfer_local_owner,
    AccountError,
)
from lightrag.product_deletion import (
    apply_metadata_purge,
    begin_purge,
    consistency_report,
    filenames_used_by_others,
    mark_job,
    queue_retry,
)
from lightrag.product_guard import access_decision, install_product_guard
from lightrag.product_migrate import migrate

sys.argv = ["lightrag"]
from lightrag.api.routers.product_shell import _record_in_kb, scoped_graph_records


PRODUCT_PATHS = [
    "/api/v1/kb",
    "/api/v1/documents",
    "/api/v1/chunks",
    "/api/v1/qa",
    "/api/v1/chat",
    "/api/v1/chat/search-test",
    "/api/v1/graph/stats",
    "/api/v1/kb/comparison",
    "/api/v1/kb/audit",
]
NATIVE_PATHS = [
    "/documents",
    "/documents/upload",
    "/query",
    "/query/stream",
    "/query/data",
    "/graphs",
    "/graph/label/list",
    "/api/chat",
    "/api/generate",
]
PUBLIC_PATHS = [
    "/health",
    "/auth-status",
    "/console",
    "/console/assets/app.js",
    "/api/v1/auth/login",
    "/api/v1/auth/register",
    "/api/v1/auth/refresh",
]


def test_unauthenticated_matrix_and_native_block(monkeypatch):
    monkeypatch.setenv("PRODUCT_AUTH", "1")
    monkeypatch.setenv("TOKEN_SECRET", "unit-test-secret-with-32-bytes!")
    for path in PRODUCT_PATHS + NATIVE_PATHS:
        assert access_decision(path, None) == "unauthorized", path
    for path in PUBLIC_PATHS:
        assert access_decision(path, None) == "allow", path
    token = issue_access_token("user-a", "session-a", 0)
    for path in NATIVE_PATHS:
        assert access_decision(path, f"Bearer {token}") == "forbidden", path
    for path in PRODUCT_PATHS:
        assert access_decision(path, f"Bearer {token}") == "allow", path
    assert access_decision("/api/v1/kb", "Bearer guest-token") == "unauthorized"
    monkeypatch.setenv("PRODUCT_AUTH", "0")
    assert access_decision("/query", None) == "allow"


def test_guard_middleware_blocks_native_routes(monkeypatch):
    monkeypatch.setenv("PRODUCT_AUTH", "1")
    monkeypatch.setenv("TOKEN_SECRET", "unit-test-secret-with-32-bytes!")
    app = FastAPI()
    install_product_guard(app)

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/query")
    def query():
        return {"secret": "should-not-leak"}

    @app.get("/api/v1/kb")
    def kb():
        return {"ok": True}

    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.post("/query", json={"query": "x"}).status_code == 401
    token = issue_access_token("user-a", "session-a", 0)
    blocked = client.post("/query", headers={"Authorization": f"Bearer {token}"})
    assert blocked.status_code == 403
    assert "should-not-leak" not in blocked.text
    assert client.get("/api/v1/kb").status_code == 401
    assert (
        client.get(
            "/api/v1/kb", headers={"Authorization": f"Bearer {token}"}
        ).status_code
        == 200
    )


def test_missing_scope_is_not_recalled_across_users_or_kbs():
    bindings = {"a.txt": "kb-a", "b.txt": "kb-b"}
    assert _record_in_kb("", "kb-a", bindings) is False
    assert _record_in_kb("a.txt", "kb-b", bindings) is False
    assert _record_in_kb("a.txt", "kb-a", bindings) is True
    nodes = [
        {"id": "甲", "file_path": "a.txt"},
        {"id": "乙", "file_path": "b.txt"},
        {"id": "无归属", "file_path": ""},
    ]
    edges = [
        {"source": "甲", "target": "乙", "file_path": "b.txt"},
        {"source": "甲", "target": "甲", "file_path": ""},
        {"source": "甲", "target": "甲", "file_path": "a.txt", "kb_id": "kb-a"},
    ]
    kept_nodes, kept_edges = scoped_graph_records(nodes, edges, "kb-a", bindings)
    assert [item["id"] for item in kept_nodes] == ["甲"]
    assert kept_edges == [edges[2]]
    assert all(
        "b.txt" not in str(item.get("file_path") or "")
        for item in kept_nodes + kept_edges
    )


def test_delete_keeps_other_kb_file_and_can_retry(tmp_path: Path):
    data = {
        "kbs": [
            {"id": "kb-a", "owner_id": "user-a"},
            {"id": "kb-b", "owner_id": "user-a"},
        ],
        "file_bindings": {"a.txt": "kb-a", "same-name.txt": "kb-b"},
        "qa_pairs": [{"id": "q1", "kb_id": "kb-a"}, {"id": "q2", "kb_id": "kb-b"}],
        "comparisons": [],
        "audits": {},
        "sessions": {},
        "purge_jobs": [],
    }
    protected = filenames_used_by_others(data, "kb-a")
    assert "same-name.txt" in protected
    assert "a.txt" not in protected
    job = begin_purge(data, data["kbs"][0])
    cache = apply_metadata_purge(
        data,
        "kb-a",
        [
            {"kb_id": "kb-a", "user_id": "user-a"},
            {"kb_id": "kb-b", "user_id": "user-a"},
        ],
    )
    assert data["file_bindings"] == {"same-name.txt": "kb-b"}
    assert cache == [{"kb_id": "kb-b", "user_id": "user-a"}]
    report = consistency_report(data, "kb-a", cache)
    assert report["clean"] is True
    assert consistency_report(data, "kb-b", cache)["bindings"] == ["same-name.txt"]
    mark_job(job, ok=False, error="中断", leftover=["a.txt"])
    queue_retry(job)
    mark_job(job, ok=True, error="", leftover=[])
    assert job["status"] == "succeeded"
    assert job["attempts"] == 2
    with pytest.raises(ValueError):
        queue_retry(job)
    assert tmp_path.exists()


def test_owner_migration_is_idempotent(tmp_path: Path):
    folder = tmp_path / "rag"
    folder.mkdir()
    data = {
        "kbs": [{"id": "legacy", "owner_id": "local-owner"}],
        "users": [],
        "file_bindings": {"a.txt": "legacy"},
    }
    register_user(data, "ada@example.com", "correct-horse")
    (folder / "product_shell.json").write_text(json.dumps(data), encoding="utf-8")
    (folder / "semantic_cache.json").write_text(
        json.dumps([{"kb_id": "legacy", "user_id": "local-owner", "answer": "旧答案"}]),
        encoding="utf-8",
    )
    first = migrate(folder, "ada@example.com")
    second = migrate(folder, "ada@example.com")
    assert first["moved_kb_ids"] == ["legacy"]
    assert second["moved_kb_ids"] == []
    saved = json.loads((folder / "product_shell.json").read_text(encoding="utf-8"))
    assert saved["kbs"][0]["owner_id"] != "local-owner"
    assert saved["file_bindings"]["a.txt"] == "legacy"
    cache = json.loads((folder / "semantic_cache.json").read_text(encoding="utf-8"))
    assert cache[0]["user_id"] == saved["kbs"][0]["owner_id"]
    again = transfer_local_owner(saved, saved["kbs"][0]["owner_id"], cache)
    assert again["moved_kb_ids"] == []


def test_upload_limits_reject_oversize_zip_and_long_pdf(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    monkeypatch.setenv("MAX_PDF_PAGES", "2")
    with pytest.raises(AccountError) as size_error:
        inspect_upload("big.txt", b"a" * (1024 * 1024 + 1))
    assert size_error.value.code == "too_large"
    with pytest.raises(AccountError):
        inspect_upload("bomb.zip", b"PK\x03\x04")
    pages = b"%PDF-1.4 /Type /Page /Type /Page /Type /Page"
    with pytest.raises(AccountError) as page_error:
        inspect_upload("long.pdf", pages)
    assert page_error.value.code == "too_many_pages"
    assert inspect_upload("note.pdf", b"%PDF-1.4 /Type /Page") is None
