# -*- coding: utf-8 -*-
import os

from lightrag.product_accounts import (
    issue_refresh_token,
    list_device_sessions,
    upload_capability,
)


def test_device_session_marks_current_and_last_used():
    data = {"device_sessions": []}
    _raw, session_id = issue_refresh_token(data, "user-1", "测试手机")
    rows = list_device_sessions(data, "user-1", session_id)
    assert rows[0]["is_current"] is True
    assert rows[0]["last_used_at"]
    assert rows[0]["device_label"] == "测试手机"
    assert "token_hash" not in rows[0]
    other = list_device_sessions(data, "user-1", "someone-else")
    assert other[0]["is_current"] is False


def test_upload_capability_reads_env(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "7")
    monkeypatch.setenv("MAX_FILES_PER_KB", "3")
    monkeypatch.setenv("EMBEDDING_MODEL", "unit-embed")
    monkeypatch.setenv("RERANK_MODEL", "")
    info = upload_capability()
    assert info["max_upload_bytes"] == 7 * 1024 * 1024
    assert info["max_files_per_kb"] == 3
    assert info["embedding_model"] == "unit-embed"
    assert info["rerank_model"] == ""
    assert ".pdf" in info["extensions"]
    assert "bge-reranker" not in os.getenv("RERANK_MODEL", "")
