from __future__ import annotations

import sqlite3
import multiprocessing
from pathlib import Path

from lightrag.product_appdb import (
    appdb_integrity_check,
    backup_appdb,
    open_appdb,
    reset_appdb_cache,
    restore_appdb,
)


def _concurrent_memory_writer(path: str, prefix: str) -> None:
    store = open_appdb(Path(path))
    for index in range(20):
        store.add_memory(owner_id=prefix, content=f"{prefix}-{index}")
    reset_appdb_cache()


def test_memory_scope_retrieval_filters_and_budgets(tmp_path):
    store = open_appdb(tmp_path)
    global_memory = store.add_memory(
        owner_id="user-a",
        content="我喜欢简洁的回答",
        scope="global",
        embedding=[1.0, 0.0],
        embedding_model="test-model",
    )
    kb_memory = store.add_memory(
        owner_id="user-a",
        content="这个资料库讨论水泵维护",
        scope="kb",
        kb_id="kb-a",
        embedding=[0.9, 0.1],
        embedding_model="test-model",
    )
    store.add_memory(
        owner_id="user-a",
        content="另一个资料库的秘密",
        scope="kb",
        kb_id="kb-b",
        embedding=[1.0, 0.0],
        embedding_model="test-model",
    )
    store.add_memory(
        owner_id="user-b",
        content="其他用户的偏好",
        scope="global",
        embedding=[1.0, 0.0],
        embedding_model="test-model",
    )
    selected = store.retrieve_memories(
        "user-a",
        "kb-a",
        "简洁回答 水泵",
        top_k=5,
        token_budget=20,
        embedding=[1.0, 0.0],
        embedding_model="test-model",
    )
    assert {item["id"] for item in selected} == {global_memory["id"], kb_memory["id"]}
    assert all(
        "memory_score" in item and item["scope"] in {"global", "kb"}
        for item in selected
    )
    assert not any(item["kb_id"] == "kb-b" for item in selected)
    assert not store.retrieve_memories(
        "user-a", "kb-a", "秘密", embedding=[0.0, 1.0], embedding_model="test-model"
    )

    store.update_memory(global_memory["id"], "user-a", enabled=False)
    assert all(
        item["id"] != global_memory["id"]
        for item in store.retrieve_memories(
            "user-a", "kb-a", "简洁", embedding=[1.0, 0.0], embedding_model="test-model"
        )
    )
    reset_appdb_cache()


def test_memory_migration_and_backup_restore(tmp_path):
    legacy = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(legacy)
    conn.executescript(
        """
        CREATE TABLE schema_migrations(version TEXT PRIMARY KEY, applied_at TEXT NOT NULL);
        INSERT INTO schema_migrations VALUES ('app-001', '2026-01-01T00:00:00+00:00');
        INSERT INTO schema_migrations VALUES ('app-002', '2026-01-01T00:00:01+00:00');
        CREATE TABLE conversations (id TEXT PRIMARY KEY, owner_id TEXT, kb_id TEXT, title TEXT, summary TEXT, is_pinned INTEGER, is_archived INTEGER, created_at TEXT, updated_at TEXT, last_message_at TEXT);
        CREATE TABLE messages (id TEXT PRIMARY KEY, conversation_id TEXT, owner_id TEXT, kb_id TEXT, role TEXT, content TEXT, status TEXT, citations_json TEXT, retrieval_meta_json TEXT, index_version TEXT, embedding_model TEXT, rerank_model TEXT, llm_model TEXT, created_at TEXT);
        CREATE TABLE answer_feedback (id TEXT PRIMARY KEY, message_id TEXT, owner_id TEXT, kb_id TEXT, rating TEXT, reason TEXT, comment TEXT, question TEXT, answer TEXT, citations_json TEXT, index_version TEXT, embedding_model TEXT, rerank_model TEXT, llm_model TEXT, created_at TEXT);
        CREATE TABLE offline_eval_candidates (id TEXT PRIMARY KEY, feedback_id TEXT, owner_id TEXT, kb_id TEXT, payload_json TEXT, created_at TEXT);
        CREATE TABLE memory_candidates (id TEXT PRIMARY KEY, owner_id TEXT, kb_id TEXT, conversation_id TEXT, message_id TEXT, content TEXT, category TEXT, source_refs_json TEXT, status TEXT, created_at TEXT);
        CREATE TABLE memories (id TEXT PRIMARY KEY, owner_id TEXT, content TEXT, category TEXT, source_type TEXT, source_id TEXT, enabled INTEGER, created_at TEXT, updated_at TEXT);
        INSERT INTO memories VALUES ('legacy-1', 'user-a', '旧记忆', 'other', 'candidate', '', 1, '2026-01-01', '2026-01-01');
        """
    )
    conn.commit()
    conn.close()
    working = tmp_path / "working"
    working.mkdir()
    (working / "product_app.sqlite").write_bytes(legacy.read_bytes())
    store = open_appdb(working)
    assert "app-004" in store.schema_versions()
    assert store.get_memory("legacy-1", "user-a")["scope"] == "global"
    backup = tmp_path / "backup.sqlite"
    backup_appdb(working, backup)
    assert appdb_integrity_check(working) == "ok"
    store.add_memory(owner_id="user-a", content="恢复后记忆")
    reset_appdb_cache()
    restore_appdb(working, backup)
    restored = open_appdb(working)
    assert restored.get_memory("legacy-1", "user-a")["content"] == "旧记忆"
    assert not restored.get_memory("legacy-1", "user-a")["embedding"]
    assert appdb_integrity_check(working) == "ok"
    reset_appdb_cache()


def test_two_process_writers_keep_all_memory_rows(tmp_path):
    working = tmp_path / "concurrent"
    working.mkdir()
    first = multiprocessing.Process(
        target=_concurrent_memory_writer, args=(str(working), "writer-a")
    )
    second = multiprocessing.Process(
        target=_concurrent_memory_writer, args=(str(working), "writer-b")
    )
    first.start()
    second.start()
    first.join(30)
    second.join(30)
    assert first.exitcode == 0
    assert second.exitcode == 0
    store = open_appdb(working)
    assert len(store.list_memories("writer-a")) == 20
    assert len(store.list_memories("writer-b")) == 20
    assert appdb_integrity_check(working) == "ok"
    reset_appdb_cache()
