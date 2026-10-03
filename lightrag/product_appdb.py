"""个人知识库应用数据库。

开发环境使用 SQLite，表结构按后续 PostgreSQL 迁移来写。
会话和消息不写入 product_shell.json。
知识库数据必须匹配 owner_id 和 kb_id；用户级记忆显式标记为 global，不能隐式跨用户共享。
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


SCHEMA_VERSION = "app-005"
APP_DB_NAME = "product_app.sqlite"

_STORES: dict[str, "AppStore"] = {}
_GUARD = threading.Lock()

_MIGRATIONS = (
    (
        "app-001",
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            kb_id TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL DEFAULT '',
            is_pinned INTEGER NOT NULL DEFAULT 0,
            is_archived INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_message_at TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_conversations_scope
            ON conversations(owner_id, kb_id, is_archived, is_pinned, last_message_at);
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            kb_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            status TEXT NOT NULL,
            citations_json TEXT NOT NULL DEFAULT '[]',
            retrieval_meta_json TEXT NOT NULL DEFAULT '{}',
            index_version TEXT NOT NULL DEFAULT '',
            embedding_model TEXT NOT NULL DEFAULT '',
            rerank_model TEXT NOT NULL DEFAULT '',
            llm_model TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_messages_scope
            ON messages(owner_id, kb_id, conversation_id, created_at);
        """,
    ),
    (
        "app-002",
        """
        CREATE TABLE IF NOT EXISTS answer_feedback (
            id TEXT PRIMARY KEY,
            message_id TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            kb_id TEXT NOT NULL,
            rating TEXT NOT NULL,
            reason TEXT NOT NULL DEFAULT '',
            comment TEXT NOT NULL DEFAULT '',
            question TEXT NOT NULL DEFAULT '',
            answer TEXT NOT NULL DEFAULT '',
            citations_json TEXT NOT NULL DEFAULT '[]',
            index_version TEXT NOT NULL DEFAULT '',
            embedding_model TEXT NOT NULL DEFAULT '',
            rerank_model TEXT NOT NULL DEFAULT '',
            llm_model TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            UNIQUE(message_id, owner_id)
        );
        CREATE INDEX IF NOT EXISTS idx_feedback_scope
            ON answer_feedback(owner_id, kb_id, message_id);
        CREATE TABLE IF NOT EXISTS offline_eval_candidates (
            id TEXT PRIMARY KEY,
            feedback_id TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            kb_id TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_offline_eval_scope
            ON offline_eval_candidates(owner_id, kb_id, created_at);
        """,
    ),
    (
        "app-003",
        """
        CREATE TABLE IF NOT EXISTS memory_candidates (
            id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            kb_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL DEFAULT '',
            message_id TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'other',
            source_refs_json TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_memory_candidates_scope
            ON memory_candidates(owner_id, kb_id, status, created_at);
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            content TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'other',
            source_type TEXT NOT NULL DEFAULT 'candidate',
            source_id TEXT NOT NULL DEFAULT '',
            enabled INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_memories_owner
            ON memories(owner_id, enabled, updated_at);
        """,
    ),
    (
        "app-004",
        """
        ALTER TABLE memories ADD COLUMN scope TEXT NOT NULL DEFAULT 'global';
        ALTER TABLE memories ADD COLUMN kb_id TEXT NOT NULL DEFAULT '';
        ALTER TABLE memories ADD COLUMN expires_at TEXT NOT NULL DEFAULT '';
        ALTER TABLE memories ADD COLUMN source_status TEXT NOT NULL DEFAULT 'active';
        ALTER TABLE memories ADD COLUMN embedding_json TEXT NOT NULL DEFAULT '[]';
        ALTER TABLE memories ADD COLUMN embedding_model TEXT NOT NULL DEFAULT '';
        CREATE INDEX IF NOT EXISTS idx_memories_scope
            ON memories(owner_id, scope, kb_id, enabled, source_status, expires_at, updated_at);
        """,
    ),
    (
        "app-005",
        """
        CREATE TABLE IF NOT EXISTS voice_practice_sessions (
            id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            kb_id TEXT NOT NULL,
            goal TEXT NOT NULL DEFAULT 'free',
            status TEXT NOT NULL DEFAULT 'active',
            conversation_id TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            ended_at TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_voice_sessions_scope
            ON voice_practice_sessions(owner_id, kb_id, status, updated_at);
        CREATE TABLE IF NOT EXISTS voice_practice_turns (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            kb_id TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            transcript TEXT NOT NULL DEFAULT '',
            answer TEXT NOT NULL DEFAULT '',
            citations_json TEXT NOT NULL DEFAULT '[]',
            memory_refs_json TEXT NOT NULL DEFAULT '[]',
            audio_status TEXT NOT NULL DEFAULT 'not_requested',
            status TEXT NOT NULL DEFAULT 'completed',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_voice_turns_scope
            ON voice_practice_turns(owner_id, kb_id, session_id, created_at);
        """,
    ),
)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _memory_terms(text: str) -> list[str]:
    raw = str(text or "").lower()
    terms = re.findall(r"[\u4e00-\u9fff]|[a-z0-9_]+", raw)
    return list(dict.fromkeys(term for term in terms if term.strip()))


def _keyword_overlap(query_terms: list[str], memory_terms: list[str]) -> float:
    if not query_terms or not memory_terms:
        return 0.0
    overlap = len(set(query_terms) & set(memory_terms))
    return min(1.0, overlap / max(1, min(len(query_terms), 6)))


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return max(0.0, dot / (left_norm * right_norm))


def _memory_token_count(text: str) -> int:
    return max(1, len(_memory_terms(text))) if str(text or "").strip() else 0


def app_db_path(working_dir: Path) -> Path:
    return Path(working_dir) / APP_DB_NAME


def _commit_with_retry(conn: sqlite3.Connection, attempts: int = 6) -> None:
    for attempt in range(attempts):
        try:
            conn.commit()
            return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or attempt == attempts - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def _pragma_with_retry(conn: sqlite3.Connection, statement: str, attempts: int = 20) -> None:
    for attempt in range(attempts):
        try:
            conn.execute(statement)
            return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or attempt == attempts - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def backup_appdb(working_dir: Path, destination: Path) -> Path:
    """Create a consistent SQLite backup, including WAL contents."""
    source = open_appdb(working_dir)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source._lock:
        target = sqlite3.connect(str(destination))
        try:
            source._conn.backup(target)
            target.commit()
        finally:
            target.close()
    return destination


def restore_appdb(working_dir: Path, backup_path: Path) -> Path:
    """Replace the application database from a validated SQLite backup."""
    backup_path = Path(backup_path)
    check = sqlite3.connect(str(backup_path))
    try:
        result = check.execute("PRAGMA integrity_check").fetchone()[0]
        if result != "ok":
            raise sqlite3.DatabaseError(f"备份完整性检查失败: {result}")
    finally:
        check.close()
    target_path = app_db_path(working_dir)
    reset_appdb_cache()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target = sqlite3.connect(str(target_path))
    source = sqlite3.connect(str(backup_path))
    try:
        source.backup(target)
        target.commit()
    finally:
        source.close()
        target.close()
    reset_appdb_cache()
    return target_path


def appdb_integrity_check(working_dir: Path) -> str:
    store = open_appdb(working_dir)
    with store._lock:
        return str(store._conn.execute("PRAGMA integrity_check").fetchone()[0])


def open_appdb(working_dir: Path) -> "AppStore":
    key = str(app_db_path(working_dir).resolve())
    with _GUARD:
        store = _STORES.get(key)
        if store is None:
            store = AppStore(Path(key))
            _STORES[key] = store
        return store


def reset_appdb_cache() -> None:
    with _GUARD:
        for store in _STORES.values():
            store.close()
        _STORES.clear()


class AppStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=8.0)
        self._conn.row_factory = sqlite3.Row
        _pragma_with_retry(self._conn, "PRAGMA journal_mode = WAL")
        _pragma_with_retry(self._conn, "PRAGMA synchronous = NORMAL")
        _pragma_with_retry(self._conn, "PRAGMA busy_timeout = 8000")
        _pragma_with_retry(self._conn, "PRAGMA foreign_keys = ON")
        self.migrate()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def migrate(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )
            applied = {
                str(row[0])
                for row in self._conn.execute("SELECT version FROM schema_migrations")
            }
            for version, sql in _MIGRATIONS:
                if version in applied:
                    continue
                self._conn.executescript(sql)
                self._conn.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (version, _now()),
                )
            _commit_with_retry(self._conn)

    def schema_versions(self) -> list[str]:
        rows = self._all("SELECT version FROM schema_migrations ORDER BY applied_at")
        return [str(row["version"]) for row in rows]

    def create_conversation(self, owner_id: str, kb_id: str, title: str = "") -> dict[str, Any]:
        now = _now()
        row = {
            "id": uuid4().hex,
            "owner_id": owner_id,
            "kb_id": kb_id,
            "title": (title or "新会话")[:80],
            "summary": "",
            "is_pinned": 0,
            "is_archived": 0,
            "created_at": now,
            "updated_at": now,
            "last_message_at": now,
        }
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO conversations(
                    id, owner_id, kb_id, title, summary, is_pinned, is_archived,
                    created_at, updated_at, last_message_at
                ) VALUES (
                    :id, :owner_id, :kb_id, :title, :summary, :is_pinned, :is_archived,
                    :created_at, :updated_at, :last_message_at
                )
                """,
                row,
            )
            _commit_with_retry(self._conn)
        return self._public_conversation(row)

    def get_conversation(self, conversation_id: str, owner_id: str, kb_id: str) -> dict[str, Any] | None:
        row = self._one(
            "SELECT * FROM conversations WHERE id = ? AND owner_id = ? AND kb_id = ?",
            (conversation_id, owner_id, kb_id),
        )
        return self._public_conversation(row) if row else None

    def list_conversations(
        self,
        owner_id: str,
        kb_id: str,
        *,
        q: str = "",
        archived: bool = False,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT * FROM conversations
            WHERE owner_id = ? AND kb_id = ? AND is_archived = ?
        """
        params: list[Any] = [owner_id, kb_id, 1 if archived else 0]
        text = (q or "").strip()
        if text:
            query += " AND title LIKE ?"
            params.append(f"%{text}%")
        query += " ORDER BY is_pinned DESC, last_message_at DESC, updated_at DESC"
        return [self._public_conversation(row) for row in self._all(query, tuple(params))]

    def update_conversation(
        self,
        conversation_id: str,
        owner_id: str,
        kb_id: str,
        *,
        title: str | None = None,
        summary: str | None = None,
        is_pinned: bool | None = None,
        is_archived: bool | None = None,
    ) -> dict[str, Any] | None:
        current = self.get_conversation(conversation_id, owner_id, kb_id)
        if current is None:
            return None
        if title is not None:
            current["title"] = title.strip()[:80] or current["title"]
        if summary is not None:
            current["summary"] = summary[:1200]
        if is_pinned is not None:
            current["is_pinned"] = bool(is_pinned)
        if is_archived is not None:
            current["is_archived"] = bool(is_archived)
        current["updated_at"] = _now()
        with self._lock:
            self._conn.execute(
                """
                UPDATE conversations
                SET title = ?, summary = ?, is_pinned = ?, is_archived = ?, updated_at = ?
                WHERE id = ? AND owner_id = ? AND kb_id = ?
                """,
                (
                    current["title"],
                    current["summary"],
                    1 if current["is_pinned"] else 0,
                    1 if current["is_archived"] else 0,
                    current["updated_at"],
                    conversation_id,
                    owner_id,
                    kb_id,
                ),
            )
            _commit_with_retry(self._conn)
        return self.get_conversation(conversation_id, owner_id, kb_id)

    def delete_conversation(self, conversation_id: str, owner_id: str, kb_id: str) -> bool:
        found = self.get_conversation(conversation_id, owner_id, kb_id)
        if found is None:
            return False
        with self._lock:
            self._conn.execute(
                "DELETE FROM memory_candidates WHERE conversation_id = ? AND owner_id = ? AND kb_id = ?",
                (conversation_id, owner_id, kb_id),
            )
            self._conn.execute(
                "DELETE FROM messages WHERE conversation_id = ? AND owner_id = ? AND kb_id = ?",
                (conversation_id, owner_id, kb_id),
            )
            self._conn.execute(
                "DELETE FROM conversations WHERE id = ? AND owner_id = ? AND kb_id = ?",
                (conversation_id, owner_id, kb_id),
            )
            _commit_with_retry(self._conn)
        return True

    def add_message(
        self,
        *,
        conversation_id: str,
        owner_id: str,
        kb_id: str,
        role: str,
        content: str,
        status: str,
        citations: list[dict[str, Any]] | None = None,
        retrieval_meta: dict[str, Any] | None = None,
        index_version: str = "",
        embedding_model: str = "",
        rerank_model: str = "",
        llm_model: str = "",
    ) -> dict[str, Any]:
        now = _now()
        row = {
            "id": uuid4().hex,
            "conversation_id": conversation_id,
            "owner_id": owner_id,
            "kb_id": kb_id,
            "role": role,
            "content": content,
            "status": status,
            "citations_json": json.dumps(citations or [], ensure_ascii=False),
            "retrieval_meta_json": json.dumps(retrieval_meta or {}, ensure_ascii=False),
            "index_version": index_version,
            "embedding_model": embedding_model,
            "rerank_model": rerank_model,
            "llm_model": llm_model,
            "created_at": now,
        }
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO messages(
                    id, conversation_id, owner_id, kb_id, role, content, status,
                    citations_json, retrieval_meta_json, index_version,
                    embedding_model, rerank_model, llm_model, created_at
                ) VALUES (
                    :id, :conversation_id, :owner_id, :kb_id, :role, :content, :status,
                    :citations_json, :retrieval_meta_json, :index_version,
                    :embedding_model, :rerank_model, :llm_model, :created_at
                )
                """,
                row,
            )
            self._conn.execute(
                """
                UPDATE conversations
                SET updated_at = ?, last_message_at = ?
                WHERE id = ? AND owner_id = ? AND kb_id = ?
                """,
                (now, now, conversation_id, owner_id, kb_id),
            )
            _commit_with_retry(self._conn)
        return self._public_message(row)

    def list_messages(self, conversation_id: str, owner_id: str, kb_id: str) -> list[dict[str, Any]]:
        rows = self._all(
            """
            SELECT * FROM messages
            WHERE conversation_id = ? AND owner_id = ? AND kb_id = ?
            ORDER BY created_at ASC, rowid ASC
            """,
            (conversation_id, owner_id, kb_id),
        )
        return [self._public_message(row) for row in rows]

    def get_message(self, message_id: str, owner_id: str, kb_id: str) -> dict[str, Any] | None:
        row = self._one(
            "SELECT * FROM messages WHERE id = ? AND owner_id = ? AND kb_id = ?",
            (message_id, owner_id, kb_id),
        )
        return self._public_message(row) if row else None

    def add_feedback(
        self,
        *,
        message_id: str,
        owner_id: str,
        kb_id: str,
        rating: str,
        reason: str = "",
        comment: str = "",
        question: str = "",
        answer: str = "",
        citations: list[dict[str, Any]] | None = None,
        index_version: str = "",
        embedding_model: str = "",
        rerank_model: str = "",
        llm_model: str = "",
    ) -> dict[str, Any]:
        now = _now()
        payload = {
            "id": uuid4().hex,
            "message_id": message_id,
            "owner_id": owner_id,
            "kb_id": kb_id,
            "rating": rating,
            "reason": reason if rating == "negative" else "",
            "comment": comment,
            "question": question,
            "answer": answer,
            "citations_json": json.dumps(citations or [], ensure_ascii=False),
            "index_version": index_version,
            "embedding_model": embedding_model,
            "rerank_model": rerank_model,
            "llm_model": llm_model,
            "created_at": now,
        }
        with self._lock:
            self._conn.execute("DELETE FROM answer_feedback WHERE message_id = ? AND owner_id = ?", (message_id, owner_id))
            self._conn.execute(
                """
                INSERT INTO answer_feedback(
                    id, message_id, owner_id, kb_id, rating, reason, comment,
                    question, answer, citations_json, index_version,
                    embedding_model, rerank_model, llm_model, created_at
                ) VALUES (
                    :id, :message_id, :owner_id, :kb_id, :rating, :reason, :comment,
                    :question, :answer, :citations_json, :index_version,
                    :embedding_model, :rerank_model, :llm_model, :created_at
                )
                """,
                payload,
            )
            if rating == "negative":
                self._conn.execute(
                    """
                    INSERT INTO offline_eval_candidates(id, feedback_id, owner_id, kb_id, payload_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid4().hex,
                        payload["id"],
                        owner_id,
                        kb_id,
                        json.dumps({
                            "question": question,
                            "answer": answer,
                            "citations": citations or [],
                            "reason": reason,
                            "comment": comment,
                            "index_version": index_version,
                            "embedding_model": embedding_model,
                            "rerank_model": rerank_model,
                            "llm_model": llm_model,
                        }, ensure_ascii=False),
                        now,
                    ),
                )
            _commit_with_retry(self._conn)
        return self._public_feedback(payload)

    def list_offline_eval(self, owner_id: str, kb_id: str) -> list[dict[str, Any]]:
        rows = self._all(
            "SELECT * FROM offline_eval_candidates WHERE owner_id = ? AND kb_id = ? ORDER BY created_at DESC",
            (owner_id, kb_id),
        )
        items = []
        for row in rows:
            data = dict(row)
            try:
                payload = json.loads(data.get("payload_json") or "{}")
            except json.JSONDecodeError:
                payload = {}
            items.append({
                "id": data["id"],
                "feedback_id": data["feedback_id"],
                "owner_id": data["owner_id"],
                "kb_id": data["kb_id"],
                "payload": payload,
                "created_at": data["created_at"],
            })
        return items

    def history_for_answer(self, conversation_id: str, owner_id: str, kb_id: str) -> list[dict[str, str]]:
        items = []
        for row in self.list_messages(conversation_id, owner_id, kb_id):
            if row["status"] != "completed" or row["role"] not in {"user", "assistant"}:
                continue
            if not row["content"]:
                continue
            items.append({"role": row["role"], "content": row["content"]})
        return items

    def add_memory_candidate(
        self,
        *,
        owner_id: str,
        kb_id: str,
        content: str,
        category: str = "other",
        conversation_id: str = "",
        message_id: str = "",
        source_refs: list[dict[str, Any]] | None = None,
        status: str = "pending",
    ) -> dict[str, Any]:
        now = _now()
        row = {
            "id": uuid4().hex,
            "owner_id": owner_id,
            "kb_id": kb_id,
            "conversation_id": conversation_id,
            "message_id": message_id,
            "content": (content or "").strip(),
            "category": category or "other",
            "source_refs_json": json.dumps(source_refs or [], ensure_ascii=False),
            "status": status if status in {"pending", "accepted", "rejected"} else "pending",
            "created_at": now,
        }
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO memory_candidates(
                    id, owner_id, kb_id, conversation_id, message_id, content,
                    category, source_refs_json, status, created_at
                ) VALUES (
                    :id, :owner_id, :kb_id, :conversation_id, :message_id, :content,
                    :category, :source_refs_json, :status, :created_at
                )
                """,
                row,
            )
            _commit_with_retry(self._conn)
        return self._public_candidate(row)

    def get_memory_candidate(self, candidate_id: str, owner_id: str, kb_id: str | None = None) -> dict[str, Any] | None:
        if kb_id:
            row = self._one(
                "SELECT * FROM memory_candidates WHERE id = ? AND owner_id = ? AND kb_id = ?",
                (candidate_id, owner_id, kb_id),
            )
        else:
            row = self._one(
                "SELECT * FROM memory_candidates WHERE id = ? AND owner_id = ?",
                (candidate_id, owner_id),
            )
        return self._public_candidate(row) if row else None

    def list_memory_candidates(self, owner_id: str, kb_id: str | None = None, status: str = "pending") -> list[dict[str, Any]]:
        if kb_id:
            rows = self._all(
                """
                SELECT * FROM memory_candidates
                WHERE owner_id = ? AND kb_id = ? AND status = ?
                ORDER BY created_at DESC
                """,
                (owner_id, kb_id, status),
            )
        else:
            rows = self._all(
                """
                SELECT * FROM memory_candidates
                WHERE owner_id = ? AND status = ?
                ORDER BY created_at DESC
                """,
                (owner_id, status),
            )
        return [self._public_candidate(row) for row in rows]

    def set_candidate_status(self, candidate_id: str, owner_id: str, status: str, kb_id: str | None = None) -> dict[str, Any] | None:
        current = self.get_memory_candidate(candidate_id, owner_id, kb_id)
        if current is None:
            return None
        with self._lock:
            self._conn.execute(
                "UPDATE memory_candidates SET status = ? WHERE id = ? AND owner_id = ?",
                (status, candidate_id, owner_id),
            )
            _commit_with_retry(self._conn)
        return self.get_memory_candidate(candidate_id, owner_id, kb_id)

    def accept_memory_candidate(
        self,
        candidate_id: str,
        owner_id: str,
        kb_id: str | None = None,
        *,
        embedding: list[float] | None = None,
        embedding_model: str = "",
    ) -> dict[str, Any] | None:
        current = self.get_memory_candidate(candidate_id, owner_id, kb_id)
        if current is None or current["status"] != "pending":
            return None
        memory = self.add_memory(
            owner_id=owner_id,
            content=current["content"],
            category=current["category"],
            source_type="candidate",
            source_id=current["id"],
            scope="kb",
            kb_id=kb_id or current.get("kb_id") or "",
            embedding=embedding,
            embedding_model=embedding_model,
        )
        self.set_candidate_status(candidate_id, owner_id, "accepted", kb_id)
        return memory

    def add_memory(
        self,
        *,
        owner_id: str,
        content: str,
        category: str = "other",
        source_type: str = "candidate",
        source_id: str = "",
        enabled: bool = True,
        scope: str = "global",
        kb_id: str = "",
        expires_at: str = "",
        source_status: str = "active",
        embedding: list[float] | None = None,
        embedding_model: str = "",
    ) -> dict[str, Any]:
        now = _now()
        scope = scope if scope in {"global", "kb"} else "global"
        kb_id = str(kb_id or "") if scope == "kb" else ""
        row = {
            "id": uuid4().hex,
            "owner_id": owner_id,
            "content": (content or "").strip(),
            "category": category or "other",
            "source_type": source_type,
            "source_id": source_id,
            "enabled": 1 if enabled else 0,
            "scope": scope,
            "kb_id": kb_id,
            "expires_at": str(expires_at or ""),
            "source_status": source_status if source_status in {"active", "deleted", "unavailable", "expired"} else "active",
            "embedding_json": json.dumps(embedding or [], ensure_ascii=False),
            "embedding_model": embedding_model or "",
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO memories(
                    id, owner_id, content, category, source_type, source_id, enabled,
                    scope, kb_id, expires_at, source_status, embedding_json, embedding_model,
                    created_at, updated_at
                ) VALUES (
                    :id, :owner_id, :content, :category, :source_type, :source_id, :enabled,
                    :scope, :kb_id, :expires_at, :source_status, :embedding_json, :embedding_model,
                    :created_at, :updated_at
                )
                """,
                row,
            )
            _commit_with_retry(self._conn)
        return self._public_memory(row)

    def get_memory(self, memory_id: str, owner_id: str) -> dict[str, Any] | None:
        row = self._one("SELECT * FROM memories WHERE id = ? AND owner_id = ?", (memory_id, owner_id))
        return self._public_memory(row) if row else None

    def list_memories(self, owner_id: str, kb_id: str | None = None) -> list[dict[str, Any]]:
        if kb_id:
            rows = self._all(
                """SELECT * FROM memories
                   WHERE owner_id = ? AND (scope = 'global' OR (scope = 'kb' AND kb_id = ?))
                   ORDER BY updated_at DESC""",
                (owner_id, kb_id),
            )
        else:
            rows = self._all("SELECT * FROM memories WHERE owner_id = ? ORDER BY updated_at DESC", (owner_id,))
        return [self._public_memory(row) for row in rows]

    def enabled_memories(self, owner_id: str) -> list[dict[str, Any]]:
        rows = self._all(
            "SELECT * FROM memories WHERE owner_id = ? AND enabled = 1 ORDER BY updated_at DESC",
            (owner_id,),
        )
        return [self._public_memory(row) for row in rows]

    def retrieve_memories(
        self,
        owner_id: str,
        kb_id: str,
        query: str,
        top_k: int = 5,
        token_budget: int = 1200,
        *,
        embedding: list[float] | None = None,
        embedding_model: str = "",
    ) -> list[dict[str, Any]]:
        """召回当前用户的全局记忆和当前知识库记忆，不跨库泄漏。"""
        now = _now()
        rows = self._all(
            """SELECT * FROM memories
               WHERE owner_id = ? AND enabled = 1 AND source_status = 'active'
                 AND (scope = 'global' OR (scope = 'kb' AND kb_id = ?))
                 AND (expires_at = '' OR expires_at > ?)
               ORDER BY updated_at DESC""",
            (owner_id, kb_id, now),
        )
        query_terms = _memory_terms(query)
        candidates: list[dict[str, Any]] = []
        for row in rows:
            item = self._public_memory(row)
            terms = _memory_terms(item["content"])
            keyword_score = _keyword_overlap(query_terms, terms)
            vector_score = 0.0
            stored_model = str(item.get("embedding_model") or "")
            if embedding and (not embedding_model or not stored_model or stored_model == embedding_model):
                vector_score = _cosine(embedding, item.get("embedding") or [])
            score = (0.7 * vector_score) + (0.3 * keyword_score)
            if not embedding:
                score = keyword_score
            # Memory is a secondary signal; require a minimum score so an unrelated
            # global preference cannot enter every answer merely because it is enabled.
            if score < 0.2:
                continue
            item["memory_score"] = round(float(score), 6)
            item["retrieval_reason"] = "语义与关键词均相关" if vector_score and keyword_score else ("语义相关" if vector_score else "关键词命中")
            candidates.append(item)
        candidates.sort(key=lambda item: (float(item.get("memory_score") or 0), item.get("updated_at") or ""), reverse=True)
        selected: list[dict[str, Any]] = []
        used_tokens = 0
        for item in candidates[: max(1, min(int(top_k or 5), 10))]:
            tokens = _memory_token_count(item["content"])
            if selected and used_tokens + tokens > max(1, int(token_budget or 1200)):
                continue
            if not selected and tokens > max(1, int(token_budget or 1200)):
                item["content"] = item["content"][: max(1, int(token_budget or 1200)) * 2]
                tokens = _memory_token_count(item["content"])
            selected.append(item)
            used_tokens += tokens
        return selected

    def update_memory(
        self,
        memory_id: str,
        owner_id: str,
        *,
        content: str | None = None,
        category: str | None = None,
        enabled: bool | None = None,
        scope: str | None = None,
        kb_id: str | None = None,
        expires_at: str | None = None,
    ) -> dict[str, Any] | None:
        current = self.get_memory(memory_id, owner_id)
        if current is None:
            return None
        next_content = current["content"] if content is None else content.strip()
        next_category = current["category"] if category is None else category
        next_enabled = current["enabled"] if enabled is None else enabled
        next_scope = current.get("scope") if scope is None else (scope if scope in {"global", "kb"} else current.get("scope"))
        next_kb_id = current.get("kb_id") if kb_id is None else str(kb_id or "")
        if next_scope == "global":
            next_kb_id = ""
        next_expires_at = current.get("expires_at") if expires_at is None else str(expires_at or "")
        with self._lock:
            self._conn.execute(
                """
                UPDATE memories
                SET content = ?, category = ?, enabled = ?, scope = ?, kb_id = ?, expires_at = ?, updated_at = ?
                WHERE id = ? AND owner_id = ?
                """,
                (next_content, next_category, 1 if next_enabled else 0, next_scope, next_kb_id, next_expires_at, _now(), memory_id, owner_id),
            )
            _commit_with_retry(self._conn)
        return self.get_memory(memory_id, owner_id)

    def delete_memory(self, memory_id: str, owner_id: str) -> bool:
        found = self.get_memory(memory_id, owner_id)
        if found is None:
            return False
        with self._lock:
            self._conn.execute("DELETE FROM memories WHERE id = ? AND owner_id = ?", (memory_id, owner_id))
            _commit_with_retry(self._conn)
        return True

    def count_documents_untouched(self) -> None:
        return None

    def purge_kb(self, kb_id: str, owner_id: str | None = None) -> None:
        """删除知识库时清理会话、消息、反馈和候选，并停用该库的记忆。

        global 记忆属于用户，不随知识库删除；kb 记忆标记为 deleted，避免成为孤儿上下文。
        """
        with self._lock:
            if owner_id:
                self._conn.execute("DELETE FROM offline_eval_candidates WHERE kb_id = ? AND owner_id = ?", (kb_id, owner_id))
                self._conn.execute("DELETE FROM answer_feedback WHERE kb_id = ? AND owner_id = ?", (kb_id, owner_id))
                self._conn.execute("DELETE FROM memory_candidates WHERE kb_id = ? AND owner_id = ?", (kb_id, owner_id))
                self._conn.execute("DELETE FROM messages WHERE kb_id = ? AND owner_id = ?", (kb_id, owner_id))
                self._conn.execute("DELETE FROM conversations WHERE kb_id = ? AND owner_id = ?", (kb_id, owner_id))
                self._conn.execute("DELETE FROM voice_practice_turns WHERE kb_id = ? AND owner_id = ?", (kb_id, owner_id))
                self._conn.execute("DELETE FROM voice_practice_sessions WHERE kb_id = ? AND owner_id = ?", (kb_id, owner_id))
                self._conn.execute(
                    "UPDATE memories SET source_status = 'deleted', enabled = 0, updated_at = ? WHERE kb_id = ? AND owner_id = ? AND scope = 'kb'",
                    (_now(), kb_id, owner_id),
                )
            else:
                self._conn.execute("DELETE FROM offline_eval_candidates WHERE kb_id = ?", (kb_id,))
                self._conn.execute("DELETE FROM answer_feedback WHERE kb_id = ?", (kb_id,))
                self._conn.execute("DELETE FROM memory_candidates WHERE kb_id = ?", (kb_id,))
                self._conn.execute("DELETE FROM messages WHERE kb_id = ?", (kb_id,))
                self._conn.execute("DELETE FROM conversations WHERE kb_id = ?", (kb_id,))
                self._conn.execute("DELETE FROM voice_practice_turns WHERE kb_id = ?", (kb_id,))
                self._conn.execute("DELETE FROM voice_practice_sessions WHERE kb_id = ?", (kb_id,))
                self._conn.execute(
                    "UPDATE memories SET source_status = 'deleted', enabled = 0, updated_at = ? WHERE kb_id = ? AND scope = 'kb'",
                    (_now(), kb_id),
                )
            _commit_with_retry(self._conn)

    def create_voice_practice_session(
        self,
        owner_id: str,
        kb_id: str,
        *,
        goal: str = "free",
        conversation_id: str = "",
    ) -> dict[str, Any]:
        now = _now()
        row = {
            "id": uuid4().hex,
            "owner_id": owner_id,
            "kb_id": kb_id,
            "goal": goal if goal in {"free", "recall", "interview", "review"} else "free",
            "status": "active",
            "conversation_id": conversation_id or "",
            "summary": "",
            "created_at": now,
            "updated_at": now,
            "ended_at": "",
        }
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO voice_practice_sessions(
                    id, owner_id, kb_id, goal, status, conversation_id, summary,
                    created_at, updated_at, ended_at
                ) VALUES (:id, :owner_id, :kb_id, :goal, :status, :conversation_id,
                          :summary, :created_at, :updated_at, :ended_at)
                """,
                row,
            )
            _commit_with_retry(self._conn)
        return self._public_voice_session(row)

    def get_voice_practice_session(self, session_id: str, owner_id: str, kb_id: str | None = None) -> dict[str, Any] | None:
        sql = "SELECT * FROM voice_practice_sessions WHERE id = ? AND owner_id = ?"
        params: list[Any] = [session_id, owner_id]
        if kb_id is not None:
            sql += " AND kb_id = ?"
            params.append(kb_id)
        row = self._one(sql, tuple(params))
        return self._public_voice_session(row) if row else None

    def list_voice_practice_sessions(self, owner_id: str, kb_id: str, limit: int = 30) -> list[dict[str, Any]]:
        rows = self._all(
            """SELECT
                   sessions.*,
                   COUNT(turns.id) AS turn_count,
                   COALESCE(SUM(CASE WHEN turns.status = 'completed' THEN 1 ELSE 0 END), 0)
                       AS completed_turn_count,
                   COALESCE(SUM(CASE
                       WHEN turns.citations_json IS NOT NULL
                            AND turns.citations_json NOT IN ('', '[]')
                       THEN 1 ELSE 0 END), 0) AS cited_turn_count,
                   COALESCE((
                       SELECT first_turn.transcript
                       FROM voice_practice_turns AS first_turn
                       WHERE first_turn.session_id = sessions.id
                         AND first_turn.owner_id = sessions.owner_id
                         AND first_turn.kb_id = sessions.kb_id
                       ORDER BY first_turn.created_at ASC
                       LIMIT 1
                   ), '') AS first_transcript
               FROM voice_practice_sessions AS sessions
               LEFT JOIN voice_practice_turns AS turns
                 ON turns.session_id = sessions.id
                AND turns.owner_id = sessions.owner_id
                AND turns.kb_id = sessions.kb_id
               WHERE sessions.owner_id = ? AND sessions.kb_id = ?
               GROUP BY sessions.id
               ORDER BY sessions.updated_at DESC LIMIT ?""",
            (owner_id, kb_id, max(1, min(int(limit or 30), 100))),
        )
        return [self._public_voice_session(row) for row in rows]

    def update_voice_practice_session(
        self,
        session_id: str,
        owner_id: str,
        kb_id: str,
        *,
        status: str | None = None,
        summary: str | None = None,
        conversation_id: str | None = None,
    ) -> dict[str, Any] | None:
        current = self.get_voice_practice_session(session_id, owner_id, kb_id)
        if current is None:
            return None
        next_status = status if status in {"active", "completed", "cancelled", "failed"} else current["status"]
        next_summary = current["summary"] if summary is None else str(summary or "")[:2000]
        next_conversation = current["conversation_id"] if conversation_id is None else str(conversation_id or "")
        ended_at = current.get("ended_at") or ""
        if next_status in {"completed", "cancelled", "failed"} and not ended_at:
            ended_at = _now()
        with self._lock:
            self._conn.execute(
                """UPDATE voice_practice_sessions
                   SET status = ?, summary = ?, conversation_id = ?, updated_at = ?, ended_at = ?
                   WHERE id = ? AND owner_id = ? AND kb_id = ?""",
                (next_status, next_summary, next_conversation, _now(), ended_at, session_id, owner_id, kb_id),
            )
            _commit_with_retry(self._conn)
        return self.get_voice_practice_session(session_id, owner_id, kb_id)

    def add_voice_practice_turn(
        self,
        *,
        session_id: str,
        owner_id: str,
        kb_id: str,
        transcript: str,
        answer: str = "",
        citations: list[dict[str, Any]] | None = None,
        memory_refs: list[dict[str, Any]] | None = None,
        audio_status: str = "not_requested",
        status: str = "completed",
    ) -> dict[str, Any]:
        now = _now()
        row = {
            "id": uuid4().hex,
            "session_id": session_id,
            "owner_id": owner_id,
            "kb_id": kb_id,
            "role": "turn",
            "transcript": str(transcript or "")[:8000],
            "answer": str(answer or "")[:16000],
            "citations_json": json.dumps(citations or [], ensure_ascii=False),
            "memory_refs_json": json.dumps(memory_refs or [], ensure_ascii=False),
            "audio_status": audio_status or "not_requested",
            "status": status or "completed",
            "created_at": now,
        }
        with self._lock:
            self._conn.execute(
                """INSERT INTO voice_practice_turns(
                    id, session_id, owner_id, kb_id, role, transcript, answer,
                    citations_json, memory_refs_json, audio_status, status, created_at
                ) VALUES (:id, :session_id, :owner_id, :kb_id, :role, :transcript, :answer,
                          :citations_json, :memory_refs_json, :audio_status, :status, :created_at)""",
                row,
            )
            _commit_with_retry(self._conn)
        return self._public_voice_turn(row)

    def list_voice_practice_turns(self, session_id: str, owner_id: str, kb_id: str) -> list[dict[str, Any]]:
        rows = self._all(
            """SELECT * FROM voice_practice_turns
               WHERE session_id = ? AND owner_id = ? AND kb_id = ?
               ORDER BY created_at ASC""",
            (session_id, owner_id, kb_id),
        )
        return [self._public_voice_turn(row) for row in rows]

    def get_voice_practice_turn(self, turn_id: str, owner_id: str) -> dict[str, Any] | None:
        row = self._one(
            "SELECT * FROM voice_practice_turns WHERE id = ? AND owner_id = ?",
            (turn_id, owner_id),
        )
        return self._public_voice_turn(row) if row else None

    def update_voice_practice_turn_audio(self, turn_id: str, owner_id: str, audio_status: str) -> dict[str, Any] | None:
        status = audio_status if audio_status in {"not_requested", "generating", "ready", "failed"} else "failed"
        with self._lock:
            self._conn.execute(
                "UPDATE voice_practice_turns SET audio_status = ? WHERE id = ? AND owner_id = ?",
                (status, turn_id, owner_id),
            )
            _commit_with_retry(self._conn)
        return self.get_voice_practice_turn(turn_id, owner_id)

    def _one(self, sql: str, params: tuple[Any, ...]) -> sqlite3.Row | None:
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        return row

    def _all(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return list(rows)

    def _public_conversation(self, row: Any) -> dict[str, Any]:
        data = dict(row)
        return {
            "id": data["id"],
            "owner_id": data["owner_id"],
            "kb_id": data["kb_id"],
            "title": data["title"],
            "summary": data.get("summary") or "",
            "is_pinned": bool(data.get("is_pinned")),
            "is_archived": bool(data.get("is_archived")),
            "created_at": data["created_at"],
            "updated_at": data["updated_at"],
            "last_message_at": data.get("last_message_at") or "",
        }

    def _public_message(self, row: Any) -> dict[str, Any]:
        data = dict(row)
        try:
            citations = json.loads(data.get("citations_json") or "[]")
        except json.JSONDecodeError:
            citations = []
        try:
            meta = json.loads(data.get("retrieval_meta_json") or "{}")
        except json.JSONDecodeError:
            meta = {}
        return {
            "id": data["id"],
            "conversation_id": data["conversation_id"],
            "owner_id": data["owner_id"],
            "kb_id": data["kb_id"],
            "role": data["role"],
            "content": data["content"],
            "status": data["status"],
            "citations": citations if isinstance(citations, list) else [],
            "retrieval_meta": meta if isinstance(meta, dict) else {},
            "index_version": data.get("index_version") or "",
            "embedding_model": data.get("embedding_model") or "",
            "rerank_model": data.get("rerank_model") or "",
            "llm_model": data.get("llm_model") or "",
            "created_at": data["created_at"],
        }

    def _public_feedback(self, row: Any) -> dict[str, Any]:
        data = dict(row)
        try:
            citations = json.loads(data.get("citations_json") or "[]")
        except json.JSONDecodeError:
            citations = []
        return {
            "id": data["id"],
            "message_id": data["message_id"],
            "owner_id": data["owner_id"],
            "kb_id": data["kb_id"],
            "rating": data["rating"],
            "reason": data.get("reason") or "",
            "comment": data.get("comment") or "",
            "question": data.get("question") or "",
            "answer": data.get("answer") or "",
            "citations": citations if isinstance(citations, list) else [],
            "index_version": data.get("index_version") or "",
            "embedding_model": data.get("embedding_model") or "",
            "rerank_model": data.get("rerank_model") or "",
            "llm_model": data.get("llm_model") or "",
            "created_at": data["created_at"],
        }

    def _public_candidate(self, row: Any) -> dict[str, Any]:
        data = dict(row)
        try:
            refs = json.loads(data.get("source_refs_json") or "[]")
        except json.JSONDecodeError:
            refs = []
        return {
            "id": data["id"],
            "owner_id": data["owner_id"],
            "kb_id": data["kb_id"],
            "conversation_id": data.get("conversation_id") or "",
            "message_id": data.get("message_id") or "",
            "content": data.get("content") or "",
            "category": data.get("category") or "other",
            "source_refs": refs if isinstance(refs, list) else [],
            "status": data.get("status") or "pending",
            "created_at": data["created_at"],
        }

    def _public_memory(self, row: Any) -> dict[str, Any]:
        data = dict(row)
        try:
            embedding = json.loads(data.get("embedding_json") or "[]")
        except json.JSONDecodeError:
            embedding = []
        return {
            "id": data["id"],
            "owner_id": data["owner_id"],
            "content": data.get("content") or "",
            "category": data.get("category") or "other",
            "source_type": data.get("source_type") or "candidate",
            "source_id": data.get("source_id") or "",
            "enabled": bool(data.get("enabled")),
            "scope": data.get("scope") or "global",
            "kb_id": data.get("kb_id") or "",
            "expires_at": data.get("expires_at") or "",
            "source_status": data.get("source_status") or "active",
            "embedding": embedding if isinstance(embedding, list) else [],
            "embedding_model": data.get("embedding_model") or "",
            "created_at": data["created_at"],
            "updated_at": data["updated_at"],
        }

    def _public_voice_session(self, row: Any) -> dict[str, Any]:
        data = dict(row)
        session = {
            "id": data["id"],
            "owner_id": data["owner_id"],
            "kb_id": data["kb_id"],
            "goal": data.get("goal") or "free",
            "status": data.get("status") or "active",
            "conversation_id": data.get("conversation_id") or "",
            "summary": data.get("summary") or "",
            "created_at": data.get("created_at") or "",
            "updated_at": data.get("updated_at") or "",
            "ended_at": data.get("ended_at") or "",
        }
        if "turn_count" in data:
            session.update({
                "turn_count": int(data.get("turn_count") or 0),
                "completed_turn_count": int(data.get("completed_turn_count") or 0),
                "cited_turn_count": int(data.get("cited_turn_count") or 0),
                "first_transcript": str(data.get("first_transcript") or ""),
            })
        return session

    def _public_voice_turn(self, row: Any) -> dict[str, Any]:
        data = dict(row)
        try:
            citations = json.loads(data.get("citations_json") or "[]")
        except json.JSONDecodeError:
            citations = []
        try:
            memory_refs = json.loads(data.get("memory_refs_json") or "[]")
        except json.JSONDecodeError:
            memory_refs = []
        return {
            "id": data["id"],
            "session_id": data["session_id"],
            "owner_id": data["owner_id"],
            "kb_id": data["kb_id"],
            "role": data.get("role") or "turn",
            "transcript": data.get("transcript") or "",
            "answer": data.get("answer") or "",
            "citations": citations if isinstance(citations, list) else [],
            "memory_refs": memory_refs if isinstance(memory_refs, list) else [],
            "audio_status": data.get("audio_status") or "not_requested",
            "status": data.get("status") or "completed",
            "created_at": data.get("created_at") or "",
        }
