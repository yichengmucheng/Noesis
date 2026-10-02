"""个人知识库应用数据库。

开发环境使用 SQLite，表结构按后续 PostgreSQL 迁移来写。
会话和消息不写入 product_shell.json。
查询必须同时匹配 owner_id 和 kb_id。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


SCHEMA_VERSION = "app-003"
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
)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def app_db_path(working_dir: Path) -> Path:
    return Path(working_dir) / APP_DB_NAME


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
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
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
            self._conn.commit()

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
            self._conn.commit()
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
            self._conn.commit()
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
            self._conn.commit()
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
            self._conn.commit()
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
            self._conn.commit()
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
            self._conn.commit()
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
            self._conn.commit()
        return self.get_memory_candidate(candidate_id, owner_id, kb_id)

    def accept_memory_candidate(self, candidate_id: str, owner_id: str, kb_id: str | None = None) -> dict[str, Any] | None:
        current = self.get_memory_candidate(candidate_id, owner_id, kb_id)
        if current is None or current["status"] != "pending":
            return None
        memory = self.add_memory(
            owner_id=owner_id,
            content=current["content"],
            category=current["category"],
            source_type="candidate",
            source_id=current["id"],
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
    ) -> dict[str, Any]:
        now = _now()
        row = {
            "id": uuid4().hex,
            "owner_id": owner_id,
            "content": (content or "").strip(),
            "category": category or "other",
            "source_type": source_type,
            "source_id": source_id,
            "enabled": 1 if enabled else 0,
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO memories(
                    id, owner_id, content, category, source_type, source_id, enabled, created_at, updated_at
                ) VALUES (
                    :id, :owner_id, :content, :category, :source_type, :source_id, :enabled, :created_at, :updated_at
                )
                """,
                row,
            )
            self._conn.commit()
        return self._public_memory(row)

    def get_memory(self, memory_id: str, owner_id: str) -> dict[str, Any] | None:
        row = self._one("SELECT * FROM memories WHERE id = ? AND owner_id = ?", (memory_id, owner_id))
        return self._public_memory(row) if row else None

    def list_memories(self, owner_id: str) -> list[dict[str, Any]]:
        rows = self._all(
            "SELECT * FROM memories WHERE owner_id = ? ORDER BY updated_at DESC",
            (owner_id,),
        )
        return [self._public_memory(row) for row in rows]

    def enabled_memories(self, owner_id: str) -> list[dict[str, Any]]:
        rows = self._all(
            "SELECT * FROM memories WHERE owner_id = ? AND enabled = 1 ORDER BY updated_at DESC",
            (owner_id,),
        )
        return [self._public_memory(row) for row in rows]

    def update_memory(
        self,
        memory_id: str,
        owner_id: str,
        *,
        content: str | None = None,
        category: str | None = None,
        enabled: bool | None = None,
    ) -> dict[str, Any] | None:
        current = self.get_memory(memory_id, owner_id)
        if current is None:
            return None
        next_content = current["content"] if content is None else content.strip()
        next_category = current["category"] if category is None else category
        next_enabled = current["enabled"] if enabled is None else enabled
        with self._lock:
            self._conn.execute(
                """
                UPDATE memories
                SET content = ?, category = ?, enabled = ?, updated_at = ?
                WHERE id = ? AND owner_id = ?
                """,
                (next_content, next_category, 1 if next_enabled else 0, _now(), memory_id, owner_id),
            )
            self._conn.commit()
        return self.get_memory(memory_id, owner_id)

    def delete_memory(self, memory_id: str, owner_id: str) -> bool:
        found = self.get_memory(memory_id, owner_id)
        if found is None:
            return False
        with self._lock:
            self._conn.execute("DELETE FROM memories WHERE id = ? AND owner_id = ?", (memory_id, owner_id))
            self._conn.commit()
        return True

    def count_documents_untouched(self) -> None:
        return None

    def purge_kb(self, kb_id: str, owner_id: str | None = None) -> None:
        """删除知识库时清理会话、消息、反馈和记忆候选。

        产品策略：个人记忆是用户级资料，不随知识库删除。用户需在记忆页自行关闭或删除。
        """
        with self._lock:
            if owner_id:
                self._conn.execute("DELETE FROM offline_eval_candidates WHERE kb_id = ? AND owner_id = ?", (kb_id, owner_id))
                self._conn.execute("DELETE FROM answer_feedback WHERE kb_id = ? AND owner_id = ?", (kb_id, owner_id))
                self._conn.execute("DELETE FROM memory_candidates WHERE kb_id = ? AND owner_id = ?", (kb_id, owner_id))
                self._conn.execute("DELETE FROM messages WHERE kb_id = ? AND owner_id = ?", (kb_id, owner_id))
                self._conn.execute("DELETE FROM conversations WHERE kb_id = ? AND owner_id = ?", (kb_id, owner_id))
            else:
                self._conn.execute("DELETE FROM offline_eval_candidates WHERE kb_id = ?", (kb_id,))
                self._conn.execute("DELETE FROM answer_feedback WHERE kb_id = ?", (kb_id,))
                self._conn.execute("DELETE FROM memory_candidates WHERE kb_id = ?", (kb_id,))
                self._conn.execute("DELETE FROM messages WHERE kb_id = ?", (kb_id,))
                self._conn.execute("DELETE FROM conversations WHERE kb_id = ?", (kb_id,))
            self._conn.commit()

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
        return {
            "id": data["id"],
            "owner_id": data["owner_id"],
            "content": data.get("content") or "",
            "category": data.get("category") or "other",
            "source_type": data.get("source_type") or "candidate",
            "source_id": data.get("source_id") or "",
            "enabled": bool(data.get("enabled")),
            "created_at": data["created_at"],
            "updated_at": data["updated_at"],
        }
