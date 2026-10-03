"""产品任务库。

业务代码只使用 ProductStore。开发环境用 SQLite，生产环境用 PostgreSQL。
两种后端共用同一套表结构，调用方不写某一种数据库的 API。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

ACTIVE_STATUSES = ("queued", "running", "cancel_requested", "cancelling")
TERMINAL_STATUSES = ("succeeded", "failed", "timeout", "cancelled", "consistency_failed")
RETRYABLE_STATUSES = ("failed", "timeout", "consistency_failed")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS product_jobs (
    job_id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL,
    user_id TEXT NOT NULL,
    kb_id TEXT NOT NULL,
    doc_id TEXT NOT NULL DEFAULT '',
    file_name TEXT NOT NULL DEFAULT '',
    file_path TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    stage TEXT NOT NULL DEFAULT '',
    progress INTEGER NOT NULL DEFAULT 0,
    attempt INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    error_code TEXT NOT NULL DEFAULT '',
    error_message TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT '',
    heartbeat_at TEXT NOT NULL DEFAULT '',
    finished_at TEXT NOT NULL DEFAULT '',
    cancel_requested_at TEXT NOT NULL DEFAULT '',
    retry_at TEXT NOT NULL DEFAULT '',
    worker_id TEXT NOT NULL DEFAULT '',
    idempotency_key TEXT NOT NULL,
    input_snapshot TEXT NOT NULL DEFAULT '{}',
    output_summary TEXT NOT NULL DEFAULT '{}',
    consistency_check_id TEXT NOT NULL DEFAULT '',
    lease_until TEXT NOT NULL DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS product_jobs_idempotency ON product_jobs(idempotency_key);
CREATE INDEX IF NOT EXISTS product_jobs_status ON product_jobs(status, retry_at);
CREATE INDEX IF NOT EXISTS product_jobs_owner ON product_jobs(user_id, kb_id, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS product_jobs_active_doc ON product_jobs(doc_id, job_type)
    WHERE status IN ('queued', 'running', 'cancel_requested', 'cancelling') AND doc_id <> '';
CREATE TABLE IF NOT EXISTS consistency_checks (
    check_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL DEFAULT '',
    kb_id TEXT NOT NULL DEFAULT '',
    doc_id TEXT NOT NULL DEFAULT '',
    passed INTEGER NOT NULL DEFAULT 0,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL
);
"""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime | None = None) -> str:
    return (moment or _now()).isoformat()


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except ValueError:
        return default


def job_settings() -> dict[str, Any]:
    lease = max(1, env_int("JOB_LEASE_SECONDS", 30))
    heartbeat = env_int("JOB_HEARTBEAT_SECONDS", 10)
    if heartbeat >= lease:
        heartbeat = max(1, lease // 3)
    return {
        "worker_count": max(1, env_int("WORKER_COUNT", 1)),
        "poll_interval": max(0.2, env_float("WORKER_POLL_INTERVAL", 1)),
        "lease_seconds": lease,
        "heartbeat_seconds": heartbeat,
        "max_attempts": max(1, env_int("JOB_MAX_ATTEMPTS", 3)),
        "retry_backoff": max(0.0, env_float("JOB_RETRY_BACKOFF", 2)),
        "max_per_user": max(1, env_int("MAX_CONCURRENT_JOBS_PER_USER", 2)),
        "max_per_kb": max(1, env_int("MAX_CONCURRENT_JOBS_PER_KB", 2)),
        "parse_timeout": env_int("STAGE_PARSE_TIMEOUT_SECONDS", 120),
        "chunk_timeout": env_int("STAGE_CHUNK_TIMEOUT_SECONDS", 120),
        "embed_timeout": env_int("STAGE_EMBED_TIMEOUT_SECONDS", 180),
        "graph_timeout": env_int("STAGE_GRAPH_TIMEOUT_SECONDS", 600),
        "validate_timeout": env_int("STAGE_VALIDATE_TIMEOUT_SECONDS", 60),
        "overall_timeout": env_int("JOB_OVERALL_TIMEOUT_SECONDS", 900),
    }


def database_url() -> str:
    return os.getenv("DATABASE_URL", "").strip()


def production_database_error() -> str | None:
    if os.getenv("APP_ENV", "development").strip().lower() != "production":
        return None
    url = database_url()
    if url.startswith("sqlite"):
        return "生产环境必须使用 PostgreSQL，不能使用 SQLite"
    if url and not url.startswith("postgres"):
        return "生产环境 DATABASE_URL 必须是 PostgreSQL"
    return None


def default_sqlite_path() -> Path:
    configured = os.getenv("PRODUCT_DB_PATH", "").strip()
    if configured:
        return Path(configured)
    working = Path(os.getenv("WORKING_DIR", "./data/rag_storage"))
    return working / "product_jobs.sqlite"


def open_store(path: Path | None = None) -> "ProductStore":
    url = database_url()
    if url.startswith("postgres"):
        return PostgresStore(url)
    if url.startswith("sqlite:///"):
        return SqliteStore(Path(url[len("sqlite:///") :]))
    return SqliteStore(path or default_sqlite_path())


def _loads(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _row_dict(row: Any) -> dict[str, Any]:
    item = dict(row)
    item["input_snapshot"] = _loads(str(item.get("input_snapshot") or "{}"))
    item["output_summary"] = _loads(str(item.get("output_summary") or "{}"))
    item["attempt"] = int(item.get("attempt") or 0)
    item["max_attempts"] = int(item.get("max_attempts") or 0)
    item["progress"] = int(item.get("progress") or 0)
    return item


class ProductStore:
    """TaskStore、JobLeaseStore、ConsistencyCheckStore 的统一入口。"""

    def migrate(self) -> None:
        raise NotImplementedError

    def find_idempotent(self, key: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def create_job(self, fields: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def list_jobs(
        self,
        *,
        user_id: str,
        kb_id: str = "",
        doc_id: str = "",
        status: str = "",
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def latest_for_doc(self, user_id: str, doc_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def transition(self, job_id: str, status: str, **fields: Any) -> dict[str, Any] | None:
        raise NotImplementedError

    def request_cancel(self, job_id: str, user_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def request_retry(self, job_id: str, user_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def count_expired(self) -> int:
        raise NotImplementedError

    def recover_expired(self) -> int:
        raise NotImplementedError

    def acquire(self, worker_id: str, job_types: list[str] | None = None) -> dict[str, Any] | None:
        raise NotImplementedError

    def heartbeat(self, job_id: str, worker_id: str) -> bool:
        raise NotImplementedError

    def release(self, job_id: str, worker_id: str) -> None:
        raise NotImplementedError

    def save_check(self, job_id: str, report: dict[str, Any]) -> str:
        raise NotImplementedError

    def get_check(self, check_id: str) -> dict[str, Any] | None:
        raise NotImplementedError


class _SqlStore(ProductStore):
    def __init__(self) -> None:
        self._lock = threading.Lock()

    def _exec(self, sql: str, args: tuple = ()) -> int:
        raise NotImplementedError

    def _all(self, sql: str, args: tuple = ()) -> list[dict[str, Any]]:
        raise NotImplementedError

    def _one(self, sql: str, args: tuple = ()) -> dict[str, Any] | None:
        rows = self._all(sql, args)
        return rows[0] if rows else None

    def migrate(self) -> None:
        with self._lock:
            for statement in _SCHEMA.split(";"):
                text = statement.strip()
                if text:
                    self._exec(text)
            self._exec(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                ("product-jobs-1", _iso()),
            )

    def find_idempotent(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._one("SELECT * FROM product_jobs WHERE idempotency_key = ?", (key,))
        return _row_dict(row) if row else None

    def create_job(self, fields: dict[str, Any]) -> dict[str, Any]:
        key = str(fields.get("idempotency_key") or "").strip()
        if not key:
            raise ValueError("缺少幂等键")
        settings = job_settings()
        job_id = str(fields.get("job_id") or uuid4().hex)
        now = _iso()
        snapshot = fields.get("input_snapshot") or {}
        with self._lock:
            existing = self._one("SELECT * FROM product_jobs WHERE idempotency_key = ?", (key,))
            if existing:
                return _row_dict(existing)
            try:
                self._exec(
                    """
                    INSERT INTO product_jobs (
                        job_id, job_type, user_id, kb_id, doc_id, file_name, file_path,
                        status, stage, progress, attempt, max_attempts, error_code, error_message,
                        created_at, idempotency_key, input_snapshot, output_summary
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', 'queued', 0, 0, ?, '', '', ?, ?, ?, '{}')
                    """,
                    (
                        job_id,
                        str(fields.get("job_type") or "ingestion"),
                        str(fields.get("user_id") or ""),
                        str(fields.get("kb_id") or ""),
                        str(fields.get("doc_id") or ""),
                        str(fields.get("file_name") or ""),
                        str(fields.get("file_path") or ""),
                        int(fields.get("max_attempts") or settings["max_attempts"]),
                        now,
                        key,
                        json.dumps(snapshot, ensure_ascii=False),
                    ),
                )
            except Exception as exc:
                if "UNIQUE" in str(exc).upper() or "unique" in str(exc):
                    existing = self._one("SELECT * FROM product_jobs WHERE idempotency_key = ?", (key,))
                    if existing:
                        return _row_dict(existing)
                    active = self._one(
                        """
                        SELECT * FROM product_jobs
                        WHERE doc_id = ? AND job_type = ? AND status IN ('queued', 'running', 'cancel_requested', 'cancelling')
                        """,
                        (str(fields.get("doc_id") or ""), str(fields.get("job_type") or "")),
                    )
                    if active:
                        return _row_dict(active)
                raise
        created = self.get_job(job_id)
        if created is None:
            raise RuntimeError("任务创建失败")
        return created

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._one("SELECT * FROM product_jobs WHERE job_id = ?", (job_id,))
        return _row_dict(row) if row else None

    def list_jobs(
        self,
        *,
        user_id: str,
        kb_id: str = "",
        doc_id: str = "",
        status: str = "",
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        clauses = ["user_id = ?"]
        args: list[Any] = [user_id]
        if kb_id:
            clauses.append("kb_id = ?")
            args.append(kb_id)
        if doc_id:
            clauses.append("doc_id = ?")
            args.append(doc_id)
        if status:
            clauses.append("status = ?")
            args.append(status)
        where = " AND ".join(clauses)
        size = max(1, min(int(page_size or 20), 100))
        current = max(1, int(page or 1))
        offset = (current - 1) * size
        with self._lock:
            total_row = self._one(f"SELECT COUNT(*) AS total FROM product_jobs WHERE {where}", tuple(args))
            rows = self._all(
                f"SELECT * FROM product_jobs WHERE {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                tuple(args) + (size, offset),
            )
        total = int((total_row or {}).get("total") or 0)
        return {"items": [_row_dict(row) for row in rows], "total": total, "page": current, "page_size": size}

    def latest_for_doc(self, user_id: str, doc_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._one(
                "SELECT * FROM product_jobs WHERE user_id = ? AND doc_id = ? ORDER BY created_at DESC LIMIT 1",
                (user_id, doc_id),
            )
        return _row_dict(row) if row else None

    def transition(self, job_id: str, status: str, **fields: Any) -> dict[str, Any] | None:
        current = self.get_job(job_id)
        if current is None:
            return None
        if current["status"] == status and not fields:
            return current
        assignments = ["status = ?"]
        args: list[Any] = [status]
        for name, value in fields.items():
            if name in {"input_snapshot", "output_summary"} and not isinstance(value, str):
                value = json.dumps(value, ensure_ascii=False)
            assignments.append(f"{name} = ?")
            args.append(value)
        if status in TERMINAL_STATUSES and "finished_at" not in fields:
            assignments.append("finished_at = ?")
            args.append(_iso())
            assignments.append("lease_until = ?")
            args.append("")
            assignments.append("worker_id = ?")
            args.append("")
        args.append(job_id)
        with self._lock:
            self._exec(f"UPDATE product_jobs SET {', '.join(assignments)} WHERE job_id = ?", tuple(args))
        return self.get_job(job_id)

    def request_cancel(self, job_id: str, user_id: str) -> dict[str, Any] | None:
        job = self.get_job(job_id)
        if job is None or job["user_id"] != user_id:
            return None
        if job["status"] in {"cancelled", "succeeded"}:
            return job
        if job["status"] in TERMINAL_STATUSES:
            return job
        now = _iso()
        if job["status"] == "queued":
            return self.transition(job_id, "cancelled", cancel_requested_at=now, stage="cancelled", finished_at=now)
        return self.transition(job_id, "cancel_requested", cancel_requested_at=now)

    def request_retry(self, job_id: str, user_id: str) -> dict[str, Any] | None:
        job = self.get_job(job_id)
        if job is None or job["user_id"] != user_id:
            return None
        if job["status"] not in RETRYABLE_STATUSES:
            raise ValueError("当前状态不能重试")
        if job["attempt"] >= job["max_attempts"]:
            raise ValueError("已达到最大重试次数")
        return self.transition(
            job_id,
            "queued",
            stage="queued",
            error_code="",
            error_message="",
            retry_at="",
            started_at="",
            finished_at="",
            worker_id="",
            lease_until="",
        )

    def count_expired(self) -> int:
        now = _iso()
        with self._lock:
            row = self._one(
                """
                SELECT COUNT(*) AS total FROM product_jobs
                WHERE status IN ('running', 'cancelling') AND lease_until <> '' AND lease_until <= ?
                """,
                (now,),
            )
        return int((row or {}).get("total") or 0)

    def recover_expired(self) -> int:
        now = _iso()
        with self._lock:
            expired = self._all(
                """
                SELECT job_id, attempt, max_attempts FROM product_jobs
                WHERE status IN ('running', 'cancelling') AND lease_until <> '' AND lease_until <= ?
                """,
                (now,),
            )
            changed = 0
            for row in expired:
                if int(row["attempt"] or 0) >= int(row["max_attempts"] or 1):
                    self._exec(
                        """
                        UPDATE product_jobs
                        SET status = 'failed', error_code = 'internal', error_message = '任务中断次数已达上限',
                            finished_at = ?, worker_id = '', lease_until = ''
                        WHERE job_id = ? AND status IN ('running', 'cancelling')
                        """,
                        (now, row["job_id"]),
                    )
                else:
                    self._exec(
                        """
                        UPDATE product_jobs
                        SET status = 'queued', worker_id = '', lease_until = '', retry_at = ?
                        WHERE job_id = ? AND status IN ('running', 'cancelling')
                        """,
                        (now, row["job_id"]),
                    )
                changed += 1
        return changed

    def _running_count(self, column: str, value: str, now: str) -> int:
        row = self._one(
            f"""
            SELECT COUNT(*) AS total FROM product_jobs
            WHERE {column} = ? AND status = 'running' AND lease_until > ?
            """,
            (value, now),
        )
        return int((row or {}).get("total") or 0)

    def acquire(self, worker_id: str, job_types: list[str] | None = None) -> dict[str, Any] | None:
        self.recover_expired()
        settings = job_settings()
        now_dt = _now()
        now = _iso(now_dt)
        until = _iso(now_dt + timedelta(seconds=settings["lease_seconds"]))
        types = job_types or ["ingestion", "purge", "delete_document", "index_rebuild"]
        marks = ",".join("?" for _ in types)
        with self._lock:
            rows = self._all(
                f"""
                SELECT * FROM product_jobs
                WHERE job_type IN ({marks})
                  AND (
                    (status = 'queued' AND (retry_at = '' OR retry_at <= ?))
                    OR status = 'cancel_requested'
                    OR (status = 'running' AND lease_until <> '' AND lease_until <= ?)
                  )
                ORDER BY created_at
                """,
                tuple(types) + (now, now),
            )
            for row in rows:
                job = _row_dict(row)
                if job["status"] != "cancel_requested":
                    if self._running_count("user_id", job["user_id"], now) >= settings["max_per_user"]:
                        continue
                    if self._running_count("kb_id", job["kb_id"], now) >= settings["max_per_kb"]:
                        continue
                if job["attempt"] >= job["max_attempts"] and job["status"] != "cancel_requested":
                    self._exec(
                        """
                        UPDATE product_jobs
                        SET status = 'failed', error_code = 'internal', error_message = '已达到最大重试次数', finished_at = ?
                        WHERE job_id = ?
                        """,
                        (now, job["job_id"]),
                    )
                    continue
                changed = self._exec(
                    """
                    UPDATE product_jobs
                    SET status = 'running', worker_id = ?, lease_until = ?, heartbeat_at = ?,
                        started_at = CASE WHEN started_at = '' THEN ? ELSE started_at END,
                        attempt = attempt + 1
                    WHERE job_id = ? AND status = ? AND lease_until = ?
                    """,
                    (worker_id, until, now, now, job["job_id"], job["status"], job["lease_until"]),
                )
                if changed:
                    taken = self._one("SELECT * FROM product_jobs WHERE job_id = ?", (job["job_id"],))
                    return _row_dict(taken) if taken else None
        return None

    def heartbeat(self, job_id: str, worker_id: str) -> bool:
        settings = job_settings()
        now_dt = _now()
        changed = 0
        with self._lock:
            changed = self._exec(
                """
                UPDATE product_jobs
                SET heartbeat_at = ?, lease_until = ?
                WHERE job_id = ? AND worker_id = ? AND status IN ('running', 'cancel_requested', 'cancelling')
                """,
                (_iso(now_dt), _iso(now_dt + timedelta(seconds=settings["lease_seconds"])), job_id, worker_id),
            )
        return bool(changed)

    def release(self, job_id: str, worker_id: str) -> None:
        with self._lock:
            self._exec(
                """
                UPDATE product_jobs
                SET status = 'queued', worker_id = '', lease_until = '',
                    attempt = CASE WHEN attempt > 0 THEN attempt - 1 ELSE 0 END
                WHERE job_id = ? AND worker_id = ? AND status = 'running'
                """,
                (job_id, worker_id),
            )

    def save_check(self, job_id: str, report: dict[str, Any]) -> str:
        check_id = uuid4().hex
        with self._lock:
            self._exec(
                """
                INSERT INTO consistency_checks (check_id, job_id, kb_id, doc_id, passed, payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    check_id,
                    job_id,
                    str(report.get("checked_kb_id") or ""),
                    ",".join(report.get("checked_doc_ids") or []),
                    1 if report.get("passed") else 0,
                    json.dumps(report, ensure_ascii=False),
                    _iso(),
                ),
            )
            self._exec(
                "UPDATE product_jobs SET consistency_check_id = ? WHERE job_id = ?",
                (check_id, job_id),
            )
        return check_id

    def get_check(self, check_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._one("SELECT * FROM consistency_checks WHERE check_id = ?", (check_id,))
        if row is None:
            return None
        payload = _loads(str(row.get("payload") or "{}"))
        payload["check_id"] = row["check_id"]
        return payload


class SqliteStore(_SqlStore):
    def __init__(self, path: Path):
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self.migrate()

    def _exec(self, sql: str, args: tuple = ()) -> int:
        try:
            cursor = self._conn.execute(sql, args)
            self._conn.commit()
        except sqlite3.IntegrityError as exc:
            self._conn.rollback()
            raise exc
        return int(cursor.rowcount or 0)

    def _all(self, sql: str, args: tuple = ()) -> list[dict[str, Any]]:
        cursor = self._conn.execute(sql, args)
        return [dict(row) for row in cursor.fetchall()]

    def migrate(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                ("product-jobs-1", _iso()),
            )
            self._conn.commit()


class PostgresStore(_SqlStore):
    def __init__(self, url: str):
        super().__init__()
        self.url = url.replace("postgresql+asyncpg://", "postgresql://", 1)
        self._pool = None
        self.migrate()

    def _run(self, sql: str, args: tuple = (), *, fetch: str = "rowcount") -> Any:
        import asyncio

        import asyncpg

        async def work() -> Any:
            conn = await asyncpg.connect(self.url)
            try:
                statement, values = _postgres_sql(sql, args)
                if fetch == "all":
                    rows = await conn.fetch(statement, *values)
                    return [dict(row) for row in rows]
                if fetch == "exec":
                    await conn.execute(statement, *values)
                    return 0
                status = await conn.execute(statement, *values)
                parts = str(status).split()
                return int(parts[-1]) if parts and parts[-1].isdigit() else 0
            finally:
                await conn.close()

        return asyncio.run(work())

    def _exec(self, sql: str, args: tuple = ()) -> int:
        if sql.strip().upper().startswith("INSERT INTO SCHEMA_MIGRATIONS"):
            try:
                return int(self._run(sql, args) or 0)
            except Exception as exc:
                if "unique" in str(exc).lower() or "duplicate" in str(exc).lower():
                    return 0
                raise
        return int(self._run(sql, args) or 0)

    def _all(self, sql: str, args: tuple = ()) -> list[dict[str, Any]]:
        return list(self._run(sql, args, fetch="all") or [])

    def migrate(self) -> None:
        with self._lock:
            for statement in _SCHEMA.split(";"):
                text = statement.strip()
                if text:
                    self._run(text, fetch="exec")
            try:
                self._run(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    ("product-jobs-1", _iso()),
                    fetch="exec",
                )
            except Exception as exc:
                if "unique" not in str(exc).lower() and "duplicate" not in str(exc).lower():
                    raise


def _postgres_sql(sql: str, args: tuple) -> tuple[str, list[Any]]:
    parts = sql.split("?")
    if len(parts) == 1:
        return sql, []
    numbered = []
    for index, part in enumerate(parts[:-1], start=1):
        numbered.append(part)
        numbered.append(f"${index}")
    numbered.append(parts[-1])
    return "".join(numbered), list(args)


def public_job(job: dict[str, Any]) -> dict[str, Any]:
    message = _public_message(str(job.get("error_message") or job.get("error") or ""))
    return {
        "job_id": job.get("job_id"),
        "job_type": job.get("job_type"),
        "user_id": job.get("user_id"),
        "kb_id": job.get("kb_id"),
        "doc_id": job.get("doc_id") or "",
        "file_name": job.get("file_name") or "",
        "status": job.get("status"),
        "stage": job.get("stage") or job.get("step") or "",
        "step": job.get("stage") or job.get("step") or "",
        "progress": int(job.get("progress") or 0),
        "attempt": int(job.get("attempt") or job.get("attempts") or 0),
        "attempts": int(job.get("attempt") or job.get("attempts") or 0),
        "max_attempts": int(job.get("max_attempts") or 0),
        "error_code": job.get("error_code") or "",
        "error_message": message,
        "error": message,
        "created_at": job.get("created_at") or "",
        "started_at": job.get("started_at") or "",
        "finished_at": job.get("finished_at") or "",
        "retry_at": job.get("retry_at") or "",
        "consistency_check_id": job.get("consistency_check_id") or "",
    }


def _public_message(message: str) -> str:
    lowered = message.lower()
    if any(word in lowered for word in ("traceback", "token", "cookie", "password", "api_key", "secret", "sk-")):
        return "处理失败"
    if ":\\" in message or message.startswith("/app/") or message.startswith("/users/"):
        return "处理失败"
    return message[:300]


def backoff_seconds(attempt: int) -> float:
    base = job_settings()["retry_backoff"]
    return min(300.0, base * (2 ** max(0, attempt - 1)))
