"""记录单个文件当前处理到哪一步，供知识列表轮询。"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}


def _name(file_path: str | Path) -> str:
    return Path(str(file_path)).name


def mark(
    file_path: str | Path,
    stage: str,
    progress: int,
    file_size: int = 0,
    kb_id: str = "",
    job_key: str = "",
) -> None:
    name = _name(file_path)
    key = job_key or name
    if not key:
        return
    with _LOCK:
        current = _JOBS.get(key, {})
        size = file_size or int(current.get("file_size") or 0)
        _JOBS[key] = {
            "name": name,
            "doc_id": job_key or str(current.get("doc_id") or ""),
            "file_path": str(file_path),
            "stage": stage,
            "progress": max(0, min(int(progress), 99)),
            "file_size": size,
            "kb_id": kb_id or str(current.get("kb_id") or ""),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }


def clear(file_path: str | Path) -> None:
    name = _name(file_path)
    with _LOCK:
        _JOBS.pop(name, None)


def cancel(file_path: str | Path) -> bool:
    name = _name(file_path)
    with _LOCK:
        job = _JOBS.get(name)
        if not job or job.get("stage") in {"ready", "failed", "cancelled"}:
            return False
        job["stage"] = "cancelled"
        job["progress"] = 0
        job["updated_at"] = datetime.now(timezone.utc).isoformat()
        return True


def is_cancelled(file_path: str | Path) -> bool:
    target = str(file_path)
    name = _name(file_path)
    with _LOCK:
        rows = [dict(row) for row in _JOBS.values()]
    for job in rows:
        if job.get("stage") != "cancelled":
            continue
        if (
            job.get("file_path") == target
            or job.get("name") == name
            or job.get("doc_id") == name
        ):
            return True
    return False


def expire_stale_jobs(minutes: int | None = None) -> None:
    import os

    if minutes is None:
        try:
            minutes = int(os.getenv("UPLOAD_TIMEOUT_MINUTES", "30") or "30")
        except ValueError:
            minutes = 30
    cutoff = datetime.now(timezone.utc).timestamp() - max(1, minutes) * 60
    with _LOCK:
        for job in _JOBS.values():
            if job.get("stage") in {"ready", "failed", "cancelled"}:
                continue
            try:
                updated = datetime.fromisoformat(str(job.get("updated_at") or ""))
            except ValueError:
                continue
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            if updated.timestamp() < cutoff:
                job["failed_stage"] = str(job.get("stage") or "parsing")
                job["stage"] = "failed"
                job["error_msg"] = "处理超时"


def get(name: str) -> dict[str, Any] | None:
    key = _name(name)
    with _LOCK:
        row = _JOBS.get(key)
        return dict(row) if row else None


def all_jobs() -> list[dict[str, Any]]:
    with _LOCK:
        return [dict(row) for row in _JOBS.values()]
