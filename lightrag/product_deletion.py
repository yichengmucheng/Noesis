"""知识库软删除和清理任务。

删除先标记 deleted_at 并记下任务，真正清文档、向量和源文件由调用方按步骤执行。
失败后保留任务记录，可以重试。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job.get("job_id"),
        "kb_id": job.get("kb_id"),
        "status": job.get("status"),
        "step": job.get("step"),
        "error": job.get("error") or "",
        "attempts": int(job.get("attempts") or 0),
    }


def begin_purge(data: dict[str, Any], kb: dict[str, Any]) -> dict[str, Any]:
    kb["deleted_at"] = _now()
    job = {
        "job_id": uuid4().hex,
        "kb_id": kb.get("id"),
        "status": "pending",
        "step": "queued",
        "error": "",
        "attempts": 0,
        "created_at": _now(),
    }
    kb["status"] = "deleting"
    jobs = data.get("purge_jobs")
    if not isinstance(jobs, list):
        jobs = []
        data["purge_jobs"] = jobs
    jobs.append(job)
    return job


def filenames_used_by_others(data: dict[str, Any], kb_id: str) -> set[str]:
    return {
        name
        for name, bound in (data.get("file_bindings") or {}).items()
        if bound != kb_id
    }


def consistency_report(
    data: dict[str, Any],
    kb_id: str,
    cache_entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    bindings = bound_filenames(data, kb_id)
    qa = [
        item.get("id")
        for item in (data.get("qa_pairs") or [])
        if item.get("kb_id") == kb_id
    ]
    comparisons = [
        item.get("task_id")
        for item in (data.get("comparisons") or [])
        if item.get("kb_id") == kb_id
    ]
    cache = [
        item.get("kb_id")
        for item in (cache_entries or [])
        if item.get("kb_id") == kb_id
    ]
    return {
        "kb_id": kb_id,
        "bindings": bindings,
        "qa": qa,
        "comparisons": comparisons,
        "cache": cache,
        "clean": not bindings and not qa and not comparisons and not cache,
    }


def bound_filenames(data: dict[str, Any], kb_id: str) -> list[str]:
    return [
        name
        for name, bound in (data.get("file_bindings") or {}).items()
        if bound == kb_id
    ]


def apply_metadata_purge(
    data: dict[str, Any],
    kb_id: str,
    cache_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    data["file_bindings"] = {
        name: bound
        for name, bound in (data.get("file_bindings") or {}).items()
        if bound != kb_id
    }
    data["qa_pairs"] = [
        item for item in (data.get("qa_pairs") or []) if item.get("kb_id") != kb_id
    ]
    data["comparisons"] = [
        item for item in (data.get("comparisons") or []) if item.get("kb_id") != kb_id
    ]
    audits = data.get("audits") or {}
    if isinstance(audits, dict):
        data["audits"] = {
            key: value
            for key, value in audits.items()
            if not isinstance(value, dict) or value.get("kb_id") != kb_id
        }
    sessions = data.get("sessions") or {}
    if isinstance(sessions, dict):
        data["sessions"] = {
            key: value
            for key, value in sessions.items()
            if not isinstance(value, dict) or value.get("kb_id") != kb_id
        }
    return [item for item in cache_entries if item.get("kb_id") != kb_id]


def mark_job(job: dict[str, Any], *, ok: bool, error: str, leftover: list[str]) -> None:
    job["attempts"] = int(job.get("attempts") or 0) + 1
    if ok and not leftover:
        job["status"] = "succeeded"
        job["step"] = "checked"
        job["error"] = ""
        return
    job["status"] = "failed"
    job["step"] = "checked"
    detail = error.strip()
    if leftover:
        names = "、".join(leftover[:8])
        detail = (detail + "；" if detail else "") + f"仍有文件绑定：{names}"
    job["error"] = detail[:300]


def queue_retry(job: dict[str, Any]) -> None:
    if job.get("status") in {"succeeded", "success"}:
        raise ValueError("清理已完成")
    job["status"] = "pending"
    job["step"] = "queued"
    job["error"] = ""
