"""上传和删除任务的阶段执行。

每个阶段先在内存里准备结果，确认任务没有被取消后再写入。
重试前清掉这个文档上一轮的切块、向量和图，源文件保留。
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from lightrag.product_db import backoff_seconds, job_settings
from lightrag.product_document_ir import public_location
from lightrag.product_ids import chunk_record
from lightrag.product_index import embed_texts, extract_and_store, write_chunk_vectors
from lightrag.product_parse import ParseError, chunks_from_document, parse_bytes, plain_text, save_ir
from lightrag.product_storage import (
    CommitGate,
    bind_commit_gate,
    document_consistency,
    execute_purge,
    load_shell,
    remove_documents,
    save_shell,
)

STAGES = ("parsing", "chunking", "embedding", "graphing", "validating")
_PROGRESS = {
    "parsing": 20,
    "chunking": 40,
    "embedding": 60,
    "graphing": 80,
    "validating": 92,
    "ready": 100,
}
_FAILS = {"used": 0}
_FAIL_LOCK = threading.Lock()


class StageError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def reset_stage_failures() -> None:
    with _FAIL_LOCK:
        _FAILS["used"] = 0


def _pause(stage: str) -> None:
    if os.getenv("JOB_PAUSE_STAGE", "") != stage:
        return
    try:
        seconds = float(os.getenv("JOB_PAUSE_SECONDS", "0") or "0")
    except ValueError:
        seconds = 0
    if seconds > 0:
        marker = os.getenv("WORKING_DIR", "")
        if marker:
            with open(Path(marker) / "stage.log", "a", encoding="utf-8") as handle:
                handle.write(stage + "\n")
        deadline = time.time() + seconds
        while time.time() < deadline:
            if marker and (Path(marker) / "cancel.flag").exists():
                raise StageError("cancelled", "已取消")
            time.sleep(0.05)


def _maybe_fail(stage: str) -> None:
    target = os.getenv("JOB_FAIL_STAGE", "").strip()
    if target != stage:
        return
    try:
        times = int(os.getenv("JOB_FAIL_TIMES", "1") or "1")
    except ValueError:
        times = 1
    with _FAIL_LOCK:
        if _FAILS["used"] >= times:
            return
        _FAILS["used"] += 1
    code = os.getenv("JOB_FAIL_CODE", "external").strip() or "external"
    raise StageError(code, f"{stage} 失败，可以重试")


def _timeout(stage: str) -> int:
    settings = job_settings()
    return {
        "parsing": settings["parse_timeout"],
        "chunking": settings["chunk_timeout"],
        "embedding": settings["embed_timeout"],
        "graphing": settings["graph_timeout"],
        "validating": settings["validate_timeout"],
    }.get(stage, 60)


def _run_bounded(stage: str, func: Callable[[], Any]) -> Any:
    seconds = _timeout(stage)
    if seconds <= 0:
        return func()
    outcome: dict[str, Any] = {}
    gate = CommitGate()

    def target() -> None:
        bind_commit_gate(gate)
        try:
            outcome["value"] = func()
        except Exception as exc:
            outcome["error"] = exc
        finally:
            bind_commit_gate(None)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        gate.close()
        raise StageError("external", f"{stage} 超时")
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


def _source(inputs: Path, file_path: str) -> Path:
    path = Path(file_path)
    if path.is_file():
        return path
    return inputs / file_path


def _stored_location(item: dict[str, Any]) -> dict[str, Any]:
    located = public_location(item)
    kept = {key: value for key, value in located.items() if value not in (None, "", [])}
    if item.get("evidence_ids"):
        kept["evidence_ids"] = list(item["evidence_ids"])
    if item.get("evidence_refs"):
        kept["evidence_refs"] = list(item["evidence_refs"])
    return kept


def _chunks_for(text: str, doc_id: str, user_id: str, kb_id: str, chunk_size: int = 512, chunk_overlap: int = 0) -> list[dict[str, Any]]:
    from lightrag.product_parse import _split_tokens

    pieces = [part.strip() for part in text.split("\n\n") if part.strip()] or ([text.strip()] if text.strip() else [])
    rows: list[dict[str, Any]] = []
    for part in pieces:
        for piece in _split_tokens(part, chunk_size, chunk_overlap):
            rows.append(chunk_record(user_id, kb_id, doc_id, len(rows) + 1, piece))
    return rows


def _clear_partial(working: Path, inputs: Path, data: dict[str, Any], kb_id: str, doc_id: str) -> None:
    remove_documents(working, inputs, data, kb_id, [doc_id], preserve_index=True)
    save_shell(working, data)


def _set_doc_status(data: dict[str, Any], doc_id: str, status: str, chunk_ids: list[str] | None = None) -> None:
    row = (data.get("doc_index") or {}).get(doc_id)
    if isinstance(row, dict):
        row["status"] = status
        if chunk_ids is not None:
            row["chunk_ids"] = chunk_ids


def _write_records(working: Path, doc_id: str, kb_id: str, owner_id: str, text: str, chunks: list[dict[str, str]], file_path: str) -> None:
    from lightrag.product_storage import _kv, _save_kv

    docs = _kv(working, "full_docs")
    docs[doc_id] = {"content": text, "doc_id": doc_id, "kb_id": kb_id, "user_id": owner_id, "owner_id": owner_id}
    _save_kv(working, "full_docs", docs)
    status = _kv(working, "doc_status")
    status[doc_id] = {
        "status": "processing",
        "doc_id": doc_id,
        "kb_id": kb_id,
        "owner_id": owner_id,
        "file_path": file_path,
        "content_length": len(text),
        "chunks_count": len(chunks),
        "updated_at": _now(),
    }
    _save_kv(working, "doc_status", status)
    rows = _kv(working, "text_chunks")
    for item in chunks:
        record = {
            "content": item["content"],
            "chunk_id": item["chunk_id"],
            "source_id": item["source_id"],
            "doc_id": doc_id,
            "full_doc_id": doc_id,
            "kb_id": kb_id,
            "user_id": owner_id,
            "owner_id": owner_id,
            "file_path": file_path,
        }
        record.update(_stored_location(item))
        if item.get("parent_id"):
            record["parent_id"] = item["parent_id"]
        if item.get("index_text"):
            record["index_text"] = item["index_text"]
        record.pop("parent_content", None)
        rows[item["chunk_id"]] = record
    _save_kv(working, "text_chunks", rows)


def _write_vectors(working: Path, doc_id: str, kb_id: str, owner_id: str, chunks: list[dict[str, Any]]) -> None:
    from lightrag.index_manifest import save_manifest
    from lightrag.product_storage import _kv

    matrix = asyncio.run(embed_texts([item.get("index_text") or item["content"] for item in chunks]))
    write_chunk_vectors(working, doc_id, kb_id, owner_id, chunks, matrix)
    stored = _kv(working, "text_chunks")
    save_manifest(working, int(matrix.shape[1]), list(stored.keys()))


def _write_graph(working: Path, doc_id: str, kb_id: str, owner_id: str, document_text: str, chunks: list[dict[str, Any]], file_path: str) -> None:
    asyncio.run(extract_and_store(working, doc_id, kb_id, owner_id, document_text, chunks, file_path))


def _mark(store: Any, job: dict[str, Any], stage: str, progress: int) -> dict[str, Any]:
    updated = store.transition(job["job_id"], "running", stage=stage, progress=progress)
    return updated or job


def _interrupt(store: Any, working: Path, inputs: Path, job: dict[str, Any], alive: Callable[[], bool]) -> dict[str, Any] | None:
    if _cancelled(store, job["job_id"]):
        return _finish_cancel(store, working, inputs, job)
    if not alive():
        store.release(job["job_id"], str(job.get("worker_id") or ""))
        return store.get_job(job["job_id"]) or job
    return None


def _cancelled(store: Any, job_id: str) -> bool:
    current = store.get_job(job_id) or {}
    return current.get("status") in {"cancel_requested", "cancelling", "cancelled"}


def _finish_cancel(store: Any, working: Path, inputs: Path, job: dict[str, Any]) -> dict[str, Any]:
    data = load_shell(working)
    doc_id = str(job.get("doc_id") or "")
    if doc_id:
        _clear_partial(working, inputs, data, str(job.get("kb_id") or ""), doc_id)
        data = load_shell(working)
        _set_doc_status(data, doc_id, "cancelled", [])
        save_shell(working, data)
    return store.transition(job["job_id"], "cancelled", stage="cancelled", progress=0, error_code="", error_message="已取消")


def _fail(store: Any, working: Path, inputs: Path, job: dict[str, Any], code: str, message: str, status: str) -> dict[str, Any]:
    data = load_shell(working)
    doc_id = str(job.get("doc_id") or "")
    if doc_id:
        _clear_partial(working, inputs, data, str(job.get("kb_id") or ""), doc_id)
        data = load_shell(working)
        _set_doc_status(data, doc_id, "cancelled" if status == "cancelled" else status, [])
        save_shell(working, data)
    attempt = int((store.get_job(job["job_id"]) or job).get("attempt") or 0)
    maximum = int(job.get("max_attempts") or 1)
    if status == "failed" and code != "user_input" and attempt < maximum:
        delay = backoff_seconds(attempt)
        retry_at = datetime.fromtimestamp(datetime.now(timezone.utc).timestamp() + delay, timezone.utc).isoformat()
        return store.transition(
            job["job_id"],
            "queued",
            stage="queued",
            error_code=code,
            error_message=message,
            retry_at=retry_at,
            worker_id="",
            lease_until="",
        )
    return store.transition(job["job_id"], status, stage=status, error_code=code, error_message=message, progress=0)


def run_ingestion(store: Any, job: dict[str, Any], working: Path, inputs: Path, alive: Callable[[], bool]) -> dict[str, Any]:
    working = Path(working)
    inputs = Path(inputs)
    if _cancelled(store, job["job_id"]):
        return _finish_cancel(store, working, inputs, job)
    snapshot = job.get("input_snapshot") or {}
    doc_id = str(job.get("doc_id") or "")
    kb_id = str(job.get("kb_id") or "")
    owner_id = str(snapshot.get("owner_id") or job.get("user_id") or "")
    source = _source(inputs, str(job.get("file_path") or snapshot.get("storage_key") or ""))
    data = load_shell(working)
    if doc_id:
        _clear_partial(working, inputs, data, kb_id, doc_id)
        data = load_shell(working)
        _set_doc_status(data, doc_id, "parsing", [])
        save_shell(working, data)
    started = job.get("started_at") or _now()
    try:
        started_dt = datetime.fromisoformat(started)
    except ValueError:
        started_dt = datetime.now(timezone.utc)
    if started_dt.tzinfo is None:
        started_dt = started_dt.replace(tzinfo=timezone.utc)

    text = ""
    parsed = None
    chunks: list[dict[str, Any]] = []
    for stage in STAGES:
        if not alive():
            store.release(job["job_id"], str(job.get("worker_id") or ""))
            return store.get_job(job["job_id"]) or job
        if _cancelled(store, job["job_id"]):
            return _finish_cancel(store, working, inputs, job)
        elapsed = (datetime.now(timezone.utc) - started_dt).total_seconds()
        if elapsed > job_settings()["overall_timeout"]:
            return _fail(store, working, inputs, job, "external", "任务整体超时", "timeout")
        job = _mark(store, job, stage, _PROGRESS[stage])
        data = load_shell(working)
        _set_doc_status(data, doc_id, stage)
        save_shell(working, data)
        try:
            if stage == "parsing":
                def parse() -> tuple[str, Any]:
                    _pause(stage)
                    _maybe_fail(stage)
                    if not source.is_file():
                        raise StageError("user_input", "找不到上传文件")
                    raw = source.read_bytes()
                    if not raw.strip():
                        raise StageError("user_input", "文件内容为空")
                    try:
                        document = parse_bytes(source.name, raw, document_id=doc_id)
                    except ParseError as exc:
                        if "不支持的文件类型" not in str(exc):
                            raise StageError("user_input", str(exc)) from exc
                        fallback = raw.decode("utf-8", errors="replace").strip()
                        if not fallback:
                            raise StageError("user_input", "文件内容为空") from exc
                        return fallback, None
                    save_ir(working, document)
                    located = plain_text(document).strip()
                    if not located:
                        if document.source_type in {"image", "audio"}:
                            raise StageError("user_input", "图片和语音本阶段不生成可检索正文")
                        raise StageError("user_input", "文件没有可检索正文")
                    return located, document

                text, parsed = _run_bounded(stage, parse)
                interrupted = _interrupt(store, working, inputs, job, alive)
                if interrupted is not None:
                    return interrupted
            elif stage == "chunking":
                def chunk() -> list[dict[str, Any]]:
                    _maybe_fail(stage)
                    size = int(snapshot.get("chunk_size") or 512)
                    overlap = int(snapshot.get("chunk_overlap") or 0)
                    if parsed is not None and size >= 180:
                        made = chunks_from_document(parsed, owner_id, kb_id, doc_id, chunk_size=250, chunk_overlap=45)
                    elif parsed is not None:
                        made = chunks_from_document(parsed, owner_id, kb_id, doc_id, chunk_size=size, chunk_overlap=overlap)
                    else:
                        made = _chunks_for(text, doc_id, owner_id, kb_id, chunk_size=size, chunk_overlap=overlap)
                    from lightrag.chunk_hierarchy import assign_parents
                    from lightrag.product_storage import _kv, _save_kv

                    made, parents = assign_parents(made, str(job.get("file_name") or doc_id), doc_id, split=size >= 180)
                    parent_rows = _kv(working, "parent_chunks")
                    for parent in parents:
                        parent_rows[parent["parent_id"]] = parent
                    _save_kv(working, "parent_chunks", parent_rows)
                    if not made:
                        raise StageError("user_input", "没有切出可用文本")
                    return made

                chunks = _run_bounded(stage, chunk)
                interrupted = _interrupt(store, working, inputs, job, alive)
                if interrupted is not None:
                    return interrupted
                _write_records(working, doc_id, kb_id, owner_id, text, chunks, str(job.get("file_path") or ""))
                data = load_shell(working)
                _set_doc_status(data, doc_id, "chunking", [item["chunk_id"] for item in chunks])
                save_shell(working, data)
            elif stage == "embedding":
                def embed() -> None:
                    _pause(stage)
                    _maybe_fail(stage)
                    _write_vectors(working, doc_id, kb_id, owner_id, chunks)

                _run_bounded(stage, embed)
                interrupted = _interrupt(store, working, inputs, job, alive)
                if interrupted is not None:
                    return interrupted
            elif stage == "graphing":
                def graph() -> None:
                    _pause(stage)
                    _maybe_fail(stage)
                    _write_graph(working, doc_id, kb_id, owner_id, text, chunks, str(job.get("file_path") or ""))

                _run_bounded(stage, graph)
                interrupted = _interrupt(store, working, inputs, job, alive)
                if interrupted is not None:
                    return interrupted
            else:
                def validate() -> dict[str, Any]:
                    _maybe_fail(stage)
                    if os.getenv("JOB_FORCE_CONSISTENCY_FAIL", "") == "1":
                        return {"passed": False, "checked_kb_id": kb_id, "checked_doc_ids": [doc_id], "checked_chunk_ids": []}
                    return document_consistency(
                        working,
                        kb_id,
                        doc_id,
                        [item["chunk_id"] for item in chunks],
                    )

                report = _run_bounded(stage, validate)
                check_id = store.save_check(job["job_id"], report)
                if not report.get("passed"):
                    data = load_shell(working)
                    _set_doc_status(data, doc_id, "consistency_failed")
                    save_shell(working, data)
                    return store.transition(
                        job["job_id"],
                        "consistency_failed",
                        stage="validating",
                        error_code="consistency",
                        error_message="一致性检查未通过",
                        consistency_check_id=check_id,
                        output_summary={"chunk_ids": [item["chunk_id"] for item in chunks]},
                    )
                data = load_shell(working)
                _set_doc_status(data, doc_id, "ready", [item["chunk_id"] for item in chunks])
                status_rows = {}
                from lightrag.product_storage import _kv, _save_kv

                status_rows = _kv(working, "doc_status")
                if doc_id in status_rows:
                    status_rows[doc_id]["status"] = "processed"
                    _save_kv(working, "doc_status", status_rows)
                save_shell(working, data)
                return store.transition(
                    job["job_id"],
                    "succeeded",
                    stage="ready",
                    progress=100,
                    error_code="",
                    error_message="",
                    consistency_check_id=check_id,
                    output_summary={"chunk_ids": [item["chunk_id"] for item in chunks], "content_hash": snapshot.get("content_hash", "")},
                )
        except StageError as exc:
            if exc.code == "cancelled":
                return _finish_cancel(store, working, inputs, job)
            status = "timeout" if "超时" in str(exc) else "failed"
            return _fail(store, working, inputs, job, exc.code, str(exc), status)
        except Exception as exc:
            return _fail(store, working, inputs, job, "internal", "内部处理失败", "failed")
    return store.get_job(job["job_id"]) or job


def run_purge(store: Any, job: dict[str, Any], working: Path, inputs: Path) -> dict[str, Any]:
    if _cancelled(store, job["job_id"]):
        return store.transition(job["job_id"], "cancelled", stage="cancelled", error_message="已取消")
    data = load_shell(working)
    shell_job = next((item for item in data.get("purge_jobs") or [] if item.get("job_id") == job["job_id"]), None)
    if shell_job is None:
        data.setdefault("purge_jobs", []).append({
            "job_id": job["job_id"],
            "kb_id": job.get("kb_id"),
            "status": "pending",
            "step": "queued",
            "error": "",
            "attempts": 0,
            "created_at": job.get("created_at") or _now(),
        })
        save_shell(working, data)
    store.transition(job["job_id"], "running", stage="deleting", progress=30)
    report = execute_purge(working, inputs, job["job_id"])
    check_id = store.save_check(job["job_id"], report if isinstance(report, dict) else {"passed": False})
    status = str((report or {}).get("job_status") or "failed")
    if status not in {"succeeded", "failed", "consistency_failed"}:
        status = "failed"
    message = "" if status == "succeeded" else str((report or {}).get("error") or "删除未完成")
    code = "" if status == "succeeded" else ("consistency" if status == "consistency_failed" else "internal")
    return store.transition(
        job["job_id"],
        status,
        stage="checked" if status == "succeeded" else status,
        progress=100 if status == "succeeded" else 0,
        error_code=code,
        error_message=message,
        consistency_check_id=check_id,
    )


def run_delete_document(store: Any, job: dict[str, Any], working: Path, inputs: Path) -> dict[str, Any]:
    if _cancelled(store, job["job_id"]):
        return store.transition(job["job_id"], "cancelled", stage="cancelled", error_message="已取消")
    doc_id = str(job.get("doc_id") or "")
    kb_id = str(job.get("kb_id") or "")
    data = load_shell(working)
    store.transition(job["job_id"], "running", stage="deleting", progress=40)
    remove_documents(working, inputs, data, kb_id, [doc_id])
    save_shell(working, data)
    report = document_consistency(working, kb_id, doc_id, [], expect="absent")
    check_id = store.save_check(job["job_id"], report)
    if report.get("passed"):
        return store.transition(job["job_id"], "succeeded", stage="checked", progress=100, error_message="", consistency_check_id=check_id)
    return store.transition(
        job["job_id"],
        "consistency_failed",
        stage="checked",
        error_code="consistency",
        error_message="删除后一致性检查未通过",
        consistency_check_id=check_id,
    )


def content_key(user_id: str, kb_id: str, payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    return f"ingestion:{user_id}:{kb_id}:{digest}"
