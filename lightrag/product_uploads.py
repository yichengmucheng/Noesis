"""上传文件的物理位置、版本和失败清理。

物理路径是 owner_id/kb_id/doc_id/显示文件名。显示名可以在不同库里重复。
同一知识库再次上传同一显示名时创建新版本，不覆盖其他用户的文件。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from uuid import uuid4

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,80}$")


def _safe_id(value: str, label: str) -> str:
    text = str(value or "").strip()
    if not _ID_RE.match(text):
        raise ValueError(f"{label}无效")
    return text


def allocate_upload(
    data: dict[str, Any], owner_id: str, kb_id: str, display_name: str
) -> dict[str, Any]:
    owner = _safe_id(owner_id, "用户")
    kb = _safe_id(kb_id, "知识库")
    name = Path(str(display_name or "")).name
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("文件名无效")
    index = data.setdefault("doc_index", {})
    versions = [
        int(item.get("version") or 1)
        for item in index.values()
        if isinstance(item, dict)
        and item.get("kb_id") == kb
        and item.get("display_name") == name
        and not item.get("deleted_at")
    ]
    version = (max(versions) if versions else 0) + 1
    doc_id = "doc-" + uuid4().hex
    storage_key = f"{owner}/{kb}/{doc_id}/{name}"
    record = {
        "doc_id": doc_id,
        "owner_id": owner,
        "kb_id": kb,
        "display_name": name,
        "storage_key": storage_key,
        "version": version,
        "status": "uploading",
        "chunk_ids": [],
    }
    index[doc_id] = record
    return record


def abort_upload(
    data: dict[str, Any],
    working_dir: Path,
    input_dir: Path,
    doc_id: str,
    stage: str,
    *,
    cancelled: bool = False,
) -> dict[str, Any]:
    """失败或取消后删掉这个文档已经写下的半成品，避免被检索到。"""
    from lightrag.product_storage import remove_documents

    index = data.setdefault("doc_index", {})
    record = index.get(doc_id) or {}
    kb_id = str(record.get("kb_id") or "")
    remove_documents(working_dir, input_dir, data, kb_id, [doc_id])
    record["status"] = "cancelled" if cancelled else "failed"
    record["failed_stage"] = stage
    jobs = data.setdefault("upload_jobs", {})
    jobs[doc_id] = {
        "doc_id": doc_id,
        "kb_id": kb_id,
        "stage": "cancelled" if cancelled else "failed",
        "error_msg": "已取消" if cancelled else "处理超时",
        "failed_stage": stage,
    }
    return jobs[doc_id]
