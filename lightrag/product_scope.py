"""记录是否属于某个知识库。

有 kb_id 时以 kb_id 为准。没有 kb_id 时，只接受已经绑定到该库的旧文件名。
两边都无法确认时，不返回这条记录。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lightrag.constants import GRAPH_FIELD_SEP


def split_values(value: Any) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for part in str(value or "").split(GRAPH_FIELD_SEP):
        part = part.strip()
        if not part or part in seen:
            continue
        seen.add(part)
        found.append(part)
    return found


_NOT_READY = {
    "uploading",
    "queued",
    "parsing",
    "chunking",
    "embedding",
    "graphing",
    "validating",
    "failed",
    "cancelled",
    "consistency_failed",
    "timeout",
    "deleting",
}


def _doc_blocked(doc_index: dict[str, Any] | None, doc_id: str) -> bool:
    row = (doc_index or {}).get(doc_id)
    if not isinstance(row, dict):
        return False
    if row.get("deleted_at"):
        return True
    return str(row.get("status") or "") in _NOT_READY


def scope_visible(
    record: dict[str, Any] | None,
    kb_id: str,
    bindings: dict[str, str] | None = None,
    doc_index: dict[str, Any] | None = None,
) -> bool:
    if not kb_id or not isinstance(record, dict):
        return False
    doc_id = str(record.get("doc_id") or record.get("full_doc_id") or "").strip()
    if doc_id and _doc_blocked(doc_index, doc_id):
        return False
    tagged = split_values(record.get("kb_id"))
    if tagged:
        return kb_id in tagged
    doc_id = str(record.get("doc_id") or record.get("full_doc_id") or "").strip()
    indexed = (doc_index or {}).get(doc_id) if doc_id else None
    if isinstance(indexed, dict):
        return indexed.get("kb_id") == kb_id and not indexed.get("deleted_at")
    file_path = str(record.get("file_path") or "").strip()
    if not file_path:
        return False
    table = bindings or {}
    for part in split_values(file_path):
        name = Path(part.replace("\\", "/")).name
        if name and table.get(name) == kb_id:
            return True
        if part in table and table.get(part) == kb_id:
            return True
    return False
