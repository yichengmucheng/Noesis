"""SourceUnit → RetrievalChunk → ParentChunk。

只向量化小块。父块单独存放，不把完整父块正文复制进每个小块。
本轮索引用原文，不用模型摘要代替正文。
"""

from __future__ import annotations

import re
from typing import Any

RETRIEVAL_MIN = 180
RETRIEVAL_MAX = 350
RETRIEVAL_TARGET = 250
OVERLAP_MIN = 30
OVERLAP_MAX = 60
OVERLAP_TARGET = 45
PARENT_MIN = 800
PARENT_MAX = 1500


def count_tokens(text: str) -> int:
    body = str(text or "")
    cjk = re.findall(r"[\u4e00-\u9fff]", body)
    words = re.findall(r"[A-Za-z0-9_]+", body)
    total = len(cjk) + len(words)
    if total:
        return total
    return 1 if body.strip() else 0


def _token_pieces(text: str) -> list[str]:
    pieces: list[str] = []
    word: list[str] = []

    def flush() -> None:
        if word:
            pieces.append("".join(word))
            word.clear()

    for char in str(text or ""):
        if re.match(r"[A-Za-z0-9_]", char):
            word.append(char)
            continue
        flush()
        if char.strip():
            pieces.append(char)
    flush()
    return pieces


def split_token_windows(text: str, size: int = RETRIEVAL_TARGET, overlap: int = OVERLAP_TARGET) -> list[str]:
    pieces = _token_pieces(text)
    if not pieces:
        return []
    size = min(RETRIEVAL_MAX, max(RETRIEVAL_MIN, int(size)))
    overlap = min(OVERLAP_MAX, max(OVERLAP_MIN, int(overlap)))
    if overlap >= size:
        overlap = OVERLAP_MIN
    if len(pieces) <= size:
        return ["".join(pieces).strip()]
    windows: list[str] = []
    start = 0
    while start < len(pieces):
        end = min(len(pieces), start + size)
        piece = "".join(pieces[start:end]).strip()
        if piece:
            windows.append(piece)
        if end >= len(pieces):
            break
        start = end - overlap
    return windows


def index_text(document_name: str, section_path: list[str] | None, content: str) -> str:
    path = " / ".join(str(part) for part in (section_path or []) if str(part).strip())
    lines = [str(document_name or "").strip(), path, str(content or "").strip()]
    return "\n".join(line for line in lines if line)


def _section_key(item: dict[str, Any]) -> tuple[str, ...]:
    path = item.get("section_path") or []
    if isinstance(path, str):
        path = [part for part in path.split("/") if part.strip()]
    page = item.get("page_number") or item.get("page_num") or ""
    slide = item.get("slide_number") or ""
    sheet = item.get("sheet_name") or ""
    return (str(page), str(slide), str(sheet), *[str(part) for part in path])


def _split_oversized(item: dict[str, Any]) -> list[dict[str, Any]]:
    content = str(item.get("content") or "")
    if count_tokens(content) <= RETRIEVAL_MAX:
        return [dict(item)]
    windows = split_token_windows(content)
    if len(windows) <= 1:
        return [dict(item)]
    rows = []
    base = str(item.get("chunk_id") or "chunk")
    for index, piece in enumerate(windows, start=1):
        row = dict(item)
        row["content"] = piece
        row["chunk_id"] = f"{base}-r{index:02d}"
        row["source_id"] = row["chunk_id"]
        rows.append(row)
    return rows


def assign_parents(
    chunks: list[dict[str, Any]],
    document_name: str,
    document_id: str = "",
    *,
    split: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """给小块写 parent_id 和 index_text。父块正文只出现在父块记录里。"""
    prepared: list[dict[str, Any]] = []
    for item in chunks:
        prepared.extend(_split_oversized(item) if split else [dict(item)])
    parents: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0
    current_key: tuple[str, ...] | None = None

    def flush() -> None:
        nonlocal current_tokens
        if not current:
            return
        parent_id = f"{document_id or current[0].get('doc_id') or 'doc'}-p{len(parents) + 1:04d}"
        content = "\n".join(str(item.get("content") or "") for item in current).strip()
        unit_ids: list[str] = []
        for item in current:
            unit_id = str(item.get("unit_id") or "")
            if unit_id and unit_id not in unit_ids:
                unit_ids.append(unit_id)
            item["parent_id"] = parent_id
            item["index_text"] = index_text(document_name, item.get("section_path") or [], str(item.get("content") or ""))
            item.pop("parent_content", None)
        parents.append(
            {
                "parent_id": parent_id,
                "document_id": document_id or str(current[0].get("doc_id") or ""),
                "doc_id": document_id or str(current[0].get("doc_id") or ""),
                "section_path": list(current[0].get("section_path") or []),
                "source_unit_ids": unit_ids,
                "content": content,
                "token_count": count_tokens(content),
                "chunk_ids": [str(item.get("chunk_id") or "") for item in current],
                "kb_id": str(current[0].get("kb_id") or ""),
                "owner_id": str(current[0].get("owner_id") or current[0].get("user_id") or ""),
            }
        )
        current.clear()
        current_tokens = 0

    for item in prepared:
        key = _section_key(item)
        tokens = count_tokens(str(item.get("content") or ""))
        if current and key != current_key:
            flush()
        elif current and current_tokens + tokens > PARENT_MAX and current_tokens >= PARENT_MIN:
            flush()
        current.append(item)
        current_key = key
        current_tokens += tokens
        if current_tokens >= PARENT_MAX:
            flush()
            current_key = None
    flush()
    return prepared, parents
