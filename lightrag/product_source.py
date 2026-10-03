"""原文来源阅读。只返回位置和摘录，不返回本地路径。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

from lightrag.product_document_ir import DocumentIR, SourceUnit

_MIME = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "markdown": "text/markdown",
    "txt": "text/plain",
    "text": "text/plain",
    "html": "text/html",
}

_KIND = {
    "pdf": "pdf",
    "docx": "docx",
    "pptx": "pptx",
    "xlsx": "xlsx",
    "markdown": "markdown",
    "txt": "text",
    "text": "text",
    "html": "text",
}


def source_kind(name: str, source_type: str = "") -> str:
    kind = (source_type or "").lower()
    if kind in _KIND:
        return _KIND[kind]
    suffix = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return _KIND.get(suffix, "text")


def mime_of(name: str, source_type: str = "") -> str:
    kind = source_kind(name, source_type)
    if kind == "text" and name.lower().endswith(".md"):
        return _MIME["markdown"]
    return _MIME.get(kind, "application/octet-stream")


def _unit_view(unit: SourceUnit | None) -> dict[str, Any]:
    if unit is None:
        return {
            "unit_id": "",
            "page_number": None,
            "slide_number": None,
            "section_path": [],
            "sheet_name": None,
            "cell_range": None,
            "bbox": None,
            "line_start": None,
            "line_end": None,
        }
    return {
        "unit_id": unit.unit_id,
        "page_number": unit.page_number,
        "slide_number": unit.slide_number,
        "section_path": list(unit.section_path),
        "sheet_name": unit.sheet_name or None,
        "cell_range": unit.cell_range or None,
        "bbox": list(unit.bbox) if unit.bbox else None,
        "line_start": unit.line_start,
        "line_end": unit.line_end,
    }


def _parent_text(document: DocumentIR, unit: SourceUnit | None) -> str:
    if unit is None:
        return ""
    if unit.unit_type == "cell":
        table = next(
            (
                item
                for item in document.units
                if any(child.unit_id == unit.unit_id for child in item.children)
            ),
            None,
        )
        if table is not None:
            return "\n".join(
                child.content for child in table.children if child.content
            )[:1200]
    if unit.section_path:
        related = [
            item.content
            for item in document.walk()
            if item.section_path == unit.section_path and item.content
        ]
        return "\n".join(related)[:1200]
    return ""


def build_source_view(
    document: DocumentIR | None,
    *,
    doc_name: str,
    chunk_id: str = "",
    unit_id: str = "",
    excerpt: str = "",
    parent_context: str = "",
) -> dict[str, Any]:
    kind = source_kind(doc_name, document.source_type if document else "")
    unit = document.find_unit(unit_id) if document and unit_id else None
    if unit is None and document and not unit_id and chunk_id:
        unit = None
    source_text = unit.content if unit else ""
    parent = parent_context or (_parent_text(document, unit) if document else "")
    if parent == source_text:
        parent = ""
    preview_kind = (
        "pdf"
        if kind == "pdf"
        else ("text" if source_text or kind in {"text", "markdown"} else "none")
    )
    return {
        "document_id": document.document_id if document else "",
        "version_id": document.version_id if document else "",
        "doc_name": doc_name,
        "mime_type": mime_of(doc_name, document.source_type if document else ""),
        "source_kind": kind,
        "unit": _unit_view(unit),
        "matched_chunk": {
            "chunk_id": chunk_id,
            "excerpt": (excerpt or source_text)[:500],
        },
        "parent_context": parent[:1200],
        "source_text": source_text[:4000],
        "preview": {"available": kind == "pdf", "kind": preview_kind, "url": None},
        "layout": "structured" if kind in {"docx", "pptx", "xlsx"} else "original",
    }


def safe_download_name(name: str) -> str:
    base = Path(str(name or "").replace("\\", "/")).name
    cleaned = re.sub(r"[^A-Za-z0-9._\-\u4e00-\u9fff]", "_", base).strip("._")
    return (cleaned or "download")[:120]


def content_disposition(name: str, inline: bool = False) -> str:
    filename = safe_download_name(name)
    kind = "inline" if inline else "attachment"
    return f"{kind}; filename*=UTF-8''{quote(filename)}"


def resolve_owned_file(input_dir: Path, storage_key: str) -> Path | None:
    """只打开输入目录内、且由 doc_index 记录的文件。"""
    raw = Path(str(storage_key or ""))
    if not str(storage_key or "").strip() or any(part == ".." for part in raw.parts):
        return None
    root = Path(input_dir).resolve()
    candidate = raw if raw.is_absolute() else root / raw
    try:
        path = candidate.resolve()
    except OSError:
        return None
    if path != root and root not in path.parents:
        return None
    if not path.is_file():
        return None
    return path


def build_reading(
    document: DocumentIR | None, *, kind: str, file_text: str = ""
) -> dict[str, Any]:
    units = []
    if document is not None:
        for unit in document.walk():
            if (
                not unit.content
                and not unit.section_path
                and unit.page_number is None
                and unit.slide_number is None
            ):
                continue
            view = _unit_view(unit)
            view["unit_type"] = unit.unit_type
            view["content"] = (unit.content or "")[:1000]
            units.append(view)
    structured = kind in {"docx", "pptx", "xlsx"}
    return {
        "source_kind": kind,
        "text": file_text[:200000] if kind in {"text", "markdown"} else "",
        "units": units[:400],
        "layout": "structured" if structured else "original",
        "notice": "这是结构化阅读视图，不是原始 Office 排版。" if structured else "",
    }
