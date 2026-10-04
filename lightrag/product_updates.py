"""Knowledge versioning and incremental update primitives.

The current product still stores legacy document records in the shell JSON and
uses document-scoped JSON stores.  This module keeps the update rules
storage-independent so the same change set can later be applied to PostgreSQL,
Milvus and Neo4j.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Iterable


def _normalise(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def content_hash(payload: bytes | str) -> str:
    raw = payload if isinstance(payload, bytes) else str(payload).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def structure_hash(document: Any) -> str:
    """Hash structural anchors, not text, so content and layout changes differ."""
    rows: list[str] = []
    for unit in (
        document.walk() if document is not None and hasattr(document, "walk") else []
    ):
        rows.append(
            "|".join(
                (
                    str(getattr(unit, "unit_id", "")),
                    str(getattr(unit, "unit_type", "")),
                    "/".join(getattr(unit, "section_path", []) or []),
                    str(getattr(unit, "page_number", "") or ""),
                    str(getattr(unit, "slide_number", "") or ""),
                    str(getattr(unit, "sheet_name", "") or ""),
                    str(getattr(unit, "cell_range", "") or ""),
                )
            )
        )
    return content_hash("\n".join(rows))


def version_id(payload: bytes | str) -> str:
    return content_hash(payload)[:16]


def _anchor(chunk: dict[str, Any], index: int) -> str:
    refs = chunk.get("evidence_refs") or []
    if isinstance(refs, str):
        refs = []
    first = refs[0] if refs and isinstance(refs[0], dict) else {}
    location = "/".join(
        str(first.get(key) or "")
        for key in (
            "unit_id",
            "page_number",
            "slide_number",
            "section_path",
            "sheet_name",
            "cell_range",
        )
    )
    return location.strip("/") or f"index:{index + 1}"


def stable_chunk_id(document_id: str, chunk: dict[str, Any], index: int = 0) -> str:
    """Return an identity stable across versions for unchanged content."""
    anchor = _anchor(chunk, index)
    body = _normalise(chunk.get("content") or chunk.get("excerpt") or "")
    if anchor.startswith("index:"):
        # Legacy/plain-text chunks do not carry a structural location.  Their
        # content identity must survive insertion before the chunk.
        anchor = "content"
    digest = hashlib.sha256(
        f"{document_id}\n{anchor}\n{body}".encode("utf-8")
    ).hexdigest()[:24]
    return f"schunk-{digest}"


def decorate_chunks(
    document_id: str, chunks: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(chunks):
        row = dict(raw)
        body = _normalise(row.get("content") or "")
        row["content_hash"] = content_hash(body)
        row["structural_anchor"] = _anchor(row, index)
        row["stable_chunk_id"] = stable_chunk_id(document_id, row, index)
        rows.append(row)
    return rows


def _chunk_key(chunk: dict[str, Any]) -> str:
    return str(chunk.get("stable_chunk_id") or "")


def compare_chunks(
    old_chunks: Iterable[dict[str, Any]],
    new_chunks: Iterable[dict[str, Any]],
    *,
    old_document_id: str = "old",
    new_document_id: str = "new",
) -> list[dict[str, Any]]:
    """Match chunks by stable identity, then use similarity only as fallback."""
    old = decorate_chunks(old_document_id, old_chunks)
    new = decorate_chunks(new_document_id, new_chunks)

    # When comparing two versions, the logical document identity is not the
    # physical doc id.  Content plus structural anchor is the cross-version key.
    def cross_key(item: dict[str, Any]) -> str:
        return f"{item.get('structural_anchor')}|{item.get('content_hash')}"

    old_by_key = {cross_key(item): (index, item) for index, item in enumerate(old)}
    new_by_key = {cross_key(item): (index, item) for index, item in enumerate(new)}
    old_by_hash: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    new_by_hash: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, item in enumerate(old):
        old_by_hash.setdefault(str(item.get("content_hash") or ""), []).append(
            (index, item)
        )
    for index, item in enumerate(new):
        new_by_hash.setdefault(str(item.get("content_hash") or ""), []).append(
            (index, item)
        )
    matched_old: set[int] = set()
    matched_new: set[int] = set()
    rows: list[dict[str, Any]] = []

    for key, (new_index, new_item) in new_by_key.items():
        if key not in old_by_key:
            continue
        old_index, old_item = old_by_key[key]
        matched_old.add(old_index)
        matched_new.add(new_index)
        rows.append(
            {
                "type": "unchanged",
                "old_index": old_index + 1,
                "new_index": new_index + 1,
                "old": old_item,
                "new": new_item,
                "stable_chunk_id": new_item["stable_chunk_id"],
            }
        )

    # A paragraph can move when text is inserted before it.  A unique content
    # hash is stronger evidence of no content change than its numeric position.
    for digest, new_matches in new_by_hash.items():
        old_matches = old_by_hash.get(digest) or []
        if not digest or len(new_matches) != 1 or len(old_matches) != 1:
            continue
        new_index, new_item = new_matches[0]
        old_index, old_item = old_matches[0]
        if new_index in matched_new or old_index in matched_old:
            continue
        matched_old.add(old_index)
        matched_new.add(new_index)
        rows.append(
            {
                "type": "unchanged",
                "old_index": old_index + 1,
                "new_index": new_index + 1,
                "old": old_item,
                "new": new_item,
                "stable_chunk_id": new_item["stable_chunk_id"],
            }
        )

    # Pair remaining chunks at the same structural position when possible.
    for new_index, new_item in enumerate(new):
        if new_index in matched_new:
            continue
        candidate: tuple[float, int] | None = None
        for old_index, old_item in enumerate(old):
            if old_index in matched_old:
                continue
            score = SequenceMatcher(
                None,
                _normalise(old_item.get("content")),
                _normalise(new_item.get("content")),
            ).ratio()
            if old_item.get("structural_anchor") == new_item.get("structural_anchor"):
                score += 0.25
            if candidate is None or score > candidate[0]:
                candidate = (score, old_index)
        if candidate and candidate[0] >= 0.48:
            score, old_index = candidate
            old_item = old[old_index]
            matched_old.add(old_index)
            matched_new.add(new_index)
            moved = old_item.get("structural_anchor") != new_item.get(
                "structural_anchor"
            )
            rows.append(
                {
                    "type": "moved" if moved and score >= 0.85 else "modified",
                    "old_index": old_index + 1,
                    "new_index": new_index + 1,
                    "old": old_item,
                    "new": new_item,
                    "similarity": round(score, 6),
                    "stable_chunk_id": new_item["stable_chunk_id"],
                }
            )
        else:
            matched_new.add(new_index)
            rows.append(
                {
                    "type": "added",
                    "old_index": None,
                    "new_index": new_index + 1,
                    "old": None,
                    "new": new_item,
                    "stable_chunk_id": new_item["stable_chunk_id"],
                }
            )

    for old_index, old_item in enumerate(old):
        if old_index not in matched_old:
            rows.append(
                {
                    "type": "deleted",
                    "old_index": old_index + 1,
                    "new_index": None,
                    "old": old_item,
                    "new": None,
                    "stable_chunk_id": old_item["stable_chunk_id"],
                }
            )
    order = {"added": 0, "modified": 1, "moved": 2, "deleted": 3, "unchanged": 4}
    return sorted(
        rows,
        key=lambda row: (
            row.get("new_index") or row.get("old_index") or 0,
            order[row["type"]],
        ),
    )


def current_time() -> str:
    return datetime.now(timezone.utc).isoformat()


def fact_state(
    *,
    fact_id: str,
    source_version_id: str,
    valid_at: str | None = None,
    invalid_at: str | None = None,
    superseded_by: str = "",
    status: str = "active",
) -> dict[str, Any]:
    return {
        "fact_id": fact_id,
        "source_version_id": source_version_id,
        "valid_at": valid_at or current_time(),
        "invalid_at": invalid_at or "",
        "superseded_by": superseded_by,
        "status": status
        if status in {"candidate", "active", "superseded", "retracted", "expired"}
        else "candidate",
    }


def change_summary(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    summary = {
        name: 0 for name in ("unchanged", "added", "modified", "deleted", "moved")
    }
    for row in rows:
        kind = str(row.get("type") or "")
        if kind in summary:
            summary[kind] += 1
    summary["changed"] = sum(
        summary[name] for name in ("added", "modified", "deleted", "moved")
    )
    return summary
