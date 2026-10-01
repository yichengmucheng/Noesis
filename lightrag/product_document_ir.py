"""文档中间表示。

解析器以后把 PDF、Word、PPT、Excel 写成这套结构。
检索和引用只保存 EvidenceRef，用 document_id、version_id、chunk_id、unit_id 回到原文位置。
拿不到坐标时 bbox 保持 null，不猜测页码。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

UNIT_TYPES = {
    "page",
    "slide",
    "section",
    "sheet",
    "text",
    "table",
    "cell",
    "image",
    "diagram",
}

SOURCE_TYPES = {"pdf", "docx", "pptx", "xlsx", "markdown", "txt", "html", "image", "audio"}


def checksum_of(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _text(value: Any) -> str:
    return str(value or "").strip()


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    number = int(value)
    if number < 1:
        raise ValueError("页码、幻灯片序号必须从 1 开始")
    return number


def _path(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.split("/") if part.strip()]
    elif isinstance(value, list):
        parts = [str(part).strip() for part in value if str(part).strip()]
    else:
        raise ValueError("section_path 必须是字符串或列表")
    return parts


def _bbox(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("bbox 必须是四个数字或 null")
    return [float(item) for item in value]


@dataclass
class SourceUnit:
    unit_id: str
    unit_type: str
    content: str = ""
    page_number: int | None = None
    slide_number: int | None = None
    section_path: list[str] = field(default_factory=list)
    sheet_name: str = ""
    cell_range: str = ""
    bbox: list[float] | None = None
    line_start: int | None = None
    line_end: int | None = None
    children: list["SourceUnit"] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.unit_id = _text(self.unit_id)
        self.unit_type = _text(self.unit_type)
        if not self.unit_id:
            raise ValueError("unit_id 不能为空")
        if self.unit_type not in UNIT_TYPES:
            raise ValueError(f"不支持的单元类型：{self.unit_type}")
        self.page_number = _optional_int(self.page_number)
        self.slide_number = _optional_int(self.slide_number)
        self.section_path = _path(self.section_path)
        self.sheet_name = _text(self.sheet_name)
        self.cell_range = _text(self.cell_range)
        self.bbox = _bbox(self.bbox)
        self.line_start = _optional_int(self.line_start)
        self.line_end = _optional_int(self.line_end)

    def walk(self) -> list["SourceUnit"]:
        found = [self]
        for child in self.children:
            found.extend(child.walk())
        return found

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "unit_type": self.unit_type,
            "page_number": self.page_number,
            "slide_number": self.slide_number,
            "section_path": list(self.section_path),
            "sheet_name": self.sheet_name,
            "cell_range": self.cell_range,
            "content": self.content,
            "bbox": self.bbox,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SourceUnit":
        children = [cls.from_dict(item) for item in raw.get("children") or [] if isinstance(item, dict)]
        return cls(
            unit_id=raw.get("unit_id") or "",
            unit_type=raw.get("unit_type") or "",
            content=str(raw.get("content") or ""),
            page_number=raw.get("page_number"),
            slide_number=raw.get("slide_number"),
            section_path=raw.get("section_path") or [],
            sheet_name=raw.get("sheet_name") or "",
            cell_range=raw.get("cell_range") or "",
            bbox=raw.get("bbox"),
            line_start=raw.get("line_start"),
            line_end=raw.get("line_end"),
            children=children,
        )


@dataclass
class EvidenceRef:
    evidence_id: str
    document_id: str
    version_id: str
    chunk_id: str
    unit_id: str
    page_number: int | None = None
    slide_number: int | None = None
    section_path: list[str] = field(default_factory=list)
    sheet_name: str = ""
    cell_range: str = ""
    bbox: list[float] | None = None
    excerpt: str = ""

    def __post_init__(self) -> None:
        for name in ("evidence_id", "document_id", "version_id", "chunk_id", "unit_id"):
            if not _text(getattr(self, name)):
                raise ValueError(f"{name} 不能为空")
            setattr(self, name, _text(getattr(self, name)))
        self.page_number = _optional_int(self.page_number)
        self.slide_number = _optional_int(self.slide_number)
        self.section_path = _path(self.section_path)
        self.sheet_name = _text(self.sheet_name)
        self.cell_range = _text(self.cell_range)
        self.bbox = _bbox(self.bbox)
        self.excerpt = str(self.excerpt or "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "document_id": self.document_id,
            "version_id": self.version_id,
            "chunk_id": self.chunk_id,
            "unit_id": self.unit_id,
            "page_number": self.page_number,
            "slide_number": self.slide_number,
            "section_path": list(self.section_path),
            "sheet_name": self.sheet_name,
            "cell_range": self.cell_range,
            "bbox": self.bbox,
            "excerpt": self.excerpt,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EvidenceRef":
        return cls(
            evidence_id=raw.get("evidence_id") or "",
            document_id=raw.get("document_id") or "",
            version_id=raw.get("version_id") or "",
            chunk_id=raw.get("chunk_id") or "",
            unit_id=raw.get("unit_id") or "",
            page_number=raw.get("page_number"),
            slide_number=raw.get("slide_number"),
            section_path=raw.get("section_path") or [],
            sheet_name=raw.get("sheet_name") or "",
            cell_range=raw.get("cell_range") or "",
            bbox=raw.get("bbox"),
            excerpt=raw.get("excerpt") or "",
        )


@dataclass
class DocumentIR:
    document_id: str
    version_id: str
    source_name: str
    source_type: str
    checksum: str
    units: list[SourceUnit] = field(default_factory=list)
    parse_summary: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.document_id = _text(self.document_id)
        self.version_id = _text(self.version_id)
        self.source_name = _text(self.source_name)
        self.source_type = _text(self.source_type).lower()
        self.checksum = _text(self.checksum)
        if not self.document_id or not self.version_id or not self.checksum:
            raise ValueError("document_id、version_id、checksum 不能为空")
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(f"不支持的来源类型：{self.source_type}")
        if not isinstance(self.parse_summary, dict):
            raise ValueError("parse_summary 必须是对象")
        seen: set[str] = set()
        for unit in self.walk():
            if unit.unit_id in seen:
                raise ValueError(f"unit_id 重复：{unit.unit_id}")
            seen.add(unit.unit_id)

    def walk(self) -> list[SourceUnit]:
        found: list[SourceUnit] = []
        for unit in self.units:
            found.extend(unit.walk())
        return found

    def find_unit(self, unit_id: str) -> SourceUnit | None:
        target = _text(unit_id)
        for unit in self.walk():
            if unit.unit_id == target:
                return unit
        return None

    def evidence(self, chunk_id: str, unit_id: str, excerpt: str = "") -> EvidenceRef:
        unit = self.find_unit(unit_id)
        if unit is None:
            raise ValueError("引用指向了文档中不存在的单元")
        digest = hashlib.sha256(f"{self.document_id}\n{self.version_id}\n{chunk_id}\n{unit_id}".encode("utf-8")).hexdigest()[:16]
        return EvidenceRef(
            evidence_id=f"ev-{digest}",
            document_id=self.document_id,
            version_id=self.version_id,
            chunk_id=_text(chunk_id),
            unit_id=unit.unit_id,
            page_number=unit.page_number,
            slide_number=unit.slide_number,
            section_path=list(unit.section_path),
            sheet_name=unit.sheet_name,
            cell_range=unit.cell_range,
            bbox=list(unit.bbox) if unit.bbox is not None else None,
            excerpt=excerpt or unit.content[:240],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "version_id": self.version_id,
            "source_name": self.source_name,
            "source_type": self.source_type,
            "checksum": self.checksum,
            "units": [unit.to_dict() for unit in self.units],
            "parse_summary": self.parse_summary,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DocumentIR":
        return cls(
            document_id=raw.get("document_id") or "",
            version_id=raw.get("version_id") or "",
            source_name=raw.get("source_name") or "",
            source_type=raw.get("source_type") or "",
            checksum=raw.get("checksum") or "",
            units=[SourceUnit.from_dict(item) for item in raw.get("units") or [] if isinstance(item, dict)],
            parse_summary=raw.get("parse_summary") if isinstance(raw.get("parse_summary"), dict) else {},
        )

    @classmethod
    def from_json(cls, raw: str) -> "DocumentIR":
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("文档中间表示必须是对象")
        return cls.from_dict(payload)


def public_location(source: dict[str, Any] | None) -> dict[str, Any]:
    raw = source or {}
    page = raw.get("page_number")
    if page in (None, ""):
        page = raw.get("page_num")
    if page in (None, ""):
        page = None
    else:
        page = _optional_int(page)
    section = raw.get("section_path") or []
    if isinstance(section, str):
        section = _path(section)
    elif isinstance(section, list):
        section = [str(part).strip() for part in section if str(part).strip()]
    else:
        section = []
    slide = raw.get("slide_number")
    slide = None if slide in (None, "") else _optional_int(slide)
    return {
        "evidence_id": _text(raw.get("evidence_id")),
        "document_id": _text(raw.get("document_id") or raw.get("doc_id")),
        "version_id": _text(raw.get("version_id")),
        "unit_id": _text(raw.get("unit_id")),
        "page_number": page,
        "page_num": page,
        "slide_number": slide,
        "section_path": section,
        "sheet_name": _text(raw.get("sheet_name")),
        "cell_range": _text(raw.get("cell_range")),
        "bbox": raw.get("bbox") if isinstance(raw.get("bbox"), list) else None,
        "line_start": None if raw.get("line_start") in (None, "") else _optional_int(raw.get("line_start")),
        "line_end": None if raw.get("line_end") in (None, "") else _optional_int(raw.get("line_end")),
        "excerpt": str(raw.get("excerpt") or ""),
    }


def apply_location(row: dict[str, Any], source: dict[str, Any] | None) -> dict[str, Any]:
    for key, value in public_location(source).items():
        if value in (None, "", []):
            continue
        row[key] = value
    raw = source or {}
    refs = raw.get("evidence_refs")
    if isinstance(refs, list) and refs:
        row["evidence_refs"] = refs
        row["evidence_ids"] = list(raw.get("evidence_ids") or [item.get("evidence_id") for item in refs if isinstance(item, dict) and item.get("evidence_id")])
    elif raw.get("evidence_ids"):
        row["evidence_ids"] = list(raw["evidence_ids"])
    return row


def location_label(source: dict[str, Any] | None) -> str:
    row = public_location(source)
    parts: list[str] = []
    if row["page_number"]:
        parts.append(f"第{row['page_number']}页")
    if row["slide_number"]:
        parts.append(f"第{row['slide_number']}张幻灯片")
    if row["section_path"]:
        parts.append(" / ".join(row["section_path"]))
    if row["sheet_name"]:
        parts.append(row["sheet_name"])
    if row["cell_range"]:
        parts.append(row["cell_range"])
    return " · ".join(parts)
