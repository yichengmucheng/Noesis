"""把上传文件解析成带位置的文档中间表示，并生成可入库的切块。

正文仍然进现有的 JSON 存储。页、节、幻灯片、工作表和单元格写在切块上。
拿不到坐标时 bbox 保持空，不为旧数据猜测页码。
"""

from __future__ import annotations

import hashlib
import re
from io import BytesIO
from pathlib import Path
from typing import Any

from lightrag.product_document_ir import (
    DocumentIR,
    SourceUnit,
    checksum_of,
)
from lightrag.product_ids import chunk_record
from lightrag.utils import logger

_SUFFIX = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".pptx": "pptx",
    ".xlsx": "xlsx",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "txt",
    ".html": "html",
    ".htm": "html",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".gif": "image",
    ".mp3": "audio",
    ".wav": "audio",
    ".m4a": "audio",
}


class ParseError(ValueError):
    pass


def source_type_of(name: str) -> str:
    return _SUFFIX.get(Path(name).suffix.lower(), "")


def leaf_units(document: DocumentIR) -> list[SourceUnit]:
    rows: list[SourceUnit] = []
    for unit in document.walk():
        if not unit.content.strip():
            continue
        if any(child.content.strip() for child in unit.children):
            continue
        rows.append(unit)
    return rows


def plain_text(document: DocumentIR) -> str:
    return "\n\n".join(unit.content.strip() for unit in leaf_units(document))


def parse_bytes(name: str, payload: bytes, document_id: str) -> DocumentIR:
    source_type = source_type_of(name)
    if not source_type:
        raise ParseError("不支持的文件类型")
    if not payload:
        raise ParseError("文件内容为空")
    checksum = checksum_of(payload)
    document_id = str(document_id or "").strip() or checksum[:16]
    if source_type == "pdf":
        units, summary = _pdf_units(payload)
    elif source_type == "docx":
        units, summary = _docx_units(payload)
    elif source_type == "pptx":
        units, summary = _pptx_units(payload)
    elif source_type == "xlsx":
        units, summary = _xlsx_units(payload)
    elif source_type == "markdown":
        units, summary = _markdown_units(_decode(payload))
    elif source_type == "html":
        units, summary = _markdown_units(_html_text(payload))
        summary["from"] = "html"
    elif source_type == "txt":
        units, summary = _text_units(_decode(payload))
    else:
        units, summary = (
            [],
            {"extracted": False, "reason": "本阶段不解析图片和语音正文"},
        )
    return DocumentIR(
        document_id=document_id,
        version_id=checksum[:16],
        source_name=Path(name).name,
        source_type=source_type,
        checksum=checksum,
        units=units,
        parse_summary=summary,
    )


def chunks_from_document(
    document: DocumentIR,
    user_id: str,
    kb_id: str,
    doc_id: str,
    chunk_size: int = 512,
    chunk_overlap: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for text, units in _structure_blocks(document):
        for piece in _split_tokens(text, chunk_size, chunk_overlap):
            covered = [
                unit
                for unit in units
                if unit.content.strip()
                and (unit.content.strip() in piece or piece in unit.content.strip())
            ]
            if not covered:
                covered = list(units[:1])
            record = chunk_record(user_id, kb_id, doc_id, len(rows) + 1, piece)
            _attach_evidence(record, document, covered, piece)
            rows.append(record)
    return rows


def save_ir(working_dir: str | Path, document: DocumentIR) -> None:
    from lightrag.product_storage import _write_json

    _write_json(
        Path(working_dir) / "document_ir" / f"{document.document_id}.json",
        document.to_dict(),
    )


def remember_document(
    working_dir: str | Path, source_path: str | Path, document: DocumentIR
) -> None:
    from lightrag.product_storage import _write_json

    _write_json(_pending_path(working_dir, source_path), document.to_dict())


def claim_document(
    working_dir: str | Path, source_path: str | Path, doc_id: str
) -> DocumentIR | None:
    """只读取这条文件路径对应的一份中间表示，并改记为当前 doc_id。"""
    from lightrag.product_storage import _read_json

    pending = _pending_path(working_dir, source_path)
    payload = _read_json(pending)
    if not isinstance(payload, dict):
        return load_document(working_dir, doc_id)
    payload["document_id"] = doc_id
    try:
        document = DocumentIR.from_dict(payload)
    except ValueError:
        return None
    save_ir(working_dir, document)
    pending.unlink(missing_ok=True)
    backup = pending.with_name(pending.name + ".bak")
    if backup.exists():
        backup.unlink()
    return document


def load_document(working_dir: str | Path, doc_id: str) -> DocumentIR | None:
    from lightrag.product_storage import _read_json

    payload = _read_json(Path(working_dir) / "document_ir" / f"{doc_id}.json")
    if not isinstance(payload, dict):
        return None
    try:
        return DocumentIR.from_dict(payload)
    except ValueError:
        return None


def prefer_located_text(
    working_dir: str | Path,
    name: str,
    payload: bytes | None,
    content: str,
    replace: bool,
    source_path: str | Path | None = None,
) -> str:
    if not payload:
        return content
    try:
        document = parse_bytes(name, payload, document_id=checksum_of(payload)[:16])
        if source_path is not None:
            remember_document(working_dir, source_path, document)
        else:
            save_ir(working_dir, document)
    except Exception as exc:
        logger.warning("位置解析未写入：%s", exc)
        return content
    if not replace:
        return content
    located = plain_text(document).strip()
    return located or content


def bind_pipeline_chunks(
    chunks: dict[str, Any], file_path: str, doc_id: str, working_dir: str | Path
) -> None:
    document = claim_document(working_dir, file_path, doc_id)
    if document is None:
        return
    bind_document_chunks(chunks, document)


def bind_document_chunks(chunks: dict[str, Any], document: DocumentIR | None) -> None:
    if document is None:
        return
    leaves = leaf_units(document)
    for key, chunk in chunks.items():
        if not isinstance(chunk, dict) or chunk.get("evidence_ids"):
            continue
        content = str(chunk.get("content") or "").strip()
        if not content:
            continue
        matched = [
            unit
            for unit in leaves
            if unit.content.strip()
            and len(unit.content.strip()) >= 2
            and (unit.content.strip() in content or content in unit.content.strip())
        ]
        if not matched:
            continue
        chunk_id = str(chunk.get("chunk_id") or key)
        if not chunk.get("chunk_id"):
            chunk["chunk_id"] = chunk_id
        _attach_evidence(chunk, document, matched, content)


def _pending_path(working_dir: str | Path, source_path: str | Path) -> Path:
    digest = hashlib.sha256(str(Path(source_path)).encode("utf-8")).hexdigest()[:24]
    return Path(working_dir) / "document_ir" / "pending" / f"{digest}.json"


def _attach_evidence(
    record: dict[str, Any], document: DocumentIR, units: list[SourceUnit], excerpt: str
) -> None:
    refs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for unit in units:
        if unit.unit_id in seen:
            continue
        seen.add(unit.unit_id)
        evidence = document.evidence(
            str(record["chunk_id"]), unit.unit_id, excerpt=excerpt[:240]
        )
        data = evidence.to_dict()
        refs.append(
            {
                "evidence_id": data["evidence_id"],
                "unit_id": data["unit_id"],
                "page_number": data["page_number"],
                "slide_number": data["slide_number"],
                "section_path": data["section_path"],
                "sheet_name": data["sheet_name"],
                "cell_range": data["cell_range"],
                "excerpt": data["excerpt"],
            }
        )
    record["evidence_ids"] = [item["evidence_id"] for item in refs]
    record["evidence_refs"] = refs
    if not refs:
        return
    first = refs[0]
    record.update(
        {
            "evidence_id": first["evidence_id"],
            "document_id": document.document_id,
            "version_id": document.version_id,
            "unit_id": first["unit_id"],
            "page_number": first["page_number"],
            "page_num": first["page_number"],
            "slide_number": first["slide_number"],
            "section_path": first["section_path"],
            "sheet_name": first["sheet_name"],
            "cell_range": first["cell_range"],
            "excerpt": first["excerpt"],
        }
    )


def _structure_blocks(document: DocumentIR) -> list[tuple[str, list[SourceUnit]]]:
    if document.source_type == "pdf":
        return [
            (unit.content.strip(), [unit])
            for unit in document.units
            if unit.unit_type == "page" and unit.content.strip()
        ]
    if document.source_type == "pptx":
        return [
            (unit.content.strip(), [unit])
            for unit in document.units
            if unit.unit_type == "slide" and unit.content.strip()
        ]
    if document.source_type == "xlsx":
        return _sheet_blocks(document)
    if document.source_type == "docx":
        return _docx_blocks(document)
    return [
        (unit.content.strip(), [unit])
        for unit in leaf_units(document)
        if unit.content.strip()
    ]


def _docx_blocks(document: DocumentIR) -> list[tuple[str, list[SourceUnit]]]:
    blocks: list[tuple[str, list[SourceUnit]]] = []
    current: list[SourceUnit] = []

    def flush() -> None:
        texts = [unit.content.strip() for unit in current if unit.content.strip()]
        if texts:
            blocks.append(("\n\n".join(texts), list(current)))
        current.clear()

    for unit in document.units:
        if unit.unit_type == "table":
            flush()
            cells = [
                child
                for child in unit.walk()
                if child.unit_type == "cell" and child.content.strip()
            ]
            body = "\n".join(child.content.strip() for child in cells)
            if body:
                blocks.append((body, cells or [unit]))
            continue
        if unit.unit_type == "section" and current:
            flush()
        current.append(unit)
    flush()
    return blocks


def _sheet_blocks(document: DocumentIR) -> list[tuple[str, list[SourceUnit]]]:
    blocks: list[tuple[str, list[SourceUnit]]] = []
    for sheet in document.units:
        if sheet.unit_type != "sheet":
            continue
        cells = [
            child
            for child in sheet.walk()
            if child.unit_type == "cell" and child.content.strip()
        ]
        grouped: dict[int, list[SourceUnit]] = {}
        for cell in cells:
            matched = re.search(r"(\d+)$", cell.cell_range or "")
            if not matched:
                continue
            grouped.setdefault(int(matched.group(1)), []).append(cell)
        if not grouped:
            continue
        header_row = min(grouped)
        headers = {_column_of(cell.cell_range): cell for cell in grouped[header_row]}
        data_rows = [row for row in sorted(grouped) if row != header_row]
        if not data_rows:
            blocks.append(
                (
                    "\n".join(cell.content.strip() for cell in grouped[header_row]),
                    grouped[header_row],
                )
            )
            continue
        for row in data_rows:
            lines = [f"工作表 {sheet.sheet_name}".strip()]
            used = []
            for cell in grouped[row]:
                header = headers.get(_column_of(cell.cell_range))
                label = (
                    header.content.strip() if header is not None else cell.cell_range
                )
                lines.append(f"{label}: {cell.content.strip()}")
                used.append(cell)
                if header is not None:
                    used.append(header)
            blocks.append(("\n".join(lines), used))
    return blocks


def _column_of(cell_range: str) -> str:
    matched = re.match(r"([A-Za-z]+)", cell_range or "")
    return matched.group(1).upper() if matched else cell_range


def _split_tokens(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    body = text.strip()
    if not body:
        return []
    size = max(1, int(chunk_size or 512))
    overlap = max(0, int(chunk_overlap or 0))
    if overlap >= size:
        overlap = size - 1
    tokenizer = _tokenizer()
    tokens = tokenizer.encode(body)
    if len(tokens) <= size:
        return [body]
    pieces: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + size)
        piece = tokenizer.decode(tokens[start:end]).strip()
        if piece:
            pieces.append(piece)
        if end >= len(tokens):
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return pieces


_TOKENIZER = None


def _tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from lightrag.utils import TiktokenTokenizer

        _TOKENIZER = TiktokenTokenizer()
    return _TOKENIZER


def _decode(payload: bytes) -> str:
    return payload.decode("utf-8", errors="replace")


def _html_text(payload: bytes) -> str:
    text = _decode(payload)
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", "\n", text)
    return text


def _text_units(text: str) -> tuple[list[SourceUnit], dict[str, Any]]:
    units: list[SourceUnit] = []
    buffer: list[str] = []
    start: int | None = None
    index = 0
    lines = text.splitlines() or [text]

    def flush(end: int) -> None:
        nonlocal index, start
        body = "\n".join(buffer).strip()
        buffer.clear()
        if not body or start is None or end < start:
            start = None
            return
        index += 1
        units.append(
            SourceUnit(
                unit_id=f"t{index:04d}",
                unit_type="text",
                content=body,
                line_start=start,
                line_end=end,
            )
        )
        start = None

    for number, line in enumerate(lines, start=1):
        if not line.strip():
            flush(number - 1)
            continue
        if start is None:
            start = number
        buffer.append(line)
    flush(len(lines))
    return units, {"blocks": len(units)}


def _markdown_units(text: str) -> tuple[list[SourceUnit], dict[str, Any]]:
    units: list[SourceUnit] = []
    stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    counter = 0
    start: int | None = None
    end = 0

    def path() -> list[str]:
        return [name for _, name in stack]

    def take(kind: str) -> str:
        nonlocal counter
        counter += 1
        return f"{kind}{counter:04d}"

    def flush() -> None:
        nonlocal counter, start
        body = "\n".join(buffer).strip()
        buffer.clear()
        if not body or start is None:
            start = None
            return
        units.append(
            SourceUnit(
                unit_id=take("t"),
                unit_type="text",
                section_path=path(),
                content=body,
                line_start=start,
                line_end=end,
            )
        )
        start = None

    for line_no, line in enumerate(text.splitlines(), start=1):
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading:
            flush()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            stack = [
                (item_level, item_name)
                for item_level, item_name in stack
                if item_level < level
            ]
            stack.append((level, title))
            units.append(
                SourceUnit(
                    unit_id=take("s"),
                    unit_type="section",
                    section_path=path(),
                    content=title,
                    line_start=line_no,
                    line_end=line_no,
                )
            )
            continue
        if not line.strip():
            flush()
            continue
        if start is None:
            start = line_no
        end = line_no
        buffer.append(line)
    flush()
    return units, {"blocks": len(units)}


def _pdf_units(payload: bytes) -> tuple[list[SourceUnit], dict[str, Any]]:
    try:
        try:
            from pypdf import PdfReader
        except ImportError:
            from PyPDF2 import PdfReader
        reader = PdfReader(BytesIO(payload))
    except Exception as exc:
        raise ParseError(f"PDF 解析失败：{exc}") from exc
    units: list[SourceUnit] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""
        if not text:
            continue
        units.append(
            SourceUnit(
                unit_id=f"p{index}", unit_type="page", page_number=index, content=text
            )
        )
    return units, {
        "page_count": len(reader.pages),
        "located_pages": len(units),
        "bbox": None,
    }


def _docx_units(payload: bytes) -> tuple[list[SourceUnit], dict[str, Any]]:
    try:
        from docx import Document
        from docx.oxml.ns import qn
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except Exception as exc:
        raise ParseError(f"Word 解析失败：{exc}") from exc
    try:
        doc = Document(BytesIO(payload))
    except Exception as exc:
        raise ParseError(f"Word 解析失败：{exc}") from exc
    units: list[SourceUnit] = []
    stack: list[tuple[int, str]] = []
    counter = 0

    def path() -> list[str]:
        return [name for _, name in stack]

    def take(kind: str) -> str:
        nonlocal counter
        counter += 1
        return f"{kind}{counter:04d}"

    def level_of(style_name: str) -> int:
        name = (style_name or "").strip().lower()
        for number in range(1, 7):
            if name in (f"heading {number}", f"标题 {number}", f"标题{number}"):
                return number
        if name in ("title", "标题"):
            return 1
        return 0

    def blocks(parent):
        for child in parent.element.body.iterchildren():
            if child.tag == qn("w:p"):
                yield Paragraph(child, parent)
            elif child.tag == qn("w:tbl"):
                yield Table(child, parent)

    for block in blocks(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            level = level_of(block.style.name if block.style is not None else "")
            if level:
                stack = [
                    (item_level, item_name)
                    for item_level, item_name in stack
                    if item_level < level
                ]
                stack.append((level, text))
                units.append(
                    SourceUnit(
                        unit_id=take("s"),
                        unit_type="section",
                        section_path=path(),
                        content=text,
                    )
                )
            else:
                units.append(
                    SourceUnit(
                        unit_id=take("t"),
                        unit_type="text",
                        section_path=path(),
                        content=text,
                    )
                )
            continue
        cells: list[SourceUnit] = []
        table_id = take("tbl")
        for row_index, row in enumerate(block.rows, start=1):
            seen = set()
            for col_index, cell in enumerate(row.cells, start=1):
                value = cell.text.strip().replace("\n", " ")
                if not value or value in seen:
                    continue
                seen.add(value)
                cells.append(
                    SourceUnit(
                        unit_id=f"{table_id}-r{row_index}c{col_index}",
                        unit_type="cell",
                        section_path=path(),
                        cell_range=f"{_column_name(col_index)}{row_index}",
                        content=value,
                    )
                )
        if cells:
            units.append(
                SourceUnit(
                    unit_id=table_id,
                    unit_type="table",
                    section_path=path(),
                    children=cells,
                )
            )
    return units, {"blocks": len(units), "bbox": None}


def _pptx_units(payload: bytes) -> tuple[list[SourceUnit], dict[str, Any]]:
    try:
        from pptx import Presentation
    except Exception as exc:
        raise ParseError(f"PPT 解析失败：{exc}") from exc
    try:
        presentation = Presentation(BytesIO(payload))
    except Exception as exc:
        raise ParseError(f"PPT 解析失败：{exc}") from exc
    units: list[SourceUnit] = []
    for index, slide in enumerate(presentation.slides, start=1):
        lines = []
        for shape in slide.shapes:
            text = getattr(shape, "text", "") or ""
            text = text.strip()
            if text:
                lines.append(text)
        if not lines:
            continue
        units.append(
            SourceUnit(
                unit_id=f"slide{index}",
                unit_type="slide",
                slide_number=index,
                content="\n".join(lines),
            )
        )
    return units, {
        "slide_count": len(presentation.slides),
        "located_slides": len(units),
        "bbox": None,
    }


def _xlsx_units(payload: bytes) -> tuple[list[SourceUnit], dict[str, Any]]:
    try:
        from openpyxl import load_workbook
    except Exception as exc:
        raise ParseError(f"Excel 解析失败：{exc}") from exc
    try:
        book = load_workbook(BytesIO(payload), read_only=True, data_only=True)
    except Exception as exc:
        raise ParseError(f"Excel 解析失败：{exc}") from exc
    units: list[SourceUnit] = []
    truncated = False
    try:
        for sheet_index, sheet in enumerate(book.worksheets, start=1):
            cells: list[SourceUnit] = []
            sheet_id = f"sheet{sheet_index}"
            for row_index, row in enumerate(
                sheet.iter_rows(max_row=200, max_col=40), start=1
            ):
                if row_index >= 200:
                    truncated = True
                for cell in row:
                    if cell.value is None:
                        continue
                    text = str(cell.value).strip()
                    if not text:
                        continue
                    coordinate = str(cell.coordinate or "")
                    if not coordinate:
                        continue
                    cells.append(
                        SourceUnit(
                            unit_id=f"{sheet_id}-{coordinate}",
                            unit_type="cell",
                            sheet_name=str(sheet.title),
                            cell_range=coordinate,
                            content=text[:2000],
                        )
                    )
            if cells:
                units.append(
                    SourceUnit(
                        unit_id=sheet_id,
                        unit_type="sheet",
                        sheet_name=str(sheet.title),
                        children=cells,
                    )
                )
    finally:
        book.close()
    return units, {"sheets": len(units), "truncated": truncated, "bbox": None}


def _column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name
