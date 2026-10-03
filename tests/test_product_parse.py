import io

import pytest

from lightrag.answer_pipeline import citations_of
from lightrag.product_parse import (
    bind_document_chunks,
    chunks_from_document,
    parse_bytes,
)


def _pdf(text: str) -> bytes:
    stream = f"BT /F1 24 Tf 72 120 Td ({text}) Tj ET\n".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
        + stream
        + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    header = b"%PDF-1.4\n"
    body = bytearray(header)
    offsets = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(body))
        body += f"{index} 0 obj\n".encode("ascii") + obj + b"\nendobj\n"
    xref_at = len(body)
    xref = [f"xref\n0 {len(objects) + 1}\n", "0000000000 65535 f \n"]
    xref.extend(f"{offset:010d} 00000 n \n" for offset in offsets)
    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
    return bytes(body) + "".join(xref).encode("ascii") + trailer.encode("ascii")


def test_markdown_txt_and_citations_keep_location(tmp_path):
    markdown = parse_bytes(
        "手册.md",
        "冷却系统\n\n# 冷却系统\n\n节温器打不开\n".encode("utf-8"),
        document_id="doc-1",
    )
    text_unit = next(unit for unit in markdown.walk() if unit.content == "节温器打不开")
    assert text_unit.section_path == ["冷却系统"]
    assert text_unit.bbox is None

    note = parse_bytes(
        "note.txt", "第一段\n\n第二段".encode("utf-8"), document_id="doc-2"
    )
    chunks = chunks_from_document(note, "user-a", "kb-a", "doc-2")
    assert [item["content"] for item in chunks] == ["第一段", "第二段"]
    assert chunks[0]["chunk_id"] == "doc-2-c0001"
    assert chunks[0]["unit_id"] == "t0001"
    assert chunks[0]["page_num"] is None
    assert chunks[0]["evidence_ids"]
    assert chunks[0]["evidence_refs"][0]["unit_id"] == "t0001"

    cited = citations_of(
        [
            {
                "doc_name": "手册.md",
                "chunk_id": "doc-1-c0001",
                "content": "节温器打不开",
                "page_number": 3,
                "section_path": ["冷却系统"],
                "unit_id": "s1",
                "document_id": "doc-1",
                "version_id": "ver-1",
            }
        ]
    )
    assert cited[0]["page_num"] == 3
    assert cited[0]["section_path"] == ["冷却系统"]
    assert cited[0]["unit_id"] == "s1"
    assert (
        citations_of(
            [{"doc_name": "旧文件", "chunk_id": "legacy", "content": "没有位置"}]
        )[0]["page_num"]
        is None
    )

    same = "节温器打不开".encode("utf-8")
    first = parse_bytes("a.md", same, document_id="doc-a")
    second = parse_bytes("b.md", same, document_id="doc-b")
    left = chunks_from_document(first, "user-a", "kb-a", "doc-a")
    right = chunks_from_document(second, "user-a", "kb-b", "doc-b")
    assert left[0]["document_id"] == "doc-a"
    assert right[0]["document_id"] == "doc-b"
    assert left[0]["evidence_ids"] != right[0]["evidence_ids"]
    native = {"chunk-1": {"content": "节温器打不开", "chunk_id": "chunk-1"}}
    bind_document_chunks(native, first)
    assert native["chunk-1"]["document_id"] == "doc-a"
    untouched = {"chunk-1": {"content": "节温器打不开", "chunk_id": "chunk-1"}}
    bind_document_chunks(untouched, None)
    assert "evidence_ids" not in untouched["chunk-1"]


def test_pdf_page_docx_section_pptx_slide_and_xlsx_cell():
    pdf = parse_bytes("manual.pdf", _pdf("thermostat"), document_id="doc-pdf")
    page = next(unit for unit in pdf.walk() if unit.unit_type == "page")
    assert page.page_number == 1
    assert "thermostat" in page.content
    assert page.bbox is None

    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_heading("冷却系统", level=1)
    document.add_paragraph("节温器打不开")
    table = document.add_table(rows=1, cols=1)
    table.cell(0, 0).text = "更换节温器"
    buffer = io.BytesIO()
    document.save(buffer)
    word = parse_bytes("manual.docx", buffer.getvalue(), document_id="doc-word")
    paragraph = next(unit for unit in word.walk() if unit.content == "节温器打不开")
    assert paragraph.section_path == ["冷却系统"]
    cell = next(unit for unit in word.walk() if unit.unit_type == "cell")
    assert cell.cell_range == "A1"
    assert cell.content == "更换节温器"

    pptx = pytest.importorskip("pptx")
    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    box = slide.shapes.add_textbox(
        pptx.util.Inches(1),
        pptx.util.Inches(1),
        pptx.util.Inches(4),
        pptx.util.Inches(1),
    )
    box.text_frame.text = "振动偏大"
    buffer = io.BytesIO()
    presentation.save(buffer)
    deck = parse_bytes("brief.pptx", buffer.getvalue(), document_id="doc-ppt")
    located = next(unit for unit in deck.walk() if "振动偏大" in unit.content)
    assert located.slide_number == 1
    assert located.unit_type == "slide"

    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "故障清单"
    sheet["B2"] = "节温器"
    buffer = io.BytesIO()
    book.save(buffer)
    table_doc = parse_bytes("list.xlsx", buffer.getvalue(), document_id="doc-xls")
    located = next(unit for unit in table_doc.walk() if unit.content == "节温器")
    assert located.sheet_name == "故障清单"
    assert located.cell_range == "B2"
    assert located.bbox is None
    sheet["A1"] = "部件"
    sheet["B1"] = "现象"
    sheet["A2"] = "节温器"
    sheet["B2"] = "打不开"
    buffer = io.BytesIO()
    book.save(buffer)
    table_doc = parse_bytes("list.xlsx", buffer.getvalue(), document_id="doc-xls")
    excel_chunks = chunks_from_document(
        table_doc, "user-a", "kb-a", "doc-xls", chunk_size=512, chunk_overlap=0
    )
    assert len(excel_chunks) == 1
    assert (
        "节温器" in excel_chunks[0]["content"]
        and "打不开" in excel_chunks[0]["content"]
    )
    assert len(excel_chunks[0]["evidence_refs"]) >= 2

    long_page = parse_bytes(
        "manual.pdf", _pdf("thermostat " * 80), document_id="doc-pdf-long"
    )
    page_chunks = chunks_from_document(
        long_page, "user-a", "kb-a", "doc-pdf-long", chunk_size=8, chunk_overlap=0
    )
    assert len(page_chunks) > 1
    assert {item["page_number"] for item in page_chunks} == {1}
    assert {item["evidence_refs"][0]["unit_id"] for item in page_chunks} == {"p1"}
