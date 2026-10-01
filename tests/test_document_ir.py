# -*- coding: utf-8 -*-
import pytest

from lightrag.product_document_ir import DocumentIR, EvidenceRef, SourceUnit, checksum_of


def _sample() -> DocumentIR:
    page = SourceUnit(
        unit_id="p1",
        unit_type="page",
        page_number=3,
        content="第三页",
        children=[
            SourceUnit(unit_id="p1-t", unit_type="text", page_number=3, content="节温器打不开", bbox=None),
        ],
    )
    section = SourceUnit(
        unit_id="s1",
        unit_type="section",
        section_path=["维护手册", "冷却系统"],
        children=[
            SourceUnit(
                unit_id="s1-table",
                unit_type="table",
                section_path=["维护手册", "冷却系统"],
                children=[
                    SourceUnit(unit_id="s1-c1", unit_type="cell", section_path=["维护手册", "冷却系统"], cell_range="B2", content="蜡包失效"),
                ],
            )
        ],
    )
    slide = SourceUnit(unit_id="sl2", unit_type="slide", slide_number=2, content="排查顺序")
    sheet = SourceUnit(unit_id="sheet-故障", unit_type="sheet", sheet_name="故障清单", content="表头")
    return DocumentIR(
        document_id="doc-1",
        version_id="ver-1",
        source_name="维护手册.pdf",
        source_type="pdf",
        checksum=checksum_of("维护手册".encode("utf-8")),
        units=[page, section, slide, sheet],
        parse_summary={"pages": 1, "slides": 1, "sheets": 1, "bbox_missing": 4},
    )


def test_document_ir_roundtrip_keeps_null_bbox_and_locations():
    document = _sample()
    restored = DocumentIR.from_json(document.to_json())
    assert restored.document_id == "doc-1"
    assert restored.version_id == "ver-1"
    assert restored.checksum == document.checksum
    text = restored.find_unit("p1-t")
    assert text is not None
    assert text.page_number == 3
    assert text.bbox is None
    assert restored.find_unit("s1-c1").cell_range == "B2"
    assert restored.find_unit("sl2").slide_number == 2
    assert restored.find_unit("sheet-故障").sheet_name == "故障清单"
    assert restored.to_dict()["units"][0]["children"][0]["bbox"] is None


def test_evidence_points_back_to_the_same_unit():
    document = _sample()
    evidence = document.evidence("doc-1-c0001", "s1-c1", excerpt="蜡包失效")
    assert evidence.document_id == "doc-1"
    assert evidence.version_id == "ver-1"
    assert evidence.chunk_id == "doc-1-c0001"
    assert evidence.unit_id == "s1-c1"
    assert evidence.section_path == ["维护手册", "冷却系统"]
    assert evidence.cell_range == "B2"
    assert evidence.page_number is None
    again = EvidenceRef.from_dict(evidence.to_dict())
    assert again.evidence_id == evidence.evidence_id
    with pytest.raises(ValueError):
        document.evidence("doc-1-c0001", "missing")


def test_duplicate_unit_and_unknown_type_are_rejected():
    with pytest.raises(ValueError):
        DocumentIR(
            document_id="doc-1",
            version_id="ver-1",
            source_name="a.md",
            source_type="markdown",
            checksum="abc",
            units=[
                SourceUnit(unit_id="same", unit_type="text", content="甲"),
                SourceUnit(unit_id="same", unit_type="text", content="乙"),
            ],
        )
    with pytest.raises(ValueError):
        SourceUnit(unit_id="x", unit_type="paragraph", content="不行")
