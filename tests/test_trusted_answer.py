# -*- coding: utf-8 -*-
import asyncio
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightrag.answer_pipeline import (
    DETAIL_HINT,
    REFUSAL,
    SYSTEM_PROMPT,
    build_messages,
    cache_entry_usable,
    normalize_detail,
    stream_answer,
    validate_citations,
)
from lightrag.product_document_ir import DocumentIR, SourceUnit
from lightrag.product_parse import parse_bytes, save_ir
from lightrag.product_source import build_source_view


def _pdf(text: str) -> bytes:
    stream = f"BT /F1 24 Tf 72 120 Td ({text}) Tj ET\n".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"endstream",
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


def _hit():
    return {
        "chunk_id": "c1",
        "content": "节温器打不开",
        "excerpt": "节温器打不开",
        "doc_name": "手册.md",
        "document_id": "doc-1",
        "version_id": "v1",
        "unit_id": "t1",
        "kb_id": "kb",
        "owner_id": "user-a",
        "section_path": ["冷却系统"],
        "parent_content": "冷却系统说明" * 80,
    }


def test_detail_levels_change_prompt_budget_not_system_rules():
    contexts = [
        {**_hit(), "parent_content": "父" * 2000},
        {**_hit(), "chunk_id": "c2", "content": "另一条证据", "parent_content": ""},
    ]
    prompts = {}
    for level in ("concise", "standard", "detailed"):
        system, user = build_messages("现在怎么办", contexts, [], None, detail=level, expand_context=True)
        prompts[level] = user
        assert system == SYSTEM_PROMPT
        assert DETAIL_HINT[level] in user
        assert "节温器打不开" in user
    assert prompts["concise"].count("父") < prompts["standard"].count("父") < prompts["detailed"].count("父")
    assert "只写结论" in prompts["concise"]
    assert "80字以内" in prompts["concise"]
    assert "220字以内" in prompts["standard"]
    assert "600字以内" in prompts["detailed"]
    assert "关键解释" in prompts["standard"]
    assert "分点" in prompts["detailed"]
    assert normalize_detail("brief") == "concise"
    assert normalize_detail("normal") == "standard"


def test_validator_drops_forged_cross_kb_and_stale_versions():
    citations = [
        {"citation_id": "C1", "chunk_id": "c1", "document_id": "doc-1", "version_id": "v1", "kb_id": "kb", "owner_id": "user-a"},
        {"citation_id": "C2", "chunk_id": "c2", "document_id": "doc-2", "version_id": "old", "kb_id": "kb", "owner_id": "user-a"},
        {"citation_id": "C3", "chunk_id": "c3", "document_id": "doc-3", "version_id": "v3", "kb_id": "other", "owner_id": "user-a"},
    ]
    checked = validate_citations(
        "结论 [C1]，伪造 [C9]，旧版本 [C2]，别的库 [C3]。",
        citations,
        kb_id="kb",
        owner_id="user-a",
        live_versions={"doc-1": "v1", "doc-2": "v2", "doc-3": "v3"},
        retrieved_ids={"c1", "c2", "c3"},
    )
    assert checked["answerable"] is True
    assert [item["citation_id"] for item in checked["citations"]] == ["C1"]
    assert "[C9]" not in checked["answer"]
    assert "[C2]" not in checked["answer"]
    assert "[C3]" not in checked["answer"]
    empty = validate_citations("没有引用的猜测", citations, kb_id="kb", owner_id="user-a", retrieved_ids={"c1"})
    assert empty["answer"] == REFUSAL
    assert empty["citations"] == []
    assert empty["answerable"] is False


def test_cache_rejects_index_or_detail_mismatch():
    entry = {
        "kb_id": "kb",
        "tenant": "default",
        "kb_version": "v1",
        "prompt_version": "answer-v2",
        "model": "qwen",
        "mode": "hybrid",
        "chunk_ids": ["c1"],
        "user_id": "user-a",
        "index_version": "idx-1",
        "detail": "standard",
    }
    common = dict(kb_id="kb", tenant="default", kb_version="v1", mode="hybrid", model="qwen", live_chunk_ids={"c1"}, user_id="user-a")
    assert cache_entry_usable(entry, index_version="idx-1", detail="standard", **common)
    assert not cache_entry_usable(entry, index_version="idx-2", detail="standard", **common)
    assert not cache_entry_usable(entry, index_version="idx-1", detail="detailed", **common)


def test_stream_uses_detail_and_stops_before_a_complete_save():
    class Rag:
        def __init__(self):
            self.prompts = []

        async def llm_model_func(self, prompt, system_prompt=None, stream=False):
            self.prompts.append(prompt)
            if "伪造" in prompt:
                return '{"answer":"瞎编 [C9]","answerable":true}'
            return '{"answer":"结论 [C1]","answerable":true}'

        async def embedding_func(self, texts):
            return [[1.0, 0.0]]

    prepared = {"status": "ok", "chunks": [_hit()], "answer_context": []}

    async def collect(detail):
        rag = Rag()
        events = []
        async for event in stream_answer(
            rag,
            query="怎么办",
            mode="mix",
            ratio=0.5,
            kb_chunks=[{"chunk_id": "c1"}],
            file_in_kb=lambda *_args: True,
            history=[],
            summary=None,
            kb_id="kb",
            tenant="default",
            kb_version="v1",
            cache_entries=[],
            user_id="user-a",
            detail=detail,
            index_version="idx-1",
            owner_id="user-a",
            prepared=prepared,
            live_versions={"doc-1": "v1"},
        ):
            events.append(event)
        return rag, events

    concise_rag, concise = asyncio.run(collect("concise"))
    detailed_rag, detailed = asyncio.run(collect("detailed"))
    assert "只写结论" in concise_rag.prompts[0]
    assert "分点" in detailed_rag.prompts[0]
    assert concise[-1]["complete"] is True
    assert concise[-1]["citations"][0]["citation_id"] == "C1"
    assert concise[-1]["answer"] == "结论 [C1]"

    forged = Rag()

    async def forged_answer(prompt, system_prompt=None, stream=False):
        return '{"answer":"瞎编 [C9]","answerable":true}'

    forged.llm_model_func = forged_answer
    forged.embedding_func = lambda texts: _embed()

    async def _embed():
        return [[1.0, 0.0]]

    events = asyncio.run(_events(forged, prepared))
    assert events[-1]["answer"] == REFUSAL
    assert events[-1]["citations"] == []
    assert events[-1]["store_cache"] is False

    async def stopped():
        agen = stream_answer(
            forged,
            query="怎么办",
            mode="mix",
            ratio=0.5,
            kb_chunks=[{"chunk_id": "c1"}],
            file_in_kb=lambda *_args: True,
            history=[],
            summary=None,
            kb_id="kb",
            tenant="default",
            kb_version="v1",
            cache_entries=[],
            user_id="user-a",
            prepared=prepared,
            live_versions={"doc-1": "v1"},
        )
        first = await agen.__anext__()
        await agen.aclose()
        return first

    assert asyncio.run(stopped())["type"] == "meta"


async def _events(rag, prepared):
    rows = []
    async for event in stream_answer(
        rag,
        query="怎么办",
        mode="mix",
        ratio=0.5,
        kb_chunks=[{"chunk_id": "c1"}],
        file_in_kb=lambda *_args: True,
        history=[],
        summary=None,
        kb_id="kb",
        tenant="default",
        kb_version="v1",
        cache_entries=[],
        user_id="user-a",
        prepared=prepared,
        live_versions={"doc-1": "v1"},
    ):
        rows.append(event)
    return rows


def test_source_locations_for_each_format(tmp_path):
    markdown = parse_bytes("手册.md", "冷却系统\n\n# 冷却系统\n\n节温器打不开\n".encode("utf-8"), document_id="doc-md")
    body = next(unit for unit in markdown.walk() if unit.content == "节温器打不开")
    view = build_source_view(markdown, doc_name="手册.md", chunk_id="c1", unit_id=body.unit_id)
    assert view["source_kind"] == "markdown"
    assert view["unit"]["section_path"] == ["冷却系统"]
    assert view["unit"]["line_start"]
    assert view["preview"]["available"] is False
    assert view["preview"]["url"] is None
    assert "file_path" not in view

    note = parse_bytes("note.txt", "第一段\n\n第二段".encode("utf-8"), document_id="doc-txt")
    text = build_source_view(note, doc_name="note.txt", chunk_id="c1", unit_id="t0001")
    assert text["unit"]["line_start"] == 1
    assert text["unit"]["page_number"] is None

    pdf = parse_bytes("manual.pdf", _pdf("thermostat"), document_id="doc-pdf")
    page = next(unit for unit in pdf.walk() if unit.unit_type == "page")
    located = build_source_view(pdf, doc_name="manual.pdf", chunk_id="c1", unit_id=page.unit_id)
    assert located["unit"]["page_number"] == 1
    assert located["unit"]["bbox"] is None
    boxed = DocumentIR(
        document_id="doc-box",
        version_id=pdf.version_id,
        source_name="boxed.pdf",
        source_type="pdf",
        checksum="abc",
        units=[SourceUnit(unit_id="p1", unit_type="page", page_number=2, content="坐标", bbox=[10, 20, 30, 40])],
    )
    box_view = build_source_view(boxed, doc_name="boxed.pdf", chunk_id="c1", unit_id="p1")
    assert box_view["unit"]["page_number"] == 2
    assert box_view["unit"]["bbox"] == [10, 20, 30, 40]

    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_heading("冷却系统", level=1)
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "蜡包"
    table.cell(0, 1).text = "失效"
    payload = Path(tmp_path / "a.docx")
    document.save(payload)
    parsed = parse_bytes("a.docx", payload.read_bytes(), document_id="doc-docx")
    cell = next(unit for unit in parsed.walk() if unit.unit_type == "cell" and unit.content == "失效")
    word = build_source_view(parsed, doc_name="a.docx", chunk_id="c1", unit_id=cell.unit_id)
    assert "冷却系统" in word["unit"]["section_path"]
    assert word["unit"]["cell_range"]

    pptx = pytest.importorskip("pptx")
    deck = pptx.Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[5])
    slide.shapes.title.text = "排查顺序"
    target = Path(tmp_path / "a.pptx")
    deck.save(target)
    slides = parse_bytes("a.pptx", target.read_bytes(), document_id="doc-pptx")
    slide_unit = next(unit for unit in slides.walk() if unit.unit_type == "slide")
    power = build_source_view(slides, doc_name="a.pptx", chunk_id="c1", unit_id=slide_unit.unit_id)
    assert power["unit"]["slide_number"] == 1

    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "故障清单"
    sheet["B2"] = "蜡包失效"
    sheet_path = Path(tmp_path / "a.xlsx")
    book.save(sheet_path)
    sheets = parse_bytes("a.xlsx", sheet_path.read_bytes(), document_id="doc-xlsx")
    cell_unit = next(unit for unit in sheets.walk() if "蜡包失效" in unit.content)
    excel = build_source_view(sheets, doc_name="a.xlsx", chunk_id="c1", unit_id=cell_unit.unit_id)
    assert excel["unit"]["sheet_name"] == "故障清单"
    assert excel["unit"]["cell_range"]

    missing = build_source_view(note, doc_name="note.txt", chunk_id="legacy", unit_id="missing")
    assert missing["unit"]["page_number"] is None
    assert missing["unit"]["line_start"] is None
    assert missing["unit"]["bbox"] is None


def test_source_api_isolates_owner_kb_version_and_deleted_files(tmp_path, monkeypatch):
    working = tmp_path / "rag"
    inputs = tmp_path / "inputs"
    working.mkdir()
    inputs.mkdir()
    monkeypatch.setenv("PRODUCT_AUTH", "1")
    monkeypatch.setenv("TOKEN_SECRET", "unit-test-secret-for-source-32b!")
    monkeypatch.setenv("AUTH_RATE_LIMIT", "1000")
    monkeypatch.setenv("APP_ENV", "development")

    class Rag:
        def __init__(self):
            self.working_dir = str(working)
            self.addon_params = {}

        async def get_docs_by_status(self, _status):
            return {}

    class Docs:
        def __init__(self):
            self.input_dir = str(inputs)

        def is_supported_file(self, name: str) -> bool:
            return True

    import sys

    sys.argv = [sys.argv[0]]
    from lightrag.api.routers.product_shell import create_product_shell_routes
    from lightrag.product_storage import mutate_shell

    app = FastAPI()
    app.include_router(create_product_shell_routes(Rag(), Docs()))
    client = TestClient(app)
    csrf = {"X-KB-Request": "1"}
    owner_token = client.post("/api/v1/auth/register", json={"email": "source-owner@example.com", "password": "correct-horse"}, headers=csrf)
    assert owner_token.status_code == 200, owner_token.text
    other_token = client.post("/api/v1/auth/register", json={"email": "source-other@example.com", "password": "correct-horse"}, headers=csrf)
    assert other_token.status_code == 200, other_token.text
    owner = {"Authorization": f"Bearer {owner_token.json()['access_token']}", **csrf}
    stranger = {"Authorization": f"Bearer {other_token.json()['access_token']}", **csrf}
    kb_id = client.post("/api/v1/kb", json={"name": "资料库"}, headers=owner).json()["id"]
    other_kb = client.post("/api/v1/kb", json={"name": "另一库"}, headers=owner).json()["id"]
    owner_id = client.get("/api/v1/kb", headers=owner).json()["items"][0]["owner_id"]
    note = parse_bytes("note.txt", "第一段\n\n第二段".encode("utf-8"), document_id="doc-note")
    save_ir(working, note)
    stored = inputs / "note.txt"
    stored.write_text("第一段\n\n第二段", encoding="utf-8")

    def editor(data):
        data.setdefault("doc_index", {})["doc-note"] = {
            "kb_id": kb_id,
            "owner_id": owner_id,
            "display_name": "note.txt",
            "storage_key": str(stored),
        }

    mutate_shell(working, editor)
    found = client.get("/api/v1/documents/doc-note/source", params={"kb_id": kb_id, "unit_id": "t0001"}, headers=owner)
    assert found.status_code == 200, found.text
    body = found.json()
    assert body["unit"]["line_start"] == 1
    assert "storage_key" not in json.dumps(body)
    assert str(stored) not in json.dumps(body)
    assert client.get("/api/v1/documents/doc-note/source", params={"kb_id": kb_id}, headers=stranger).status_code == 404
    assert client.get("/api/v1/documents/doc-note/source", params={"kb_id": other_kb}, headers=owner).status_code == 404
    assert client.get("/api/v1/documents/doc-note/source", params={"kb_id": kb_id, "version_id": "missing"}, headers=owner).status_code == 404
    content = client.get("/api/v1/documents/doc-note/content", params={"kb_id": kb_id, "unit_id": "t0001"}, headers=owner)
    assert content.status_code == 200, content.text
    assert "第一段" in content.json()["text"]
    assert "storage_key" not in content.text
    assert str(inputs) not in content.text
    downloaded = client.get("/api/v1/documents/doc-note/download", params={"kb_id": kb_id}, headers=owner)
    assert downloaded.status_code == 200, downloaded.text
    assert "attachment;" in downloaded.headers["content-disposition"]
    assert "note.txt" in downloaded.headers["content-disposition"]
    preview = client.get("/api/v1/documents/doc-note/preview", params={"kb_id": kb_id}, headers=owner)
    assert preview.status_code == 200
    for path in ("content", "download", "preview"):
        assert client.get(f"/api/v1/documents/doc-note/{path}", params={"kb_id": kb_id}, headers=stranger).status_code == 404
        assert client.get(f"/api/v1/documents/doc-note/{path}", params={"kb_id": other_kb}, headers=owner).status_code == 404
        assert client.get(f"/api/v1/documents/doc-note/{path}", params={"kb_id": kb_id, "version_id": "missing"}, headers=owner).status_code == 404

    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")

    def escape(data):
        data["doc_index"]["doc-note"]["storage_key"] = str(outside)

    mutate_shell(working, escape)
    assert client.get("/api/v1/documents/doc-note/download", params={"kb_id": kb_id}, headers=owner).status_code == 404
    assert "secret" not in client.get("/api/v1/documents/doc-note/content", params={"kb_id": kb_id}, headers=owner).text

    def restore(data):
        data["doc_index"]["doc-note"]["storage_key"] = str(stored)

    mutate_shell(working, restore)
    slides = inputs / "deck.pptx"
    slides.write_bytes(b"PK\x03\x04")

    def office(data):
        data["doc_index"]["doc-office"] = {
            "kb_id": kb_id,
            "owner_id": owner_id,
            "display_name": "deck.pptx",
            "storage_key": str(slides),
        }

    mutate_shell(working, office)
    office_preview = client.get("/api/v1/documents/doc-office/preview", params={"kb_id": kb_id}, headers=owner)
    assert office_preview.status_code == 415

    def remove(data):
        data["doc_index"]["doc-note"]["deleted_at"] = "2026-01-01T00:00:00Z"

    mutate_shell(working, remove)
    assert client.get("/api/v1/documents/doc-note/source", params={"kb_id": kb_id}, headers=owner).status_code == 404


def test_empty_manifest_model_requires_rebuild(tmp_path, monkeypatch):
    from lightrag.index_manifest import compatibility_error

    working = tmp_path / "rag"
    working.mkdir()
    (working / "vdb_chunks.json").write_text("{}", encoding="utf-8")
    (working / "index_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "embedding_model": "",
        "embedding_dimension": 2560,
        "embedding_instruction": "",
        "normalization": "l2",
        "chunking_version": "retrieval-180-350-overlap-45",
        "parent_chunk_version": "parent-800-1500-section",
    }), encoding="utf-8")
    monkeypatch.setenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B")
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)
    monkeypatch.delenv("EMBEDDING_INSTRUCTION", raising=False)
    monkeypatch.delenv("EMBEDDING_BINDING", raising=False)
    blocked = compatibility_error(working)
    assert blocked["status"] == "index_rebuild_required"
    assert "embedding 模型不一致" in blocked["reason"]


def test_cache_rejects_changed_document_version():
    entry = {
        "user_id": "user-a",
        "kb_id": "kb",
        "tenant": "default",
        "kb_version": "v1",
        "index_version": "idx",
        "detail": "standard",
        "prompt_version": "answer-v2",
        "model": "llm",
        "mode": "mix",
        "chunk_ids": ["c1"],
        "document_versions": {"doc-1": "v1"},
    }
    kwargs = dict(
        kb_id="kb", tenant="default", kb_version="v1", mode="mix", model="llm",
        live_chunk_ids={"c1"}, user_id="user-a", index_version="idx", detail="standard",
    )
    assert cache_entry_usable(entry, document_versions={"doc-1": "v1"}, **kwargs)
    assert not cache_entry_usable(entry, document_versions={"doc-1": "v2"}, **kwargs)


def test_cancelled_stream_does_not_save_partial_answer(tmp_path, monkeypatch):
    working = tmp_path / "rag"
    inputs = tmp_path / "inputs"
    working.mkdir()
    inputs.mkdir()
    monkeypatch.setenv("PRODUCT_AUTH", "1")
    monkeypatch.setenv("TOKEN_SECRET", "unit-test-secret-for-source-32b!")
    monkeypatch.setenv("AUTH_RATE_LIMIT", "1000")
    monkeypatch.setenv("APP_ENV", "development")

    class Rag:
        def __init__(self):
            self.working_dir = str(working)
            self.addon_params = {}

        async def get_docs_by_status(self, _status):
            return {}

    class Docs:
        def __init__(self):
            self.input_dir = str(inputs)

        def is_supported_file(self, name: str) -> bool:
            return True

    async def hanging(*_args, **_kwargs):
        yield {"type": "meta", "answerable": True, "citations": []}
        yield {"type": "token", "text": "半截答案"}
        await asyncio.Event().wait()
        yield {"type": "done", "answer": "半截答案", "complete": True, "store_cache": True}

    import sys

    sys.argv = [sys.argv[0]]
    from lightrag.api.routers import product_shell as shell_module
    from lightrag.product_storage import load_shell

    monkeypatch.setattr(shell_module, "stream_answer", hanging)
    app = FastAPI()
    app.include_router(shell_module.create_product_shell_routes(Rag(), Docs()))
    client = TestClient(app)
    csrf = {"X-KB-Request": "1"}
    token = client.post("/api/v1/auth/register", json={"email": "stream@example.com", "password": "correct-horse"}, headers=csrf).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}", **csrf}
    kb_id = client.post("/api/v1/kb", json={"name": "资料库"}, headers=headers).json()["id"]

    async def stop_early():
        payload = json.dumps({"query": "怎么办", "kb_id": kb_id}).encode("utf-8")
        state = {"phase": "request", "seen": asyncio.Event(), "bodies": []}

        async def receive():
            if state["phase"] == "request":
                state["phase"] = "open"
                return {"type": "http.request", "body": payload, "more_body": False}
            await state["seen"].wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body":
                chunk = message.get("body") or b""
                state["bodies"].append(chunk)
                if "半截答案".encode("utf-8") in chunk:
                    state["seen"].set()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/v1/chat/stream",
            "raw_path": b"/api/v1/chat/stream",
            "query_string": b"",
            "headers": [
                [b"content-type", b"application/json"],
                [b"content-length", str(len(payload)).encode("ascii")],
                [b"authorization", f"Bearer {token}".encode("utf-8")],
            ],
            "client": ("127.0.0.1", 9),
            "server": ("test", 80),
        }
        await app(scope, receive, send)
        assert any("半截答案".encode("utf-8") in chunk for chunk in state["bodies"])
        assert not any("complete".encode("utf-8") in chunk for chunk in state["bodies"])

    asyncio.run(asyncio.wait_for(stop_early(), timeout=20))
    saved = json.dumps(load_shell(working).get("qa_pairs") or [], ensure_ascii=False)
    assert "半截答案" not in saved
