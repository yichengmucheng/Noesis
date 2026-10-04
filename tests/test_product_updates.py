from pathlib import Path

from lightrag.product_appdb import open_appdb, reset_appdb_cache
from lightrag.product_updates import (
    change_summary,
    compare_chunks,
    content_hash,
    fact_state,
)
from lightrag.product_uploads import allocate_upload


def test_chunk_diff_keeps_unchanged_content_after_insertion():
    old = [
        {"chunk_id": "old-1", "content": "第一章 设备说明"},
        {"chunk_id": "old-2", "content": "泵的额定压力为 8MPa"},
    ]
    new = [
        {"chunk_id": "new-1", "content": "新增的安全提示"},
        {"chunk_id": "new-2", "content": "第一章 设备说明"},
        {"chunk_id": "new-3", "content": "泵的额定压力为 8MPa"},
    ]
    rows = compare_chunks(old, new, old_document_id="logical", new_document_id="logical")
    assert change_summary(rows)["unchanged"] == 2
    assert change_summary(rows)["added"] == 1
    assert all(row["type"] != "deleted" for row in rows)


def test_same_payload_hash_and_fact_lifecycle(tmp_path: Path):
    assert content_hash(b"same") == content_hash(b"same")
    state = fact_state(fact_id="fact-1", source_version_id="v1")
    assert state["status"] == "active"
    assert state["invalid_at"] == ""

    store = open_appdb(tmp_path)
    store.register_knowledge_version(
        document_id="logical-1",
        version_id="v1",
        owner_id="u1",
        kb_id="kb1",
        display_name="manual.txt",
        source_type="txt",
        content_hash="hash-1",
        structure_hash="struct-1",
    )
    store.register_knowledge_version(
        document_id="logical-1",
        version_id="v2",
        owner_id="u1",
        kb_id="kb1",
        display_name="manual.txt",
        source_type="txt",
        content_hash="hash-2",
        structure_hash="struct-2",
    )
    assert store.publish_knowledge_version("logical-1", "v2", "u1", "kb1") is True
    assert [row["version_id"] for row in store.list_knowledge_versions("logical-1", "u1", "kb1")] == ["v2", "v1"]
    assert store.get_knowledge_version("v2", "u1", "kb1")["status"] == "published"
    store.activate_knowledge_fact({
        "fact_id": "fact-v1",
        "owner_id": "u1",
        "kb_id": "kb1",
        "subject_id": "pump",
        "predicate": "额定压力",
        "object_id": "",
        "value": "8MPa",
        "source_version_id": "v1",
    })
    store.activate_knowledge_fact({
        "fact_id": "fact-v2",
        "owner_id": "u1",
        "kb_id": "kb1",
        "subject_id": "pump",
        "predicate": "额定压力",
        "object_id": "",
        "value": "10MPa",
        "source_version_id": "v2",
    })
    current = store.list_current_knowledge_facts("u1", "kb1")
    assert [row["fact_id"] for row in current] == ["fact-v2"]
    assert store.get_knowledge_change_set("missing", "u1", "kb1") is None
    reset_appdb_cache()


def test_allocate_upload_reuses_logical_document_and_detects_duplicate():
    data = {"doc_index": {}}
    first = allocate_upload(data, "u1", "kb1", "manual.txt", content_hash="h1", version_id="v1")
    second = allocate_upload(data, "u1", "kb1", "manual.txt", content_hash="h2", version_id="v2")
    assert first["document_id"] == second["document_id"]
    duplicate = allocate_upload(data, "u1", "kb1", "renamed.txt", content_hash="h1", version_id="v1")
    assert duplicate["duplicate"] is True
    assert duplicate["doc_id"] == first["doc_id"]
