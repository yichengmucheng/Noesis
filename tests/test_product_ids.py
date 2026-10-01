# -*- coding: utf-8 -*-
from lightrag.product_ids import (
    STORE_OF,
    check_bundle,
    chunk_record,
    entity_id,
    entity_record,
    object_key,
    relation_id,
    relation_record,
)


def _bundle():
    first = chunk_record("user-a", "kb-a", "doc-1", 1, "节温器打不开")
    second = chunk_record("user-a", "kb-a", "doc-1", 2, "检查蜡式节温器")
    pump = entity_record("user-a", "kb-a", "部件", "节温器", first["chunk_id"], "doc-1", "YC-6K")
    cause = entity_record("user-a", "kb-a", "一层原因", "蜡包失效", first["chunk_id"], "doc-1", "YC-6K")
    link = relation_record(
        "user-a", "kb-a", pump["entity_id"], "可能源于", cause["entity_id"], first["chunk_id"], "doc-1"
    )
    return [first, second, pump, cause, link]


def test_same_identity_is_stable_across_stores():
    rows = _bundle()
    check_bundle(rows)
    again = _bundle()
    assert [row["chunk_id"] for row in rows[:2]] == [row["chunk_id"] for row in again[:2]]
    assert rows[2]["entity_id"] == again[2]["entity_id"]
    assert rows[4]["relation_id"] == again[4]["relation_id"]
    assert rows[0]["source_id"] == rows[0]["chunk_id"]
    assert rows[2]["source_id"] == rows[0]["chunk_id"]
    assert rows[4]["src_entity_id"] == rows[2]["entity_id"]
    assert rows[4]["tgt_entity_id"] == rows[3]["entity_id"]
    assert STORE_OF["entity_id"]["neo4j"] == "node"
    assert STORE_OF["chunk_id"]["milvus"] == "chunks"
    assert STORE_OF["doc_id"]["postgres"] == "documents"
    assert STORE_OF["relation_id"]["neo4j"] == "relationship"
    assert STORE_OF["relation_id"]["milvus"] == "relations"


def test_scope_and_model_keep_same_names_apart():
    base = entity_id("kb-a", "部件", "节温器", "YC-6K")
    assert base == entity_id("kb-a", "部件", " 节温器 ", "yc-6k")
    assert base != entity_id("kb-b", "部件", "节温器", "YC-6K")
    assert base != entity_id("kb-a", "部件", "节温器", "YC-6M")
    assert base != entity_id("kb-a", "故障现象", "节温器", "YC-6K")
    same_ends = relation_id("kb-a", "ent-left", "导致", "ent-right")
    assert same_ends != relation_id("kb-a", "ent-left", "采取措施", "ent-right")
    assert object_key("user-a", "kb-a", "doc-1", "手册.pdf") == "user-a/kb-a/doc-1/手册.pdf"
    named = entity_record("user-a", "kb-a", "部件", "节温器", "doc-1-c0001", "doc-1")
    assert named["name"] != named["entity_id"]
