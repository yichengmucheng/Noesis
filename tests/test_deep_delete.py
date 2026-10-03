# -*- coding: utf-8 -*-
"""用真实的 JSON、NanoVectorDB 和 NetworkX 文件验证删除。"""

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from nano_vectordb import NanoVectorDB

from lightrag.operate import lookup_chunk_plan, register_chunk_plan
from lightrag.product_deletion import begin_purge
from lightrag.product_scope import scope_visible
from lightrag.product_storage import (
    ProcessCrash,
    arm_failure,
    deep_check,
    execute_purge,
    load_shell,
)
from lightrag.product_uploads import abort_upload, allocate_upload


def _write_kv(folder: Path, name: str, rows: dict) -> None:
    (folder / name).write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def _vdb(path: Path, rows: list[dict]) -> None:
    db = NanoVectorDB(embedding_dim=3, storage_file=str(path))
    payload = []
    for index, row in enumerate(rows):
        item = dict(row)
        item["__vector__"] = np.array([1.0, float(index), 0.0], dtype=np.float32)
        payload.append(item)
    db.upsert(payload)
    db.save()


def _build(root: Path) -> tuple[Path, Path, str]:
    working = root / "rag"
    inputs = root / "inputs"
    working.mkdir(parents=True)
    inputs.mkdir(parents=True)
    docs = {
        "doc-a1": {
            "doc_id": "doc-a1",
            "owner_id": "user-a",
            "kb_id": "kb-a1",
            "display_name": "report.pdf",
            "storage_key": "user-a/kb-a1/doc-a1/report.pdf",
            "chunk_ids": ["chunk-a1"],
            "version": 1,
        },
        "doc-a2": {
            "doc_id": "doc-a2",
            "owner_id": "user-a",
            "kb_id": "kb-a2",
            "display_name": "notes.txt",
            "storage_key": "user-a/kb-a2/doc-a2/notes.txt",
            "chunk_ids": ["chunk-a2"],
            "version": 1,
        },
        "doc-b1": {
            "doc_id": "doc-b1",
            "owner_id": "user-b",
            "kb_id": "kb-b1",
            "display_name": "report.pdf",
            "storage_key": "user-b/kb-b1/doc-b1/report.pdf",
            "chunk_ids": ["chunk-b1"],
            "version": 1,
        },
    }
    for row in docs.values():
        path = inputs / row["storage_key"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(row["doc_id"], encoding="utf-8")
    shell = {
        "kbs": [
            {"id": "kb-a1", "owner_id": "user-a"},
            {"id": "kb-a2", "owner_id": "user-a"},
            {"id": "kb-b1", "owner_id": "user-b"},
        ],
        "doc_index": docs,
        "file_bindings": {},
        "qa_pairs": [
            {"id": "qa-a1", "kb_id": "kb-a1"},
            {"id": "qa-a2", "kb_id": "kb-a2"},
        ],
        "comparisons": [{"task_id": "cmp-a1", "kb_id": "kb-a1"}],
        "audits": {"audit-a1": {"kb_id": "kb-a1"}},
        "sessions": {"sess-a1": {"kb_id": "kb-a1"}, "sess-a2": {"kb_id": "kb-a2"}},
        "purge_jobs": [],
    }
    job = begin_purge(shell, shell["kbs"][0])
    (working / "product_shell.json").write_text(
        json.dumps(shell, ensure_ascii=False), encoding="utf-8"
    )
    (working / "semantic_cache.json").write_text(
        json.dumps(
            [
                {
                    "id": "cache-a1",
                    "kb_id": "kb-a1",
                    "user_id": "user-a",
                    "kb_version": "v1",
                    "model": "m",
                    "prompt_version": "answer-v1",
                    "chunk_ids": ["chunk-a1"],
                },
                {
                    "id": "cache-a2",
                    "kb_id": "kb-a2",
                    "user_id": "user-a",
                    "kb_version": "v1",
                    "model": "m",
                    "prompt_version": "answer-v1",
                    "chunk_ids": ["chunk-a2"],
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_kv(
        working,
        "kv_store_doc_status.json",
        {
            "doc-a1": {
                "kb_id": "kb-a1",
                "owner_id": "user-a",
                "doc_id": "doc-a1",
                "file_path": docs["doc-a1"]["storage_key"],
            },
            "doc-a2": {
                "kb_id": "kb-a2",
                "owner_id": "user-a",
                "doc_id": "doc-a2",
                "file_path": docs["doc-a2"]["storage_key"],
            },
            "doc-b1": {
                "kb_id": "kb-b1",
                "owner_id": "user-b",
                "doc_id": "doc-b1",
                "file_path": docs["doc-b1"]["storage_key"],
            },
        },
    )
    _write_kv(
        working,
        "kv_store_full_docs.json",
        {
            "doc-a1": {
                "content": "ALPHA",
                "kb_id": "kb-a1",
                "doc_id": "doc-a1",
                "owner_id": "user-a",
            },
            "doc-a2": {
                "content": "A2",
                "kb_id": "kb-a2",
                "doc_id": "doc-a2",
                "owner_id": "user-a",
            },
            "doc-b1": {
                "content": "BETA",
                "kb_id": "kb-b1",
                "doc_id": "doc-b1",
                "owner_id": "user-b",
            },
        },
    )
    _write_kv(
        working,
        "kv_store_text_chunks.json",
        {
            "chunk-a1": {
                "content": "ALPHA",
                "kb_id": "kb-a1",
                "doc_id": "doc-a1",
                "chunk_id": "chunk-a1",
                "full_doc_id": "doc-a1",
                "owner_id": "user-a",
            },
            "chunk-a2": {
                "content": "A2",
                "kb_id": "kb-a2",
                "doc_id": "doc-a2",
                "chunk_id": "chunk-a2",
                "full_doc_id": "doc-a2",
                "owner_id": "user-a",
            },
            "chunk-b1": {
                "content": "BETA",
                "kb_id": "kb-b1",
                "doc_id": "doc-b1",
                "chunk_id": "chunk-b1",
                "full_doc_id": "doc-b1",
                "owner_id": "user-b",
            },
            "chunk-orphan": {"content": "无归属"},
        },
    )
    _write_kv(working, "kv_store_full_entities.json", {})
    _write_kv(working, "kv_store_full_relations.json", {})
    _vdb(
        working / "vdb_chunks.json",
        [
            {
                "__id__": "chunk-a1",
                "chunk_id": "chunk-a1",
                "doc_id": "doc-a1",
                "kb_id": "kb-a1",
                "owner_id": "user-a",
                "content": "ALPHA",
            },
            {
                "__id__": "chunk-a2",
                "chunk_id": "chunk-a2",
                "doc_id": "doc-a2",
                "kb_id": "kb-a2",
                "owner_id": "user-a",
                "content": "A2",
            },
            {
                "__id__": "chunk-b1",
                "chunk_id": "chunk-b1",
                "doc_id": "doc-b1",
                "kb_id": "kb-b1",
                "owner_id": "user-b",
                "content": "BETA",
            },
        ],
    )
    _vdb(
        working / "vdb_entities.json",
        [
            {
                "__id__": "ent-pump",
                "entity_name": "泵",
                "source_id": "chunk-a1<SEP>chunk-a2",
                "kb_id": "kb-a1<SEP>kb-a2",
                "content": "来自A1<SEP>来自A2",
                "file_path": "report.pdf<SEP>notes.txt",
            },
            {
                "__id__": "ent-seal",
                "entity_name": "密封圈",
                "source_id": "chunk-a1",
                "kb_id": "kb-a1",
                "content": "只属于A1",
                "file_path": "report.pdf",
            },
            {
                "__id__": "ent-bearing",
                "entity_name": "轴承",
                "source_id": "chunk-a2",
                "kb_id": "kb-a2",
                "content": "只属于A2",
                "file_path": "notes.txt",
            },
            {
                "__id__": "ent-unit",
                "entity_name": "机组",
                "source_id": "chunk-a1<SEP>chunk-a2",
                "kb_id": "kb-a1<SEP>kb-a2",
                "content": "共同<SEP>仍在",
                "file_path": "report.pdf<SEP>notes.txt",
            },
        ],
    )
    _vdb(
        working / "vdb_relationships.json",
        [
            {
                "__id__": "rel-seal",
                "src_id": "泵",
                "tgt_id": "密封圈",
                "source_id": "chunk-a1",
                "kb_id": "kb-a1",
                "content": "只属于A1",
                "file_path": "report.pdf",
            },
            {
                "__id__": "rel-bearing",
                "src_id": "泵",
                "tgt_id": "轴承",
                "source_id": "chunk-a2",
                "kb_id": "kb-a2",
                "content": "只属于A2",
                "file_path": "notes.txt",
            },
            {
                "__id__": "rel-unit",
                "src_id": "泵",
                "tgt_id": "机组",
                "source_id": "chunk-a1<SEP>chunk-a2",
                "kb_id": "kb-a1<SEP>kb-a2",
                "content": "共同<SEP>仍在",
                "file_path": "report.pdf<SEP>notes.txt",
            },
        ],
    )
    graph = nx.Graph()
    graph.add_node(
        "泵",
        source_id="chunk-a1<SEP>chunk-a2",
        kb_id="kb-a1<SEP>kb-a2",
        description="来自A1<SEP>来自A2",
        file_path="report.pdf<SEP>notes.txt",
    )
    graph.add_node(
        "密封圈",
        source_id="chunk-a1",
        kb_id="kb-a1",
        description="只属于A1",
        file_path="report.pdf",
    )
    graph.add_node(
        "轴承",
        source_id="chunk-a2",
        kb_id="kb-a2",
        description="只属于A2",
        file_path="notes.txt",
    )
    graph.add_node(
        "机组",
        source_id="chunk-a1<SEP>chunk-a2",
        kb_id="kb-a1<SEP>kb-a2",
        description="共同<SEP>仍在",
        file_path="report.pdf<SEP>notes.txt",
    )
    graph.add_edge(
        "泵",
        "密封圈",
        source_id="chunk-a1",
        kb_id="kb-a1",
        description="只属于A1",
        file_path="report.pdf",
    )
    graph.add_edge(
        "泵",
        "轴承",
        source_id="chunk-a2",
        kb_id="kb-a2",
        description="只属于A2",
        file_path="notes.txt",
    )
    graph.add_edge(
        "泵",
        "机组",
        source_id="chunk-a1<SEP>chunk-a2",
        kb_id="kb-a1<SEP>kb-a2",
        description="共同<SEP>仍在",
        file_path="report.pdf<SEP>notes.txt",
    )
    nx.write_graphml(graph, working / "graph_chunk_entity_relation.graphml")
    (working / "tmp-doc-a1.partial").write_text("partial", encoding="utf-8")
    return working, inputs, job["job_id"]


def _store_count(report: dict, name: str) -> int:
    return int(report["stores"][name]["count"])


def test_same_filename_is_isolated_and_delete_is_physical(tmp_path: Path):
    working, inputs, job_id = _build(tmp_path)
    chunks = json.loads(
        (working / "kv_store_text_chunks.json").read_text(encoding="utf-8")
    )
    shell = load_shell(working)
    assert scope_visible(chunks["chunk-a1"], "kb-a1", {}, shell["doc_index"]) is True
    assert scope_visible(chunks["chunk-a1"], "kb-a2", {}, shell["doc_index"]) is False
    assert scope_visible(chunks["chunk-a1"], "kb-b1", {}, shell["doc_index"]) is False
    assert scope_visible(chunks["chunk-b1"], "kb-b1", {}, shell["doc_index"]) is True
    assert (
        scope_visible(chunks["chunk-orphan"], "kb-a1", {}, shell["doc_index"]) is False
    )
    before = deep_check(working, "kb-a1", inputs)
    assert before["passed"] is False
    assert _store_count(before, "text_chunks") == 1
    assert _store_count(before, "chunks_vdb") == 1
    assert _store_count(before, "graph_nodes") == 3

    report = execute_purge(working, inputs, job_id)
    assert report["passed"] is True
    assert report["checker_version"] == "deep-check-1"
    assert report["checked_kb_id"] == "kb-a1"
    assert "doc-a1" in report["checked_doc_ids"]
    assert "chunk-a1" in report["checked_chunk_ids"]
    assert all(item["count"] == 0 for item in report["stores"].values())
    assert not (inputs / "user-a/kb-a1/doc-a1/report.pdf").exists()
    assert (inputs / "user-b/kb-b1/doc-b1/report.pdf").read_text(
        encoding="utf-8"
    ) == "doc-b1"
    assert (inputs / "user-a/kb-a2/doc-a2/notes.txt").read_text(
        encoding="utf-8"
    ) == "doc-a2"
    remaining = json.loads(
        (working / "kv_store_text_chunks.json").read_text(encoding="utf-8")
    )
    assert "chunk-a1" not in remaining
    assert remaining["chunk-a2"]["content"] == "A2"
    assert remaining["chunk-b1"]["content"] == "BETA"
    assert "chunk-orphan" in remaining
    graph = nx.read_graphml(working / "graph_chunk_entity_relation.graphml")
    assert "密封圈" not in graph.nodes
    assert "泵" in graph.nodes
    assert graph.nodes["泵"]["source_id"] == "chunk-a2"
    assert "kb-a1" not in graph.nodes["泵"]["kb_id"]
    assert graph.nodes["轴承"]["source_id"] == "chunk-a2"
    assert graph.has_edge("泵", "机组")
    assert graph.edges["泵", "机组"]["source_id"] == "chunk-a2"
    assert not graph.has_edge("泵", "密封圈")
    entities = json.loads((working / "vdb_entities.json").read_text(encoding="utf-8"))[
        "data"
    ]
    names = {row["entity_name"]: row for row in entities}
    assert "密封圈" not in names
    assert names["泵"]["source_id"] == "chunk-a2"
    assert names["轴承"]["kb_id"] == "kb-a2"
    cache = json.loads((working / "semantic_cache.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in cache] == ["cache-a2"]
    other = deep_check(working, "kb-a2", inputs)
    assert _store_count(other, "text_chunks") == 1
    assert _store_count(other, "source_files") == 1
    again = execute_purge(working, inputs, job_id)
    assert again["passed"] is True
    assert load_shell(working)["purge_jobs"][0]["status"] == "succeeded"


def test_purge_resumes_after_crash_and_retries_after_storage_failure(tmp_path: Path):
    crashed = tmp_path / "crash"
    working, inputs, job_id = _build(crashed)
    with pytest.raises(ProcessCrash):
        execute_purge(working, inputs, job_id, crash_after="docs")
    job = load_shell(working)["purge_jobs"][0]
    assert job["status"] == "running"
    chunks = json.loads(
        (working / "kv_store_text_chunks.json").read_text(encoding="utf-8")
    )
    assert "chunk-a1" not in chunks
    assert "chunk-a2" in chunks
    graph = nx.read_graphml(working / "graph_chunk_entity_relation.graphml")
    assert "密封圈" in graph.nodes
    resumed = execute_purge(working, inputs, job_id)
    assert resumed["passed"] is True
    graph = nx.read_graphml(working / "graph_chunk_entity_relation.graphml")
    assert "密封圈" not in graph.nodes
    assert graph.nodes["泵"]["source_id"] == "chunk-a2"
    assert (inputs / "user-a/kb-a2/doc-a2/notes.txt").exists()

    failed = tmp_path / "fail"
    working, inputs, job_id = _build(failed)
    arm_failure("vectors")
    try:
        first = execute_purge(working, inputs, job_id)
    finally:
        arm_failure("")
    assert first["job_status"] == "failed"
    assert (inputs / "user-b/kb-b1/doc-b1/report.pdf").read_text(
        encoding="utf-8"
    ) == "doc-b1"
    assert "chunk-a2" in json.loads(
        (working / "kv_store_text_chunks.json").read_text(encoding="utf-8")
    )
    second = execute_purge(working, inputs, job_id)
    assert second["passed"] is True
    assert (
        "密封圈"
        not in nx.read_graphml(working / "graph_chunk_entity_relation.graphml").nodes
    )


def test_same_display_name_and_failed_upload_are_isolated(tmp_path: Path):
    data = {"doc_index": {}}
    first = allocate_upload(data, "user-a", "kb-a1", "report.pdf")
    second = allocate_upload(data, "user-b", "kb-b1", "report.pdf")
    third = allocate_upload(data, "user-a", "kb-a1", "report.pdf")
    assert first["storage_key"] != second["storage_key"]
    assert first["version"] == 1
    assert third["version"] == 2
    assert second["display_name"] == "report.pdf"
    inputs = tmp_path / "inputs"
    working = tmp_path / "rag"
    inputs.mkdir()
    working.mkdir()
    for row, text in ((first, "A"), (second, "B")):
        path = inputs / row["storage_key"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        register_chunk_plan(str(path), {"strategy": text})
    assert lookup_chunk_plan(str(inputs / first["storage_key"]))["strategy"] == "A"
    assert lookup_chunk_plan(str(inputs / second["storage_key"]))["strategy"] == "B"
    _write_kv(
        working,
        "kv_store_text_chunks.json",
        {
            "chunk-a1": {
                "content": "半成品",
                "kb_id": "kb-a1",
                "doc_id": first["doc_id"],
                "chunk_id": "chunk-a1",
                "full_doc_id": first["doc_id"],
            },
        },
    )
    _write_kv(working, "kv_store_doc_status.json", {})
    _write_kv(working, "kv_store_full_docs.json", {})
    _write_kv(working, "kv_store_full_entities.json", {})
    _write_kv(working, "kv_store_full_relations.json", {})
    data["doc_index"][first["doc_id"]]["chunk_ids"] = ["chunk-a1"]
    (working / "product_shell.json").write_text(json.dumps(data), encoding="utf-8")
    (working / "semantic_cache.json").write_text("[]", encoding="utf-8")
    job = abort_upload(data, working, inputs, first["doc_id"], "embedding")
    assert job["stage"] == "failed"
    assert job["error_msg"] == "处理超时"
    assert not (inputs / first["storage_key"]).exists()
    assert (inputs / second["storage_key"]).read_text(encoding="utf-8") == "B"
    left = json.loads(
        (working / "kv_store_text_chunks.json").read_text(encoding="utf-8")
    )
    assert "chunk-a1" not in left
    cancelled = allocate_upload(data, "user-a", "kb-a2", "report.pdf")
    data["doc_index"][cancelled["doc_id"]]["chunk_ids"] = []
    (inputs / cancelled["storage_key"]).parent.mkdir(parents=True, exist_ok=True)
    (inputs / cancelled["storage_key"]).write_text("C", encoding="utf-8")
    _write_kv(
        working,
        "kv_store_text_chunks.json",
        {
            "chunk-c": {
                "content": "取消半成品",
                "kb_id": "kb-a2",
                "doc_id": cancelled["doc_id"],
                "chunk_id": "chunk-c",
                "full_doc_id": cancelled["doc_id"],
            },
        },
    )
    data["doc_index"][cancelled["doc_id"]]["chunk_ids"] = ["chunk-c"]
    stopped = abort_upload(
        data, working, inputs, cancelled["doc_id"], "graph", cancelled=True
    )
    assert stopped["stage"] == "cancelled"
    assert "chunk-c" not in json.loads(
        (working / "kv_store_text_chunks.json").read_text(encoding="utf-8")
    )
    assert (
        scope_visible({"doc_id": cancelled["doc_id"]}, "kb-a2", {}, data["doc_index"])
        is False
    )
