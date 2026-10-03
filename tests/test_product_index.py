# -*- coding: utf-8 -*-
import asyncio
import json
import threading

import numpy as np

from lightrag.product_db import SqliteStore
from lightrag.product_ingest import content_key, run_ingestion
from lightrag.product_scope import split_values
from lightrag.product_storage import (
    _graph,
    _kv,
    _vdb_rows,
    deep_check,
    load_shell,
    remove_documents,
    rewrite_graph,
    save_shell,
    update_vector_store,
)
from lightrag.product_uploads import allocate_upload
from lightrag.testing_index_runtime import embed_vector


def test_worker_embedding_uses_configured_qwen_dimension(monkeypatch):
    import lightrag.llm.openai as openai_module
    from lightrag.product_index import _configured_embed

    observed = {}

    async def fake_embed(texts, **kwargs):
        observed.update(kwargs)
        return [[0.0] * int(kwargs["dimensions"]) for _ in texts]

    monkeypatch.setenv("EMBEDDING_BINDING", "openai")
    monkeypatch.setenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B")
    monkeypatch.setenv("EMBEDDING_DIM", "1024")
    monkeypatch.setattr(openai_module, "openai_embed", fake_embed)
    result = asyncio.run(_configured_embed(["测试"]))
    assert observed["dimensions"] == 1024
    assert len(result[0]) == 1024


def _run(tmp_path, monkeypatch, text: str):
    monkeypatch.setenv("INGEST_RUNTIME", "lightrag.testing_index_runtime:runtime")
    working = tmp_path / "rag"
    inputs = tmp_path / "inputs"
    working.mkdir()
    inputs.mkdir()
    data = load_shell(working)
    data["kbs"] = [{"id": "kb-a", "owner_id": "user-a", "name": "kb-a", "settings": {}, "graph_config": {}}]
    save_shell(working, data)
    record = allocate_upload(data, "user-a", "kb-a", "note.txt")
    record["status"] = "queued"
    dest = inputs / record["storage_key"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = text.encode("utf-8")
    dest.write_bytes(payload)
    save_shell(working, data)
    store = SqliteStore(working / "product_jobs.sqlite")
    job = store.create_job({
        "job_type": "ingestion",
        "user_id": "user-a",
        "kb_id": "kb-a",
        "doc_id": record["doc_id"],
        "file_name": "note.txt",
        "file_path": record["storage_key"],
        "idempotency_key": content_key("user-a", "kb-a", payload),
        "input_snapshot": {"owner_id": "user-a", "storage_key": record["storage_key"], "chunk_size": 32, "chunk_overlap": 4},
    })
    done = run_ingestion(store, job, working, inputs, lambda: True)
    return working, record["doc_id"], done


def test_upload_uses_configured_embedding_and_text_grounded_graph(tmp_path, monkeypatch):
    text = "节温器导致水温升高"
    working, doc_id, done = _run(tmp_path, monkeypatch, text)
    assert done["status"] == "succeeded", done
    chunks = [row for row in _kv(working, "text_chunks").values() if row.get("doc_id") == doc_id]
    vectors = [row for row in _vdb_rows(working / "vdb_chunks.json") if row.get("doc_id") == doc_id]
    assert len(chunks) == len(vectors) == 1
    assert chunks[0]["evidence_ids"]
    import json
    from nano_vectordb.dbs import buffer_string_to_array

    raw = json.loads((working / "vdb_chunks.json").read_text(encoding="utf-8"))
    matrix = buffer_string_to_array(raw["matrix"]).reshape(-1, int(raw["embedding_dim"]))
    index = next(i for i, row in enumerate(raw["data"]) if row.get("doc_id") == doc_id)
    assert np.allclose(matrix[index], embed_vector(chunks[0]["index_text"]))
    assert chunks[0]["content"] in chunks[0]["index_text"]
    assert "parent_content" not in chunks[0]
    graph, _path = _graph(working)
    names = [str(attrs.get("name") or "") for _node, attrs in graph.nodes(data=True)]
    assert names
    assert all(name in text for name in names)
    assert f"{doc_id}-a" not in names and f"{doc_id}-b" not in names
    relations = [attrs for _src, _tgt, attrs in graph.edges(data=True)]
    assert relations
    assert all(attrs.get("relation_type") == "导致" for attrs in relations)
    assert all(attrs.get("doc_id") == doc_id for attrs in relations)


def test_fixed_cause_relation_is_not_stored_without_text(tmp_path, monkeypatch):
    text = "第一段\n\n第二段"
    working, doc_id, done = _run(tmp_path, monkeypatch, text)
    assert done["status"] == "succeeded", done
    graph, _path = _graph(working)
    names = [str(attrs.get("name") or "") for _node, attrs in graph.nodes(data=True)]
    assert names
    assert all(name in text for name in names)
    assert all(attrs.get("relation_type") != "导致" for _src, _tgt, attrs in graph.edges(data=True))
    assert f"{doc_id}-a" not in names and f"{doc_id}-b" not in names


def _open(tmp_path, monkeypatch):
    monkeypatch.setenv("INGEST_RUNTIME", "lightrag.testing_index_runtime:runtime")
    working = tmp_path / "rag"
    inputs = tmp_path / "inputs"
    working.mkdir()
    inputs.mkdir()
    data = load_shell(working)
    data["kbs"] = [{"id": "kb-a", "owner_id": "user-a", "name": "kb-a", "settings": {}, "graph_config": {}}]
    save_shell(working, data)
    store = SqliteStore(working / "product_jobs.sqlite")
    return working, inputs, store


def _put(working, inputs, store, text: str, name: str):
    data = load_shell(working)
    record = allocate_upload(data, "user-a", "kb-a", name)
    record["status"] = "queued"
    dest = inputs / record["storage_key"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = text.encode("utf-8")
    dest.write_bytes(payload)
    save_shell(working, data)
    job = store.create_job({
        "job_type": "ingestion",
        "user_id": "user-a",
        "kb_id": "kb-a",
        "doc_id": record["doc_id"],
        "file_name": name,
        "file_path": record["storage_key"],
        "idempotency_key": content_key("user-a", "kb-a", payload) + ":" + name,
        "input_snapshot": {"owner_id": "user-a", "storage_key": record["storage_key"], "chunk_size": 32, "chunk_overlap": 4},
    })
    done = run_ingestion(store, job, working, inputs, lambda: True)
    return record["doc_id"], done


def test_shared_entity_keeps_other_document_until_last_source_is_gone(tmp_path, monkeypatch):
    text = "节温器导致水温升高"
    working, inputs, store = _open(tmp_path, monkeypatch)
    first, done_first = _put(working, inputs, store, text, "a.txt")
    second, done_second = _put(working, inputs, store, text, "b.txt")
    assert done_first["status"] == "succeeded", done_first
    assert done_second["status"] == "succeeded", done_second
    first_ir = working / "document_ir" / f"{first}.json"
    second_ir = working / "document_ir" / f"{second}.json"
    assert first_ir.is_file() and second_ir.is_file()
    graph, _path = _graph(working)
    relations = [attrs for _src, _tgt, attrs in graph.edges(data=True)]
    assert len(relations) == 1
    assert set(split_values(relations[0]["source_id"])) == {f"{first}-c0001", f"{second}-c0001"}
    assert set(split_values(relations[0]["doc_id"])) == {first, second}
    entities = _vdb_rows(working / "vdb_entities.json")
    relation_vectors = _vdb_rows(working / "vdb_relationships.json")
    assert len(entities) == 2
    assert len(relation_vectors) == 1
    assert all(set(split_values(row["source_id"])) == {f"{first}-c0001", f"{second}-c0001"} for row in entities)
    assert set(split_values(relation_vectors[0]["source_id"])) == {f"{first}-c0001", f"{second}-c0001"}

    data = load_shell(working)
    remove_documents(working, inputs, data, "kb-a", [first])
    save_shell(working, data)
    assert not first_ir.exists()
    assert second_ir.is_file()
    graph, _path = _graph(working)
    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1
    kept = [attrs for _src, _tgt, attrs in graph.edges(data=True)][0]
    assert split_values(kept["source_id"]) == [f"{second}-c0001"]
    assert split_values(kept["doc_id"]) == [second]
    entities = _vdb_rows(working / "vdb_entities.json")
    relation_vectors = _vdb_rows(working / "vdb_relationships.json")
    assert len(entities) == 2
    assert len(relation_vectors) == 1
    assert all(split_values(row["source_id"]) == [f"{second}-c0001"] for row in entities)
    assert split_values(relation_vectors[0]["source_id"]) == [f"{second}-c0001"]
    chunks = [row for row in _vdb_rows(working / "vdb_chunks.json") if row.get("doc_id") == first]
    assert chunks == []

    data = load_shell(working)
    remove_documents(working, inputs, data, "kb-a", [second])
    save_shell(working, data)
    assert not second_ir.exists()
    graph, _path = _graph(working)
    assert graph.number_of_nodes() == 0
    assert graph.number_of_edges() == 0
    assert _vdb_rows(working / "vdb_entities.json") == []
    assert _vdb_rows(working / "vdb_relationships.json") == []
    report = deep_check(working, "kb-a", inputs, doc_ids=[first, second])
    assert report["stores"]["document_ir"]["count"] == 0
    assert report["passed"] is True


def test_parallel_vector_and_graph_writes_keep_every_row(tmp_path):
    working = tmp_path / "rag"
    working.mkdir()
    errors: list[BaseException] = []
    names = ("vdb_chunks.json", "vdb_entities.json", "vdb_relationships.json")

    def write_vector(index: int) -> None:
        path = working / names[index % 3]
        row_id = f"row-{index}"

        def editor(db) -> None:
            db.upsert([{
                "__id__": row_id,
                "__vector__": np.ones(4, dtype=np.float32),
                "source_id": row_id,
            }])

        try:
            update_vector_store(path, 4, editor)
        except BaseException as exc:
            errors.append(exc)

    def write_node(index: int) -> None:
        def editor(graph) -> None:
            graph.add_node(f"n{index}", name=f"n{index}", source_id=f"c{index}")

        try:
            rewrite_graph(working / "graph_chunk_entity_relation.graphml", editor)
        except BaseException as exc:
            errors.append(exc)

    threads = []
    for index in range(9):
        threads.append(threading.Thread(target=write_vector, args=(index,)))
        threads.append(threading.Thread(target=write_node, args=(index,)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    for name in names:
        payload = json.loads((working / name).read_text(encoding="utf-8"))
        found = {row["__id__"] for row in payload["data"]}
        expected = {f"row-{index}" for index in range(9) if names[index % 3] == name}
        assert found == expected
    graph, _path = _graph(working)
    assert {node for node in graph.nodes} == {f"n{index}" for index in range(9)}
