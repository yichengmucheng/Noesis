# -*- coding: utf-8 -*-
import asyncio

from lightrag.product_db import SqliteStore
from lightrag.product_ids import entity_id
from lightrag.product_ingest import content_key, run_ingestion
from lightrag.product_storage import _graph, _kv, load_shell, save_shell
from lightrag.product_uploads import allocate_upload
from lightrag.search_runtime import run_search_test


def _prepare(tmp_path, monkeypatch):
    monkeypatch.setenv("INGEST_RUNTIME", "lightrag.testing_index_runtime:runtime")
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("RETRIEVAL_PROFILE", "ci")
    working = tmp_path / "rag"
    inputs = tmp_path / "inputs"
    working.mkdir()
    inputs.mkdir()
    data = load_shell(working)
    data["kbs"] = [
        {
            "id": "kb-a",
            "owner_id": "user-a",
            "name": "主库",
            "settings": {},
            "graph_config": {},
        },
        {
            "id": "kb-b",
            "owner_id": "user-a",
            "name": "副库",
            "settings": {},
            "graph_config": {},
        },
    ]
    save_shell(working, data)
    store = SqliteStore(working / "product_jobs.sqlite")
    return working, inputs, store


def _put(working, inputs, store, kb_id: str, name: str, text: str):
    data = load_shell(working)
    record = allocate_upload(data, "user-a", kb_id, name)
    record["status"] = "queued"
    dest = inputs / record["storage_key"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = text.encode("utf-8")
    dest.write_bytes(payload)
    save_shell(working, data)
    job = store.create_job(
        {
            "job_type": "ingestion",
            "user_id": "user-a",
            "kb_id": kb_id,
            "doc_id": record["doc_id"],
            "file_name": name,
            "file_path": record["storage_key"],
            "idempotency_key": content_key("user-a", kb_id, payload) + ":" + name,
            "input_snapshot": {
                "owner_id": "user-a",
                "storage_key": record["storage_key"],
                "chunk_size": 128,
                "chunk_overlap": 0,
            },
        }
    )
    done = run_ingestion(store, job, working, inputs, lambda: True)
    assert done["status"] == "succeeded", done
    return record["doc_id"]


class _Chunks:
    def __init__(self, rows):
        self.rows = rows

    async def get_by_id(self, chunk_id):
        return self.rows.get(chunk_id)


class _GraphAdapter:
    def __init__(self, graph):
        self.graph = graph

    async def get_node(self, node_id):
        if not self.graph.has_node(node_id):
            return None
        return dict(self.graph.nodes[node_id])

    async def get_edge(self, src, tgt):
        if not self.graph.has_edge(src, tgt):
            return None
        return dict(self.graph.edges[src, tgt])

    async def get_node_edges(self, node_id):
        if not self.graph.has_node(node_id):
            return None
        return list(self.graph.edges(node_id))

    async def get_all_nodes(self):
        return [
            {"id": node_id, **attrs} for node_id, attrs in self.graph.nodes(data=True)
        ]


class _Entities:
    def __init__(self, rows):
        self.rows = rows

    async def query(self, query, top_k=10):
        del query, top_k
        return self.rows


class _Rag:
    def __init__(self, chunks, graph, entities):
        self.text_chunks = _Chunks(chunks)
        self.chunk_entity_relation_graph = _GraphAdapter(graph)
        self.entities_vdb = _Entities(entities)


def test_two_hop_path_uses_entity_id_and_keeps_other_kb_out(tmp_path, monkeypatch):
    working, inputs, store = _prepare(tmp_path, monkeypatch)
    doc_a = _put(working, inputs, store, "kb-a", "a.txt", "光合作用产生葡萄糖")
    doc_b = _put(working, inputs, store, "kb-a", "b.txt", "葡萄糖为细胞呼吸提供能量")
    doc_other = _put(working, inputs, store, "kb-b", "c.txt", "葡萄糖导致甜味增加")
    graph, _path = _graph(working)

    def node_of(kb_id: str, name: str) -> str:
        found = [
            node_id
            for node_id, attrs in graph.nodes(data=True)
            if attrs.get("name") == name and attrs.get("kb_id") == kb_id
        ]
        assert len(found) == 1, (kb_id, name, found)
        return found[0]

    assert node_of("kb-a", "光合作用") == entity_id("kb-a", "概念", "光合作用")
    assert node_of("kb-a", "葡萄糖") == entity_id("kb-a", "概念", "葡萄糖")
    assert node_of("kb-b", "葡萄糖") == entity_id("kb-b", "部件", "葡萄糖")
    chunks = _kv(working, "text_chunks")
    entities = [
        {
            "entity_id": node_of("kb-a", "光合作用"),
            "id": node_of("kb-a", "光合作用"),
            "entity_name": "光合作用",
            "distance": 0.72,
            "kb_id": "kb-a",
            "owner_id": "user-a",
        },
        {
            "entity_id": entity_id("kb-b", "部件", "葡萄糖"),
            "id": entity_id("kb-b", "部件", "葡萄糖"),
            "entity_name": "葡萄糖",
            "distance": 0.99,
            "kb_id": "kb-b",
            "owner_id": "user-a",
        },
    ]
    rag = _Rag(chunks, graph, entities)

    def file_in_kb(file_path, record=None):
        del file_path
        record = record or {}
        return "kb-a" in str(record.get("kb_id") or "").split("<SEP>")

    result = asyncio.run(
        run_search_test(
            rag,
            query="光合作用如何为细胞提供能量",
            mode="graph",
            top_k=8,
            ratio=0.5,
            enable_rerank=False,
            kb_chunks=list(chunks.values()),
            file_in_kb=file_in_kb,
            score_threshold=0.2,
            kb_id="kb-a",
            owner_id="user-a",
        )
    )
    cited = [item for item in result["chunks"] if item.get("unit_id")]
    documents = {item["document_id"] for item in cited}
    pairs = {(item["document_id"], item["unit_id"]) for item in cited}
    assert documents == {doc_a, doc_b}
    assert doc_other not in documents
    assert len(pairs) >= 2
    assert all(unit for _doc, unit in pairs)
    hops = [item for item in result["chunks"] if int(item.get("hop") or 0) >= 2]
    assert hops
    assert any(
        "光合作用" in item["path"]
        and "葡萄糖" in item["path"]
        and "能量" in item["path"]
        for item in hops
    )
    assert all("甜味" not in (item.get("path") or "") for item in result["chunks"])
    assert all(item.get("admission_score") is not None for item in result["chunks"])
    assert all(item.get("kb_id") in {"", "kb-a"} for item in result["chunks"])
