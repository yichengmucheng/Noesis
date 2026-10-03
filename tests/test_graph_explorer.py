import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient


class FakeGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    async def get_all_nodes(self):
        return list(self.nodes)

    async def get_all_edges(self):
        return list(self.edges)

    async def get_node(self, node_id):
        return next((dict(node) for node in self.nodes if node["id"] == node_id), None)

    async def get_node_edges(self, node_id):
        return [
            (edge["source"], edge["target"])
            for edge in self.edges
            if node_id in {edge["source"], edge["target"]}
        ]

    async def get_edge(self, source, target):
        return next(
            (
                dict(edge)
                for edge in self.edges
                if edge["source"] == source and edge["target"] == target
            ),
            None,
        )


def _client(tmp_path, monkeypatch):
    working = tmp_path / "rag"
    inputs = tmp_path / "inputs"
    working.mkdir()
    inputs.mkdir()
    monkeypatch.setenv("PRODUCT_AUTH", "0")
    monkeypatch.setenv("APP_ENV", "development")

    graph = FakeGraph()

    class Rag:
        def __init__(self):
            self.working_dir = str(working)
            self.addon_params = {}
            self.chunk_entity_relation_graph = graph

        async def get_docs_by_status(self, _status):
            return {}

    class Docs:
        input_dir = str(inputs)

        @staticmethod
        def is_supported_file(_name):
            return True

    sys.argv = [sys.argv[0]]
    from lightrag.api.routers.product_shell import create_product_shell_routes

    app = FastAPI()
    app.include_router(create_product_shell_routes(Rag(), Docs()))
    return TestClient(app), graph


def test_graph_explorer_uses_names_but_queries_by_entity_id(tmp_path, monkeypatch):
    client, graph = _client(tmp_path, monkeypatch)
    kb_id = client.post(
        "/api/v1/kb",
        json={"name": "个人资料"},
        headers={"X-KB-Request": "1"},
    ).json()["id"]
    graph.nodes = [
        {
            "id": "ent-a1",
            "name": "百事通",
            "entity_type": "项目",
            "kb_id": kb_id,
            "source_id": "chunk-a",
        },
        {
            "id": "ent-a2",
            "name": "MCP",
            "entity_type": "技术",
            "kb_id": kb_id,
            "source_id": "chunk-a",
        },
        {
            "id": "ent-a3",
            "name": "外部工具",
            "entity_type": "概念",
            "kb_id": kb_id,
            "source_id": "chunk-b",
        },
        {"id": "ent-orphan", "name": "孤立抽取", "entity_type": "短语", "kb_id": kb_id},
        {"id": "ent-deadbeef", "entity_type": "短语", "kb_id": kb_id},
        {
            "id": "ent-other",
            "name": "其他知识库",
            "entity_type": "项目",
            "kb_id": "other-kb",
        },
    ]
    graph.edges = [
        {"source": "ent-a1", "target": "ent-a2", "keywords": "使用", "kb_id": kb_id},
        {"source": "ent-a2", "target": "ent-a3", "keywords": "连接", "kb_id": kb_id},
        {
            "source": "ent-a1",
            "target": "ent-other",
            "keywords": "泄漏",
            "kb_id": "other-kb",
        },
    ]

    overview = client.get("/api/v1/graph/subgraph", params={"kb_id": kb_id}).json()
    assert overview["mode"] == "overview"
    assert {node["label"] for node in overview["nodes"]} == {
        "百事通",
        "MCP",
        "外部工具",
    }
    assert all(not node["label"].startswith("ent-") for node in overview["nodes"])
    assert "孤立抽取" not in {node["label"] for node in overview["nodes"]}
    assert "其他知识库" not in {node["label"] for node in overview["nodes"]}

    focused = client.get(
        "/api/v1/graph/subgraph",
        params={"kb_id": kb_id, "entity_id": "ent-a1", "hops": 2},
    ).json()
    assert focused["mode"] == "focused"
    assert focused["center_id"] == "ent-a1"
    assert {node["label"] for node in focused["nodes"]} == {"百事通", "MCP", "外部工具"}

    neighbors = client.get(
        "/api/v1/graph/entities/ent-a1/neighbors",
        params={"kb_id": kb_id},
    ).json()
    assert neighbors["entity"]["name"] == "百事通"
    assert neighbors["neighbors"][0]["neighbor_id"] == "ent-a2"
    assert neighbors["neighbors"][0]["neighbor_name"] == "MCP"

    relations = client.get("/api/v1/graph/relations", params={"kb_id": kb_id}).json()
    assert relations["items"][0]["source"] in {"百事通", "MCP"}
    assert all(not row["source"].startswith("ent-") for row in relations["items"])
    assert all(
        "其他知识库" not in {row["source"], row["target"]} for row in relations["items"]
    )

    entities = client.get("/api/v1/graph/entities", params={"kb_id": kb_id}).json()
    unnamed = next(item for item in entities["items"] if item["id"] == "ent-deadbeef")
    assert unnamed["name"] == "未命名实体"


def test_graph_search_resolves_display_name_and_keeps_internal_id(
    tmp_path, monkeypatch
):
    client, graph = _client(tmp_path, monkeypatch)
    kb_id = client.post(
        "/api/v1/kb",
        json={"name": "个人资料"},
        headers={"X-KB-Request": "1"},
    ).json()["id"]
    graph.nodes = [
        {"id": "ent-1", "name": "知识库", "entity_type": "概念", "kb_id": kb_id},
        {"id": "ent-2", "name": "资料检索", "entity_type": "能力", "kb_id": kb_id},
    ]
    graph.edges = [
        {"source": "ent-1", "target": "ent-2", "keywords": "支持", "kb_id": kb_id},
    ]

    response = client.get(
        "/api/v1/graph/subgraph",
        params={"kb_id": kb_id, "entity_name": "知识库"},
    ).json()
    assert response["center_id"] == "ent-1"
    assert response["nodes"][0]["id"] == "ent-1"
    assert response["nodes"][0]["label"] == "知识库"
