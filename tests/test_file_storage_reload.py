import json

import networkx as nx
import numpy as np
import pytest
from nano_vectordb import NanoVectorDB

from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage
from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.utils import EmbeddingFunc


async def _embed(texts, **_kwargs):
    return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


@pytest.mark.asyncio
async def test_file_storages_reload_updates_from_separate_worker(tmp_path):
    initialize_share_data()
    embedding = EmbeddingFunc(embedding_dim=2, func=_embed)
    config = {
        "working_dir": str(tmp_path),
        "embedding_batch_num": 10,
        "max_graph_nodes": 100,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.0},
    }

    kv = JsonKVStorage("reload_kv", "", config, embedding)
    vector = NanoVectorDBStorage(
        "reload_vectors", "", config, embedding, meta_fields={"content"}
    )
    graph = NetworkXStorage("reload_graph", "", config, embedding)
    await kv.initialize()
    await vector.initialize()
    await graph.initialize()

    kv_path = tmp_path / "kv_store_reload_kv.json"
    kv_path.write_text(
        json.dumps({"external": {"content": "worker value"}}), encoding="utf-8"
    )

    vector_path = tmp_path / "vdb_reload_vectors.json"
    external_vector = NanoVectorDB(2, storage_file=str(vector_path))
    external_vector.upsert(
        datas=[
            {
                "__id__": "external",
                "__vector__": np.asarray([1.0, 0.0], dtype=np.float32),
                "content": "worker vector",
            }
        ]
    )
    external_vector.save()

    graph_path = tmp_path / "graph_reload_graph.graphml"
    external_graph = nx.Graph()
    external_graph.add_node("external", description="worker node")
    nx.write_graphml(external_graph, graph_path)

    assert (await kv.get_by_id("external"))["content"] == "worker value"
    assert (await vector.get_by_id("external"))["content"] == "worker vector"
    assert (await graph.get_node("external"))["description"] == "worker node"
