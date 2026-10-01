# -*- coding: utf-8 -*-
import asyncio
import json

import numpy as np

from lightrag.chunk_hierarchy import assign_parents, count_tokens, split_token_windows
from lightrag.index_manifest import compatibility_error, rebuild_index, run_index_rebuild, save_manifest
from lightrag.product_db import SqliteStore
from lightrag.product_document_ir import DocumentIR, SourceUnit
from lightrag.product_parse import chunks_from_document
from lightrag.product_storage import _save_kv
from lightrag.retrieval_admission import annotate_admission
from lightrag.retrieval_orchestrator import (
    aggregate_parents,
    expand_admitted_bundles,
    final_child_limit,
    fuse_ranked_groups,
    path_bundles,
)
from lightrag.search_runtime import _rerank, run_search_test
from lightrag.search_strategies import filter_by_threshold


def test_query_routes_do_not_call_model_for_exact_or_factual():
    calls = {"n": 0}

    async def llm(prompt: str) -> str:
        calls["n"] += 1
        raise AssertionError(prompt)

    async def run():
        exact = await __import__("lightrag.query_plan", fromlist=["build_query_plan"]).build_query_plan(
            "合同编号 HT-2025-1107", llm=llm
        )
        dated = await __import__("lightrag.query_plan", fromlist=["build_query_plan"]).build_query_plan(
            "2024年3月的会议纪要放在哪", llm=llm
        )
        named = await __import__("lightrag.query_plan", fromlist=["build_query_plan"]).build_query_plan(
            "林知夏是谁", llm=llm
        )
        factual = await __import__("lightrag.query_plan", fromlist=["build_query_plan"]).build_query_plan(
            "光合作用产生葡萄糖", llm=llm
        )
        vague = await __import__("lightrag.query_plan", fromlist=["build_query_plan"]).build_query_plan("这个怎么办")
        compared = await __import__("lightrag.query_plan", fromlist=["build_query_plan"]).build_query_plan(
            "比较密封圈老化和安装偏心对泄漏的影响"
        )
        coref = await __import__("lightrag.query_plan", fromlist=["build_query_plan"]).build_query_plan(
            "它要停机吗", history=["密封圈的更换步骤"], llm=llm
        )
        weak = await __import__("lightrag.query_plan", fromlist=["build_query_plan"]).build_query_plan(
            "光合作用的原理是什么",
            llm=lambda prompt: json.dumps(
                {
                    "standalone_query": "光合作用的原理是什么",
                    "route": "hyde",
                    "variants": ["光合作用的原理是什么"],
                    "hyde_text": "叶绿体利用光能合成葡萄糖",
                    "subquestions": [],
                    "reason": "首轮召回弱",
                },
                ensure_ascii=False,
            ),
            first_pass_weak=True,
        )
        return exact, dated, named, factual, vague, compared, coref, weak

    exact, dated, named, factual, vague, compared, coref, weak = asyncio.run(run())
    assert calls["n"] == 0
    assert exact["route"] == "exact" and exact["variants"] == ["合同编号 HT-2025-1107"] and exact["hyde_text"] == ""
    assert dated["route"] == "exact" and dated["llm_used"] is False
    assert named["route"] == "exact"
    assert factual["route"] == "factual" and factual["llm_used"] is False and factual["variants"] == ["光合作用产生葡萄糖"]
    assert vague["fallback_reason"] == "llm_unavailable" and vague["variants"] == ["这个怎么办"]
    assert compared["route"] == "subquestions"
    assert 2 <= len(compared["subquestions"]) <= 4
    assert "密封圈的更换步骤" in coref["standalone_query"]
    assert weak["route"] == "hyde"
    assert weak["hyde_text"] not in weak["variants"]
    assert weak["llm_used"] is True
    assert len(weak["variants"]) <= 3


def test_invalid_llm_plan_falls_back_to_original_query():
    async def llm(prompt: str) -> str:
        del prompt
        return "不是 JSON"

    plan = asyncio.run(__import__("lightrag.query_plan", fromlist=["build_query_plan"]).build_query_plan("这个怎么办", llm=llm))
    assert plan["variants"] == ["这个怎么办"]
    assert plan["hyde_text"] == ""
    assert plan["fallback_reason"] == "llm_invalid"


def test_parent_child_citation_points_at_the_small_chunk():
    body = "甲" * 500
    document = DocumentIR(
        document_id="doc-h",
        version_id="v1",
        source_name="手册.md",
        source_type="markdown",
        checksum="abc",
        units=[SourceUnit(unit_id="s1", unit_type="section", content=body, section_path=["冷却系统"])],
    )
    chunks = chunks_from_document(document, "user-a", "kb-a", "doc-h", chunk_size=250, chunk_overlap=45)
    children, parents = assign_parents(chunks, "手册.md", "doc-h")
    assert parents
    assert all("parent_content" not in child for child in children)
    assert all(child["parent_id"] == parents[0]["parent_id"] for child in children)
    assert parents[0]["content"] not in {child["content"] for child in children} or len(children) == 1
    for child in children:
        assert child["index_text"].startswith("手册.md")
        assert "冷却系统" in child["index_text"]
        assert count_tokens(child["content"]) <= 350
        assert child["evidence_refs"][0]["unit_id"] == "s1"
        assert child["chunk_id"] and child["evidence_ids"]
    assert parents[0]["token_count"] <= 1500
    windows = split_token_windows("乙" * 800)
    assert all(180 <= count_tokens(item) <= 350 for item in windows[:-1])
    assert count_tokens(windows[-1]) <= 350
    selected, contexts = aggregate_parents(children, {item["parent_id"]: item for item in parents}, max_children=8)
    assert contexts[0]["chunk_ids"] == [item["chunk_id"] for item in selected]
    assert selected[0]["content"] != contexts[0]["content"] or len(children) == 1
    assert "parent_content" not in selected[0]


def test_rerank_pool_is_independent_of_final_top_n():
    hits = [{"chunk_id": f"c{index}", "content": f"片段{index}", "doc_name": "手册.md", "raw_score": 0.2} for index in range(20)]
    seen = {}

    async def fake(**kwargs):
        seen["documents"] = len(kwargs["documents"])
        seen["top_n"] = kwargs["top_n"]
        return [{"index": index, "relevance_score": 0.8} for index in range(len(kwargs["documents"]))]

    ordered, info = asyncio.run(_rerank("原始问题", hits, top_k=5, pool_size=40, rerank_func=fake))
    assert seen["documents"] == 20
    assert seen["top_n"] == 20
    assert info["pool_size"] == 40
    assert len(ordered) == 20
    assert final_child_limit(5) == 5
    assert final_child_limit(10) == 8


def test_path_bundle_keeps_bridge_and_rejects_low_path(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_PROFILE", "ci")
    monkeypatch.setenv("APP_ENV", "development")
    bridge = {
        "chunk_id": "doc-a-c0001",
        "content": "葡萄糖是中间产物",
        "path": "光合作用 → 葡萄糖",
        "path_id": "光合作用 → 葡萄糖",
        "hop": 1,
        "source_id": "doc-a-c0001",
        "document_id": "doc-a",
        "unit_id": "t0001",
        "owner_id": "user-a",
        "kb_id": "kb-a",
        "evidence_refs": [{"chunk_id": "doc-a-c0001", "unit_id": "t0001", "document_id": "doc-a"}],
    }
    answer = {
        "chunk_id": "doc-b-c0001",
        "content": "葡萄糖为细胞呼吸提供能量",
        "path": "光合作用 → 葡萄糖 → 能量",
        "path_id": "光合作用 → 葡萄糖 → 能量",
        "hop": 2,
        "source_id": "doc-b-c0001",
        "document_id": "doc-b",
        "unit_id": "t0001",
        "owner_id": "user-a",
        "kb_id": "kb-a",
        "evidence_refs": [{"chunk_id": "doc-b-c0001", "unit_id": "t0001", "document_id": "doc-b"}],
    }
    bundles = path_bundles([bridge, answer])
    assert len(bundles) == 1
    assert {item["chunk_id"] for item in bundles[0]["members"]} == {"doc-a-c0001", "doc-b-c0001"}
    low = dict(bundles[0])
    low.update({"rerank_score": 0.01, "reranked": True, "score": 0.01, "raw_score": 0.9})
    annotate_admission(low, "graph")
    assert filter_by_threshold([low], 0.2) == []
    high = dict(bundles[0])
    high.update({"rerank_score": 0.8, "reranked": True, "score": 0.8, "raw_score": 0.4})
    annotate_admission(high, "graph")
    assert filter_by_threshold([high], 0.2)
    expanded = expand_admitted_bundles([high])
    assert [item["chunk_id"] for item in expanded] == ["doc-a-c0001", "doc-b-c0001"]
    assert expanded[0]["evidence_refs"][0]["unit_id"] == "t0001"
    fused = fuse_ranked_groups(
        [
            [{"chunk_id": "direct", "raw_score": 0.2, "rerank_score": 0.9}],
            [{"chunk_id": "bridge", "raw_score": 9.0, "rerank_score": 0.2}],
        ]
    )
    assert {item["chunk_id"] for item in fused} == {"direct", "bridge"}
    assert abs(fused[0]["rrf_score"] - (1 / 61)) < 1e-9


def test_changed_reranker_does_not_reuse_old_floor(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("RETRIEVAL_PROFILE", "qwen3-embedding-4b")
    monkeypatch.setenv("RERANK_MODEL", "some-other-reranker")
    monkeypatch.setenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B")
    monkeypatch.setenv("EMBEDDING_DIM", "1024")
    item = {"reranked": True, "rerank_score": 0.0115, "score": 0.0115, "ranking_score": 0.0115}
    annotate_admission(item, "vector")
    assert item["admission_score"] is None
    assert item["admission_signal"] == "rerank_rebinding_required"


def test_model_mismatch_blocks_old_vectors(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B")
    monkeypatch.setenv("EMBEDDING_DIM", "1024")
    monkeypatch.setenv("EMBEDDING_BINDING", "siliconflow")
    monkeypatch.setenv("EMBEDDING_INSTRUCTION", "")
    monkeypatch.setenv("RETRIEVAL_PROFILE", "ci")
    working = tmp_path / "rag"
    working.mkdir()
    (working / "vdb_chunks.json").write_text('{"embedding_dim": 1024, "data": []}', encoding="utf-8")
    save_manifest(working, 1024, ["doc-1-c0001"])
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")

    class Boom:
        def __getattr__(self, name):
            raise AssertionError(name)

    result = asyncio.run(
        run_search_test(
            Boom(),
            query="光合作用产生葡萄糖",
            mode="vector",
            top_k=5,
            ratio=0.5,
            enable_rerank=False,
            kb_chunks=[],
            file_in_kb=lambda *_args, **_kwargs: True,
            working_dir=working,
        )
    )
    assert result["status"] == "index_rebuild_required"
    assert result["chunks"] == []
    assert compatibility_error(working)["status"] == "index_rebuild_required"


def test_rebuild_keeps_old_index_until_validation_passes(tmp_path, monkeypatch):
    monkeypatch.setenv("INGEST_RUNTIME", "lightrag.testing_index_runtime:runtime")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embed")
    monkeypatch.setenv("EMBEDDING_DIM", "32")
    monkeypatch.setenv("EMBEDDING_BINDING", "")
    monkeypatch.setenv("EMBEDDING_INSTRUCTION", "")
    working = tmp_path / "rag"
    working.mkdir()
    _save_kv(
        working,
        "text_chunks",
        {
            "doc-1-c0001": {
                "chunk_id": "doc-1-c0001",
                "source_id": "doc-1-c0001",
                "content": "节温器打不开",
                "index_text": "手册.md\n节温器打不开",
                "doc_id": "doc-1",
                "kb_id": "kb-a",
                "owner_id": "user-a",
                "user_id": "user-a",
            }
        },
    )
    (working / "vdb_chunks.json").write_text("OLD", encoding="utf-8")

    def fail(texts):
        del texts
        raise RuntimeError("重嵌失败")

    try:
        rebuild_index(working, fail)
    except RuntimeError:
        pass
    assert (working / "vdb_chunks.json").read_text(encoding="utf-8") == "OLD"

    store = SqliteStore(working / "jobs.sqlite")
    store.create_job(
        {
            "job_type": "index_rebuild",
            "user_id": "user-a",
            "kb_id": "kb-a",
            "doc_id": "corpus",
            "idempotency_key": "rebuild-1",
            "max_attempts": 3,
        }
    )
    acquired = store.acquire("worker-1")
    assert acquired["job_type"] == "index_rebuild"

    async def fail_embed(texts):
        del texts
        raise RuntimeError("重嵌失败")

    monkeypatch.setattr("lightrag.product_index.embed_texts", fail_embed)
    failed = run_index_rebuild(store, acquired, working)
    assert failed["status"] == "failed"
    assert (working / "vdb_chunks.json").read_text(encoding="utf-8") == "OLD"
    retried = store.request_retry(failed["job_id"], "user-a")
    assert retried["status"] == "queued"
    again = store.acquire("worker-1")

    async def embed(texts):
        from lightrag.testing_index_runtime import embed_vector

        return np.vstack([embed_vector(text) for text in texts])

    monkeypatch.setattr("lightrag.product_index.embed_texts", embed)
    done = run_index_rebuild(store, again, working)
    assert done["status"] == "succeeded", done
    payload = json.loads((working / "vdb_chunks.json").read_text(encoding="utf-8"))
    assert payload["embedding_dim"] == 32
    assert len(payload["data"]) == 1
    assert payload["data"][0]["owner_id"] == "user-a"
    manifest = json.loads((working / "index_manifest.json").read_text(encoding="utf-8"))
    assert manifest["embedding_model"] == "test-embed"
    assert manifest["embedding_dimension"] == 32
    assert compatibility_error(working) is None
