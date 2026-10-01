# -*- coding: utf-8 -*-
import asyncio
import hashlib

import numpy as np

from lightrag.search_runtime import _retrieved_score, run_search_test
from lightrag.search_strategies import (
    absolute_relevance,
    bm25_rank,
    calibrate_bm25,
    calibrate_cosine,
    expand_paths,
    filter_by_threshold,
    reorder_by_rerank,
    weighted_fuse,
)

EDU = "教育背景是天津财经大学数据科学专业，教育背景包括本科阶段的统计和机器学习。"
COOK = "红烧肉需要先焯水，再加入酱油和冰糖慢炖四十分钟，红烧肉出锅前收汁。"
TRIP = "西湖位于杭州，春季适合沿苏堤骑行看柳树，西湖游船从断桥出发。"
EDU_QUERY = "教育背景是什么"
COOK_QUERY = "红烧肉需要先焯水再加入酱油"
TRIP_QUERY = "西湖位于杭州春季适合骑行"
FAR_QUERIES = ("量子计算芯片原理", "今天天气怎么样")


def test_missing_mix_score_does_not_pass_threshold():
    assert _retrieved_score({"score": None}, 0) is None
    assert _retrieved_score({"score": 0.48}, 0) == 0.48
    assert filter_by_threshold([{"chunk_id": "c", "score": None}], 0.2) == []
    kept = filter_by_threshold(
        [{"chunk_id": "c", "score": 0.48, "calibrated_score": 0.48}],
        0.2,
    )
    assert [item["chunk_id"] for item in kept] == ["c"]


def test_single_candidate_keeps_absolute_score():
    hits = [{"chunk_id": "only", "score": 0.39, "content": "简历"}]
    assert absolute_relevance(0.39) == 0.39
    fused = weighted_fuse(hits, [], 1)
    assert fused[0]["score"] == 0.39
    assert fused[0]["calibrated_score"] == 0.39
    assert filter_by_threshold(fused, 0.5) == []
    kept = filter_by_threshold(hits, 0.2)
    assert len(kept) == 1
    assert kept[0]["score"] == 0.39


def test_equal_scores_are_not_forced_through_threshold():
    hits = [
        {"chunk_id": "a", "score": 2.0, "content": "甲"},
        {"chunk_id": "b", "score": 2.0, "content": "乙"},
    ]
    assert filter_by_threshold(hits, 0.5) == []
    assert filter_by_threshold(
        [{"chunk_id": "a", "score": 0.15}, {"chunk_id": "b", "score": 0.15}],
        0.2,
    ) == []


def test_bm25_transform_is_monotonic_from_zero():
    samples = (0.99, 1.0, 1.0001, 2, 5, 12)
    converted = [calibrate_bm25(value) for value in samples]
    assert calibrate_bm25(0) == 0
    assert converted == sorted(converted)
    assert len(set(converted)) == len(converted)
    assert all(left < right for left, right in zip(converted, converted[1:]))
    assert calibrate_cosine(1.0) == 1.0
    assert calibrate_bm25(1.0) < calibrate_bm25(1.0001) < 1.0


def test_bm25_prefers_matching_fault_text():
    docs = [
        "节温器打不开，水温持续升高，需要检查节温器。",
        "风扇轴底座断裂，位于发动机前端。",
    ]
    scores = bm25_rank("节温器打不开", docs)
    assert scores[0] > scores[1]
    assert scores[1] == 0


def test_weighted_fuse_follows_ratio():
    vector_hits = [{"chunk_id": "v", "score": 0.9, "content": "向量"}]
    keyword_hits = [{"chunk_id": "k", "score": 12, "content": "关键词"}]
    vector_first = weighted_fuse(vector_hits, keyword_hits, 1)
    keyword_first = weighted_fuse(vector_hits, keyword_hits, 0)
    assert vector_first[0]["chunk_id"] == "v"
    assert keyword_first[0]["chunk_id"] == "k"
    both = weighted_fuse(
        [{"chunk_id": "a", "score": 1.0}, {"chunk_id": "b", "score": 0.2}],
        [{"chunk_id": "a", "score": 1.0}, {"chunk_id": "b", "score": 5.0}],
        0.5,
    )
    assert {item["chunk_id"] for item in both} == {"a", "b"}
    assert both[0]["source"] == "hybrid"


def test_expand_paths_two_hops():
    adjacency = {
        "密封圈": [("偏心", "导致")],
        "偏心": [("压缩不足", "导致")],
        "压缩不足": [],
    }
    rows = expand_paths([("密封圈", 1.0)], adjacency, hops=2)
    by_name = {row["name"]: row for row in rows}
    assert by_name["密封圈"]["hop"] == 0
    assert by_name["偏心"]["hop"] == 1
    assert by_name["压缩不足"]["hop"] == 2
    assert by_name["压缩不足"]["path"] == ["密封圈", "导致", "偏心", "导致", "压缩不足"]
    assert by_name["密封圈"]["score"] > by_name["压缩不足"]["score"]


def test_reorder_by_rerank():
    hits = [
        {"chunk_id": "a", "score": 0.2},
        {"chunk_id": "b", "score": 0.9},
    ]
    ordered = reorder_by_rerank(
        hits,
        [
            {"index": 1, "relevance_score": 0.15},
            {"index": 0, "relevance_score": 0.91},
        ],
    )
    assert [item["chunk_id"] for item in ordered] == ["b", "a"]
    assert ordered[0]["score"] == 0.15
    assert ordered[0]["rerank_score"] == 0.15
    assert ordered[0]["reranked"] is True
    assert filter_by_threshold(ordered, 0.2)[0]["chunk_id"] == "a"


def test_threshold_uses_absolute_scores():
    calibrated = [{"chunk_id": "high", "score": 0.82}, {"chunk_id": "low", "score": 0.11}]
    kept = filter_by_threshold(calibrated, 0.3)
    assert [item["chunk_id"] for item in kept] == ["high"]
    raw = [{"chunk_id": "strong", "score": 12}, {"chunk_id": "weak", "score": 1.2}]
    ranked = filter_by_threshold(raw, 0.3)
    assert [item["chunk_id"] for item in ranked] == ["strong"]
    assert filter_by_threshold(calibrated, 0) == calibrated


def _embed_one(text: str, dim: int = 16384) -> np.ndarray:
    from lightrag.search_strategies import tokenize

    vector = np.zeros(dim, dtype=np.float32)
    for token in tokenize(text):
        slot = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % dim
        vector[slot] += 1.0
    norm = float(np.linalg.norm(vector)) or 1.0
    return vector / norm


async def _embed(texts: list[str]) -> np.ndarray:
    return np.vstack([_embed_one(text) for text in texts])


class _Chunks:
    def __init__(self, rows: list[dict]):
        self.rows = {row["chunk_id"]: row for row in rows}

    async def get_by_id(self, chunk_id: str):
        return self.rows.get(chunk_id)


class _VectorDB:
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.embedding_func = _embed

    async def query(self, query: str, top_k: int = 10, query_embedding=None):
        query_vector = _embed_one(query)
        scored = []
        for row in self.rows:
            cosine = float(np.dot(query_vector, row["vector"]))
            scored.append({
                "id": row["chunk_id"],
                "__id__": row["chunk_id"],
                "content": row["content"],
                "file_path": row["file_path"],
                "document_id": row.get("document_id"),
                "unit_id": row.get("unit_id"),
                "kb_id": row.get("kb_id"),
                "owner_id": row.get("owner_id"),
                "distance": cosine,
            })
        scored.sort(key=lambda item: item["distance"], reverse=True)
        return scored[:top_k]


class _Rag:
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.chunks_vdb = _VectorDB(rows)
        self.text_chunks = _Chunks(rows)
        self.embedding_func = _embed

    async def aquery_data(self, query: str, param=None):
        del query, param
        return {
            "chunks": [
                {
                    "chunk_id": row["chunk_id"],
                    "content": row["content"],
                    "file_path": row["file_path"],
                    "score": None,
                }
                for row in self.rows
            ],
            "entities": [],
            "relationships": [],
        }

    async def llm_model_func(self, prompt: str, **kwargs):
        del kwargs
        return "可能相关的说明：" + prompt[-40:]


def _library(*pairs: tuple[str, str]) -> list[dict]:
    rows = []
    for chunk_id, content in pairs:
        rows.append({
            "chunk_id": chunk_id,
            "content": content,
            "doc_name": chunk_id + ".txt",
            "file_path": chunk_id + ".txt",
            "document_id": chunk_id.split("-")[0],
            "unit_id": "t0001",
            "kb_id": "kb-test",
            "owner_id": "user-test",
            "vector": _embed_one(content),
        })
    return rows


def _search(rows: list[dict], query: str, mode: str) -> dict:
    import os

    os.environ["RETRIEVAL_PROFILE"] = "ci"
    rag = _Rag(rows)

    async def run():
        return await run_search_test(
            rag,
            query=query,
            mode=mode,
            top_k=5,
            ratio=0.5,
            enable_rerank=False,
            kb_chunks=rows,
            file_in_kb=lambda *_args, **_kwargs: True,
            score_threshold=0.2,
            kb_id="kb-test",
            owner_id="user-test",
        )

    return asyncio.run(run())


def test_single_resume_chunk_rejects_unrelated_queries():
    rows = _library(("resume-c0001", EDU))
    for mode in ("vector", "keyword", "hybrid", "mix", "tree"):
        matched = _search(rows, EDU_QUERY, mode)
        assert [item["chunk_id"] for item in matched["chunks"]] == ["resume-c0001"], mode
        assert EDU in matched["chunks"][0]["content"]
        assert matched["chunks"][0]["calibrated_score"] < 1.0
        assert matched["chunks"][0]["score"] != 1.0
        for query in FAR_QUERIES:
            missed = _search(rows, query, mode)
            assert missed["chunks"] == [], (mode, query, missed["chunks"])


def test_different_questions_hit_different_chunks():
    rows = _library(("edu-c0001", EDU), ("cook-c0001", COOK), ("trip-c0001", TRIP))
    expected = (
        (EDU_QUERY, "edu-c0001", EDU),
        (COOK_QUERY, "cook-c0001", COOK),
        (TRIP_QUERY, "trip-c0001", TRIP),
    )
    for mode in ("vector", "keyword", "hybrid", "mix", "tree"):
        seen = []
        for query, chunk_id, content in expected:
            result = _search(rows, query, mode)
            ids = [item["chunk_id"] for item in result["chunks"]]
            assert ids == [chunk_id], (mode, query, result["chunks"])
            assert content in result["chunks"][0]["content"]
            seen.append(chunk_id)
        assert seen == ["edu-c0001", "cook-c0001", "trip-c0001"]
