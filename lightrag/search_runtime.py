"""检索测试的六种模式。重排模型只使用运行配置里的 RERANK_MODEL。"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Callable

from lightrag.base import QueryParam
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.rerank import siliconflow_rerank
from lightrag.search_strategies import (
    calibrate_bm25,
    calibrate_cosine,
    bm25_rank,
    expand_paths,
    filter_by_threshold,
    reorder_by_rerank,
    weighted_fuse,
)
from lightrag.utils import logger

_HYDE_PROMPT = """你是个人知识库助手。根据下面的问题，写一段可能出现在用户资料里的说明。
只输出这段说明，80到160字。不要写“假设”“问题是”。

问题：{query}
"""


def _runtime_fields(working_dir: Path | None) -> dict[str, Any]:
    version = ""
    if working_dir is not None:
        from lightrag.index_manifest import load_manifest

        manifest = load_manifest(Path(working_dir))
        version = str((manifest or {}).get("corpus_revision") or "")
    return {
        "index_version": version,
        "embedding_model": os.getenv("EMBEDDING_MODEL") or "",
        "rerank_model": os.getenv("RERANK_MODEL") or "",
    }


def _filename(path: str) -> str:
    if not path:
        return ""
    first = str(path).split(GRAPH_FIELD_SEP)[0]
    return Path(first.replace("\\", "/")).name


def _retrieved_score(raw: dict[str, Any] | None, index: int = 0) -> float | None:
    """只读取已经存在的分数。缺分数返回 None，不用名次伪造相关度。index 不再参与计算。"""
    del index
    source = raw or {}
    for key in ("rerank_score", "vector_score", "score", "distance"):
        value = source.get(key)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _cosine(left: Any, right: Any) -> float | None:
    import numpy as np

    first = np.asarray(left, dtype=np.float64).reshape(-1)
    second = np.asarray(right, dtype=np.float64).reshape(-1)
    if first.size == 0 or first.shape != second.shape:
        return None
    denom = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denom <= 0:
        return 0.0
    return float(np.dot(first, second) / denom)


async def _query_cosine(rag, query: str, content: str) -> float | None:
    """用当前 embedding 重算问题和正文的余弦相似度。NanoVectorDB 的 distance 也是这个值。"""
    if not content.strip():
        return None
    embed = getattr(rag, "embedding_func", None)
    if embed is None:
        store = getattr(rag, "chunks_vdb", None)
        embed = getattr(store, "embedding_func", None)
    if embed is None:
        return None
    try:
        vectors = await embed([query, content])
    except Exception:
        logger.warning("重算向量相似度失败")
        return None
    if vectors is None or len(vectors) < 2:
        return None
    return _cosine(vectors[0], vectors[1])


def _clip(text: str, limit: int = 500) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _hit(
    *,
    chunk_id: str,
    doc_name: str,
    content: str,
    score: float,
    source: str,
    path: str = "",
    hop: int | None = None,
    evidence: dict[str, Any] | None = None,
    raw_score: float | None = None,
    calibrated_score: float | None = None,
    score_type: str = "",
    vector_score: float | None = None,
    keyword_score: float | None = None,
    rerank_score: float | None = None,
) -> dict[str, Any]:
    from lightrag.product_document_ir import apply_location

    row = {
        "chunk_id": chunk_id,
        "doc_name": doc_name or "文档片段",
        "page_num": None,
        "score": float(score or 0),
        "source": source,
        "excerpt": _clip(content, 700),
        "content": content or "",
        "parent_id": "",
        "parent_content": "",
        "reranked": False,
        "raw_score": raw_score if raw_score is not None else float(score or 0),
        "calibrated_score": calibrated_score,
        "score_type": score_type or source,
        "vector_score": vector_score,
        "keyword_score": keyword_score,
        "rerank_score": rerank_score,
        "untrusted": calibrated_score is None,
    }
    if isinstance(evidence, dict):
        if evidence.get("kb_id"):
            row["kb_id"] = evidence.get("kb_id")
        if evidence.get("owner_id") or evidence.get("user_id"):
            row["owner_id"] = evidence.get("owner_id") or evidence.get("user_id")
        if evidence.get("parent_id"):
            row["parent_id"] = evidence.get("parent_id")
        if evidence.get("index_text"):
            row["index_text"] = evidence.get("index_text")
        if evidence.get("source_id"):
            row["source_id"] = evidence.get("source_id")
    apply_location(row, evidence)
    if content:
        row["excerpt"] = _clip(content, 700)
    if path:
        row["path"] = path
    if hop is not None:
        row["hop"] = hop
    return row


def _in_scope(
    file_in_kb: Callable[..., bool],
    file_path: str,
    record: dict[str, Any] | None = None,
) -> bool:
    try:
        return bool(file_in_kb(file_path, record or {}))
    except TypeError:
        return bool(file_in_kb(file_path))


async def _vector_hits(
    rag, query: str, limit: int, file_in_kb: Callable[..., bool]
) -> list[dict[str, Any]]:
    rows = await rag.chunks_vdb.query(query, top_k=max(limit * 5, limit))
    hits: list[dict[str, Any]] = []
    for row in rows or []:
        file_path = row.get("file_path") or ""
        if not _in_scope(file_in_kb, file_path, row):
            continue
        content = row.get("content") or ""
        chunk_id = row.get("id") or row.get("__id__") or ""
        parent_id = ""
        index_text = ""
        stored = None
        if chunk_id:
            stored = await rag.text_chunks.get_by_id(chunk_id)
            if isinstance(stored, dict):
                content = content or stored.get("content") or ""
                parent_id = stored.get("parent_id") or ""
                index_text = stored.get("index_text") or ""
        if not content:
            continue
        evidence = stored if isinstance(stored, dict) and stored.get("unit_id") else row
        # NanoVectorDB 在余弦度量下把 __metrics__ 放进 distance。它是归一化向量的点积，越大越相似，不是距离。
        cosine = row.get("distance")
        try:
            cosine_value = (
                float(cosine) if cosine is not None and cosine != "" else None
            )
        except (TypeError, ValueError):
            cosine_value = None
        calibrated = None if cosine_value is None else calibrate_cosine(cosine_value)
        hit = _hit(
            chunk_id=str(chunk_id),
            doc_name=_filename(file_path) or "文档片段",
            content=content,
            score=0.0 if calibrated is None else calibrated,
            source="vector",
            evidence=evidence if isinstance(evidence, dict) else None,
            raw_score=cosine_value,
            calibrated_score=calibrated,
            score_type="vector",
            vector_score=calibrated,
            keyword_score=None,
        )
        hit["parent_id"] = parent_id or hit.get("parent_id") or ""
        hit["parent_content"] = ""
        if index_text:
            hit["index_text"] = index_text
        hits.append(hit)
        if len(hits) >= limit:
            break
    return hits


def _keyword_hits(
    query: str, chunks: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    documents = [item.get("content") or "" for item in chunks]
    scores = bm25_rank(query, documents)
    paired = sorted(zip(scores, chunks), key=lambda item: item[0], reverse=True)
    hits: list[dict[str, Any]] = []
    for score, chunk in paired:
        if score <= 0:
            break
        calibrated = calibrate_bm25(score)
        hit = _hit(
            chunk_id=str(chunk.get("chunk_id") or ""),
            doc_name=chunk.get("doc_name") or "文档片段",
            content=chunk.get("content") or "",
            score=calibrated,
            source="keyword",
            evidence=chunk,
            raw_score=score,
            calibrated_score=calibrated,
            score_type="keyword",
            vector_score=None,
            keyword_score=calibrated,
        )
        hit["parent_id"] = chunk.get("parent_id") or ""
        hit["parent_content"] = chunk.get("parent_content") or ""
        hits.append(hit)
        if len(hits) >= limit:
            break
    return hits


def _record_in_kb(
    record: dict[str, Any] | None, kb_id: str, owner_id: str = ""
) -> bool:
    from lightrag.product_scope import split_values

    if not isinstance(record, dict):
        return not kb_id and not owner_id
    if kb_id and kb_id not in split_values(record.get("kb_id")):
        return False
    if owner_id:
        owners = split_values(record.get("owner_id") or record.get("user_id"))
        if owner_id not in owners:
            return False
    return True


async def _graph_hits(
    rag,
    query: str,
    limit: int,
    file_in_kb: Callable[..., bool],
    kb_id: str = "",
    owner_id: str = "",
) -> list[dict[str, Any]]:
    from lightrag.product_scope import split_values

    graph = rag.chunk_entity_relation_graph
    seeds_raw = []
    if getattr(rag, "entities_vdb", None) is not None:
        seeds_raw = await rag.entities_vdb.query(query, top_k=max(8, min(limit, 12)))
    seeds: list[tuple[str, float]] = []
    names: dict[str, str] = {}
    seed_cosine: dict[str, float] = {}
    for row in seeds_raw or []:
        entity_id = str(
            row.get("entity_id") or row.get("id") or row.get("__id__") or ""
        )
        label = str(row.get("entity_name") or row.get("name") or "")
        if not entity_id:
            continue
        if not _record_in_kb(row, kb_id, owner_id):
            continue
        file_path = row.get("file_path") or ""
        if file_path and not _in_scope(file_in_kb, file_path, row):
            continue
        try:
            cosine = float(row.get("distance") or 0)
        except (TypeError, ValueError):
            cosine = 0.0
        if label and label in query:
            continue
        if cosine < 0.5:
            continue
        seeds.append((entity_id, cosine))
        names[entity_id] = label or entity_id
        seed_cosine[entity_id] = cosine
        if len(seeds) >= 8:
            break
    if hasattr(graph, "get_all_nodes"):
        matched: list[tuple[int, str, str]] = []
        for node in await graph.get_all_nodes() or []:
            node_id = str(node.get("id") or node.get("entity_id") or "")
            label = str(node.get("name") or "")
            if not node_id or len(label) < 2 or label not in query:
                continue
            if not _record_in_kb(node, kb_id, owner_id):
                continue
            names[node_id] = label
            matched.append((len(label), node_id, label))
        if matched:
            longest = max(item[0] for item in matched)
            for length, node_id, _label in matched:
                if length != longest:
                    continue
                if node_id not in seed_cosine:
                    seed_cosine[node_id] = 0.0
                    seeds.append((node_id, 0.8))
                else:
                    seeds = [
                        (item_id, max(score, 0.8) if item_id == node_id else score)
                        for item_id, score in seeds
                    ]
    if not seeds:
        return []

    async def adjacency_for(node_ids: list[str]) -> dict[str, list[tuple[str, str]]]:
        adj: dict[str, list[tuple[str, str]]] = {}
        for node_id in node_ids:
            node = await graph.get_node(node_id) or {}
            if node and not _record_in_kb(node, kb_id, owner_id):
                adj[node_id] = []
                continue
            if node.get("name"):
                names.setdefault(node_id, str(node.get("name")))
            edges = await graph.get_node_edges(node_id) or []
            neighbors: list[tuple[str, str]] = []
            for src, tgt in edges:
                other = tgt if src == node_id else src
                other_node = await graph.get_node(other) or {}
                if other_node and not _record_in_kb(other_node, kb_id, owner_id):
                    continue
                edge = (
                    await graph.get_edge(src, tgt)
                    or await graph.get_edge(tgt, src)
                    or {}
                )
                if edge and not _record_in_kb(edge, kb_id, owner_id):
                    continue
                label = str(
                    edge.get("relation_type")
                    or edge.get("keywords")
                    or edge.get("description")
                    or "相关"
                )
                label = label.split(GRAPH_FIELD_SEP)[0].strip()[:24] or "相关"
                if other_node.get("name"):
                    names.setdefault(other, str(other_node.get("name")))
                neighbors.append((other, label))
            adj[node_id] = neighbors
        return adj

    first = await adjacency_for([entity_id for entity_id, _score in seeds])
    hop_ids: list[str] = []
    for neighbors in first.values():
        for other, _label in neighbors:
            if other not in first and other not in hop_ids:
                hop_ids.append(other)
    second = await adjacency_for(hop_ids[:40])
    paths = expand_paths(seeds, {**first, **second}, hops=2)[: max(limit * 3, limit)]

    hits: list[dict[str, Any]] = []
    seen_chunks: set[str] = set()
    for item in paths:
        node = await graph.get_node(item["name"]) or {}
        if node and not _record_in_kb(node, kb_id, owner_id):
            continue
        if not _in_scope(file_in_kb, str(node.get("file_path") or ""), node):
            continue
        origin = item["path"][0] if item.get("path") else item["name"]
        cosine = seed_cosine.get(origin, seed_cosine.get(item["name"], 0.0))
        name_hit = (
            1.0
            if any(
                label and label in query
                for label in (names.get(origin, ""), names.get(item["name"], ""))
            )
            else 0.0
        )
        if cosine <= 0 and name_hit <= 0:
            continue
        shown = [names.get(part, part) for part in item["path"]]
        path_text = " → ".join(shown)
        description = str(node.get("description") or node.get("name") or "").split(
            GRAPH_FIELD_SEP
        )[0]
        etype = node.get("entity_type") or "实体"
        if isinstance(etype, list):
            etype = etype[0] if etype else "实体"
        excerpt = f"[{etype}] {path_text}"
        if description and description not in path_text:
            excerpt += f"。{description}"
        source_ids = split_values(node.get("source_id"))
        for chunk_id in source_ids:
            if not chunk_id:
                continue
            if chunk_id in seen_chunks:
                for existing in hits:
                    if existing.get("chunk_id") == chunk_id and int(
                        item.get("hop") or 0
                    ) > int(existing.get("hop") or 0):
                        existing["hop"] = item["hop"]
                        existing["path"] = path_text
                        existing["path_id"] = path_text
                        existing["ranking_score"] = item["score"] * 0.85
                continue
            stored = await rag.text_chunks.get_by_id(chunk_id)
            if not isinstance(stored, dict):
                continue
            if not _record_in_kb(stored, kb_id, owner_id):
                continue
            chunk_file = stored.get("file_path") or ""
            if not _in_scope(file_in_kb, chunk_file, stored):
                continue
            seen_chunks.add(chunk_id)
            content = stored.get("content") or excerpt
            chunk_hit = _hit(
                chunk_id=chunk_id,
                doc_name=_filename(chunk_file) or "文档片段",
                content=content,
                score=item["score"],
                source="graph",
                path=path_text,
                hop=item["hop"],
                evidence=stored,
                raw_score=cosine,
                score_type="graph",
                vector_score=calibrate_cosine(cosine),
                keyword_score=name_hit,
            )
            chunk_hit["ranking_score"] = item["score"] * 0.85
            chunk_hit["entity_id"] = item["name"]
            chunk_hit["path_id"] = path_text
            chunk_hit["source_id"] = chunk_id
            chunk_hit["channel"] = "graph"
            hits.append(chunk_hit)
        if len(hits) >= limit * 2:
            break
    return hits


async def _hyde_text(rag, query: str) -> str:
    raw = await rag.llm_model_func(_HYDE_PROMPT.format(query=query))
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else ""
    text = str(raw or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.strip()[:800]


async def _mix_hits(
    rag, query: str, limit: int, file_in_kb: Callable[..., bool]
) -> list[dict[str, Any]]:
    param = QueryParam(
        mode="mix",
        top_k=limit,
        chunk_top_k=limit,
        enable_rerank=False,
    )
    raw = await rag.aquery_data(query, param=param)
    hits: list[dict[str, Any]] = []
    for chunk in raw.get("chunks") or []:
        file_path = chunk.get("file_path") or ""
        if not _in_scope(file_in_kb, file_path, chunk):
            continue
        content = chunk.get("content") or ""
        if not content:
            continue
        chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or "")
        stored = None
        if chunk_id:
            stored = await rag.text_chunks.get_by_id(chunk_id)
        evidence = dict(stored) if isinstance(stored, dict) else dict(chunk)
        if isinstance(stored, dict):
            content = content or stored.get("content") or ""
        vector = _retrieved_score(chunk)
        if vector is None:
            vector = await _query_cosine(rag, query, content)
        keyword_raw = bm25_rank(query, [content])
        keyword = calibrate_bm25(keyword_raw[0] if keyword_raw else 0.0)
        calibrated = None if vector is None else calibrate_cosine(vector)
        if calibrated is None and keyword > 0:
            calibrated = keyword
        hits.append(
            _hit(
                chunk_id=chunk_id,
                doc_name=_filename(file_path) or "文档片段",
                content=content,
                score=0.0 if calibrated is None else calibrated,
                source="mix",
                evidence=evidence,
                raw_score=vector,
                calibrated_score=calibrated,
                score_type="mix" if vector is not None else "keyword",
                vector_score=None if vector is None else calibrate_cosine(vector),
                keyword_score=keyword,
            )
        )
    for entity in raw.get("entities") or []:
        file_path = entity.get("file_path") or ""
        if not _in_scope(file_in_kb, file_path, entity):
            continue
        name = entity.get("entity_name") or ""
        entity_score = entity.get("score")
        if entity_score is None or entity_score == "":
            continue
        try:
            entity_value = float(entity_score)
        except (TypeError, ValueError):
            continue
        calibrated = calibrate_cosine(entity_value)
        hits.append(
            _hit(
                chunk_id=f"entity:{name}",
                doc_name=_filename(file_path) or "知识图谱",
                content=f"[{entity.get('entity_type') or '实体'}] {name}：{entity.get('description') or ''}",
                score=calibrated,
                source="graph",
                raw_score=entity_value,
                calibrated_score=calibrated,
                score_type="vector",
                vector_score=calibrated,
                keyword_score=0.0,
            )
        )
    for rel in raw.get("relationships") or []:
        src = rel.get("src_id") or rel.get("source") or ""
        tgt = rel.get("tgt_id") or rel.get("target") or ""
        rel_path = rel.get("file_path") or ""
        if not src and not tgt:
            continue
        if not _in_scope(file_in_kb, rel_path, rel):
            continue
        relation_score = rel.get("score")
        if relation_score is None or relation_score == "":
            continue
        try:
            relation_value = float(relation_score)
        except (TypeError, ValueError):
            continue
        calibrated = calibrate_cosine(relation_value)
        hits.append(
            _hit(
                chunk_id=f"rel:{src}->{tgt}",
                doc_name="知识图谱",
                content=f"{src} —{rel.get('keywords') or rel.get('description') or '相关'}→ {tgt}",
                score=calibrated,
                source="graph",
                raw_score=relation_value,
                calibrated_score=calibrated,
                score_type="vector",
                vector_score=calibrated,
                keyword_score=0.0,
            )
        )
    return hits


def _safe_rerank_warning(exc: BaseException) -> str:
    text = str(exc)
    for name in (
        "RERANK_BINDING_API_KEY",
        "EMBEDDING_BINDING_API_KEY",
        "LLM_BINDING_API_KEY",
    ):
        secret = os.getenv(name) or ""
        if len(secret) > 6:
            text = text.replace(secret, "***")
    text = re.sub(r"sk-[A-Za-z0-9_\-]+", "sk-***", text)
    return text[:180]


async def _rerank(
    query: str,
    hits: list[dict[str, Any]],
    top_k: int,
    pool_size: int = 40,
    rerank_func: Callable[..., Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from lightrag.retrieval_orchestrator import rerank_document

    configured_model = os.getenv("RERANK_MODEL") or ""
    model = configured_model
    host = os.getenv("RERANK_BINDING_HOST") or "https://api.siliconflow.cn/v1/rerank"
    info = {
        "enabled": True,
        "applied": False,
        "model": configured_model,
        "strategy": "cross-encoder",
        "warning": None,
        "pool_size": max(int(pool_size), int(top_k)),
    }
    if not hits:
        info["applied"] = True
        return hits, info
    if not model:
        info["warning"] = "未配置重排模型"
        return hits, info
    pool = hits[: info["pool_size"]]
    documents = [rerank_document(item) for item in pool]
    caller = rerank_func or siliconflow_rerank
    try:
        ranked = await caller(
            query=query,
            documents=documents,
            top_n=len(pool),
            model=model,
            base_url=host,
        )
    except Exception as exc:
        warning = _safe_rerank_warning(exc)
        logger.warning("检索测试重排失败: %s", warning)
        info["warning"] = f"重排服务不可用，已保留原排序：{warning}"
        return pool, info
    ordered = reorder_by_rerank(pool, ranked)
    if not ordered:
        info["warning"] = "重排没有返回结果，已保留原排序"
        return pool, info
    info["applied"] = True
    info["reranked"] = len(ordered)
    return ordered, info


async def run_search_test(
    rag,
    *,
    query: str,
    mode: str,
    top_k: int,
    ratio: float,
    enable_rerank: bool,
    kb_chunks: list[dict[str, Any]],
    file_in_kb: Callable[[str], bool],
    score_threshold: float | None = None,
    kb_id: str = "",
    owner_id: str = "",
    working_dir: Path | None = None,
    staged: bool = False,
    history: list[str] | None = None,
    query_force: str | None = None,
    expand_parents: bool = True,
    graph_rerank: str = "bundle",
    rerank_func: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    mode = (mode or "mix").lower()
    top_k = max(1, min(int(top_k or 10), 30))
    if working_dir is not None:
        from lightrag.index_manifest import compatibility_error

        blocked = compatibility_error(Path(working_dir))
        if blocked:
            return {
                "query": query,
                "mode": mode,
                "total": 0,
                "chunks": [],
                "status": blocked["status"],
                "empty_reason": blocked["reason"],
                "rebuild_reason": blocked["reason"],
                "ratio_applied": False,
                "similarity_ratio": None,
                "hyde_text": None,
                "rerank": {
                    "enabled": bool(enable_rerank),
                    "applied": False,
                    "warning": None,
                },
                "query_plan": None,
                "answer_context": [],
                **_runtime_fields(
                    Path(working_dir) if working_dir is not None else None
                ),
            }
    if staged:
        return await _run_staged(
            rag,
            query=query,
            mode=mode,
            top_k=top_k,
            ratio=ratio,
            enable_rerank=enable_rerank,
            kb_chunks=kb_chunks,
            file_in_kb=file_in_kb,
            score_threshold=score_threshold,
            kb_id=kb_id,
            owner_id=owner_id,
            working_dir=working_dir,
            history=history,
            query_force=query_force,
            expand_parents=expand_parents,
            graph_rerank=graph_rerank,
            rerank_func=rerank_func,
        )
    # 重排前先多召回，交叉编码器再把候选收到 Top-K。
    pool = 50 if enable_rerank else top_k
    hyde_text = None
    ratio_applied = False

    if mode == "vector":
        hits = await _vector_hits(rag, query, pool, file_in_kb)
    elif mode == "keyword":
        hits = _keyword_hits(query, kb_chunks, pool)
    elif mode == "hybrid":
        ratio_applied = True
        vector_hits = await _vector_hits(rag, query, pool, file_in_kb)
        keyword_hits = _keyword_hits(query, kb_chunks, pool)
        hits = weighted_fuse(vector_hits, keyword_hits, ratio)[:pool]
    elif mode == "graph":
        hits = await _graph_hits(rag, query, pool, file_in_kb, kb_id, owner_id)
    elif mode == "tree":
        hyde_text = await _hyde_text(rag, query)
        hypothetical = hyde_text or query
        hyde_hits = await _vector_hits(rag, hypothetical, pool, file_in_kb)
        original_hits = await _vector_hits(rag, query, pool, file_in_kb)
        ranked = weighted_fuse(hyde_hits, original_hits, 0.65)[:pool]
        original_by_id = {item["chunk_id"]: item for item in original_hits}
        hits = []
        for item in ranked:
            base = original_by_id.get(item["chunk_id"])
            query_cosine = None if base is None else base.get("vector_score")
            if query_cosine is None:
                query_cosine = await _query_cosine(
                    rag, query, item.get("content") or ""
                )
                if query_cosine is not None:
                    query_cosine = calibrate_cosine(query_cosine)
            item["source"] = "tree"
            item["score_type"] = "tree"
            item["vector_score"] = query_cosine
            item["keyword_score"] = 0.0
            item["raw_score"] = query_cosine
            item["calibrated_score"] = query_cosine
            item["untrusted"] = query_cosine is None
            hits.append(item)
    elif mode == "mix":
        hits = await _mix_hits(rag, query, pool, file_in_kb)
    else:
        hits = await _vector_hits(rag, query, pool, file_in_kb)
        mode = "vector"
    if kb_id or owner_id:
        hits = [
            item
            for item in hits
            if _record_in_kb(item, kb_id, owner_id) or not item.get("kb_id")
        ]

    rerank_info = {
        "enabled": bool(enable_rerank),
        "applied": False,
        "model": os.getenv("RERANK_MODEL") or "",
        "strategy": "cross-encoder",
        "warning": None,
    }
    if enable_rerank:
        hits, rerank_info = await _rerank(query, hits, top_k)
    from lightrag.retrieval_admission import annotate_admission

    for item in hits:
        if item.get("ranking_score") is None:
            item["ranking_score"] = item.get("score")
        annotate_admission(item, mode, ratio)
    hits.sort(key=lambda item: float(item.get("ranking_score") or 0), reverse=True)
    hits = filter_by_threshold(hits, score_threshold)
    hits = hits[:top_k]

    def _round(value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            return round(float(value), 4)
        except (TypeError, ValueError):
            return None

    public = []
    for item in hits:
        shown = item.get("calibrated_score")
        if shown is None:
            shown = item.get("score")
        owner_id = str(item.get("owner_id") or "")
        kb_value = str(item.get("kb_id") or "")
        document_id = str(item.get("document_id") or "")
        if not owner_id or not kb_value or not document_id:
            continue
        public.append(
            {
                "chunk_id": item.get("chunk_id") or "",
                "doc_name": item.get("doc_name") or "文档片段",
                "page_num": None,
                "score": _round(item.get("admission_score")) or 0,
                "ranking_score": _round(item.get("ranking_score")),
                "admission_score": _round(item.get("admission_score")),
                "raw_score": _round(item.get("raw_score")),
                "calibrated_score": _round(item.get("admission_score")),
                "document_id": item.get("document_id"),
                "unit_id": item.get("unit_id"),
                "evidence_refs": item.get("evidence_refs") or [],
                "score_type": item.get("score_type") or item.get("source") or mode,
                "vector_score": _round(item.get("vector_score")),
                "keyword_score": _round(item.get("keyword_score")),
                "rerank_score": _round(item.get("rerank_score")),
                "source": item.get("source") or mode,
                "excerpt": item.get("excerpt") or _clip(item.get("content") or ""),
                "content": item.get("content") or "",
                "parent_id": item.get("parent_id") or "",
                "parent_content": item.get("parent_content") or "",
                "reranked": bool(item.get("reranked")),
                "path": item.get("path") or "",
                "hop": item.get("hop"),
                "page_number": item.get("page_number"),
                "slide_number": item.get("slide_number"),
                "section_path": item.get("section_path") or [],
                "sheet_name": item.get("sheet_name"),
                "cell_range": item.get("cell_range"),
                "entity_id": item.get("entity_id") or "",
                "kb_id": kb_value,
                "owner_id": owner_id,
                "channel": item.get("channel") or "",
                "channel_rank": item.get("channel_rank"),
                "query_variant_id": item.get("query_variant_id") or "",
                "path_id": item.get("path_id") or "",
                "source_id": item.get("source_id") or item.get("chunk_id") or "",
            }
        )
    return {
        "query": query,
        "mode": mode,
        "status": "ok",
        "total": len(public),
        "chunks": public,
        "ratio_applied": ratio_applied,
        "similarity_ratio": ratio if ratio_applied else None,
        "hyde_text": hyde_text,
        "rerank": rerank_info,
        "empty_reason": None if public else "没有达到相关度要求的内容",
        "query_plan": None,
        "answer_context": [],
        **_runtime_fields(Path(working_dir) if working_dir is not None else None),
    }


def _stamp(
    hits: list[dict[str, Any]], channel: str, variant_id: str
) -> list[dict[str, Any]]:
    stamped = []
    for rank, hit in enumerate(hits, start=1):
        row = dict(hit)
        row["channel"] = channel
        row["channel_rank"] = rank
        row["query_variant_id"] = variant_id
        stamped.append(row)
    return stamped


def _public_chunks(hits: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    def _round(value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            return round(float(value), 4)
        except (TypeError, ValueError):
            return None

    public = []
    for item in hits:
        owner = str(item.get("owner_id") or "")
        kb_value = str(item.get("kb_id") or "")
        document_id = str(item.get("document_id") or "")
        if not owner or not kb_value or not document_id:
            continue
        public.append(
            {
                "chunk_id": item.get("chunk_id") or "",
                "doc_name": item.get("doc_name") or "文档片段",
                "page_num": item.get("page_num"),
                "score": _round(item.get("admission_score")) or 0,
                "ranking_score": _round(item.get("ranking_score")),
                "admission_score": _round(item.get("admission_score")),
                "raw_score": _round(item.get("raw_score")),
                "calibrated_score": _round(item.get("admission_score")),
                "document_id": document_id,
                "unit_id": item.get("unit_id"),
                "evidence_refs": item.get("evidence_refs") or [],
                "score_type": item.get("score_type") or item.get("source") or mode,
                "vector_score": _round(item.get("vector_score")),
                "keyword_score": _round(item.get("keyword_score")),
                "rerank_score": _round(item.get("rerank_score")),
                "source": item.get("source") or mode,
                "excerpt": item.get("excerpt") or _clip(item.get("content") or ""),
                "content": item.get("content") or "",
                "parent_id": item.get("parent_id") or "",
                "parent_content": "",
                "reranked": bool(item.get("reranked")),
                "path": item.get("path") or "",
                "hop": item.get("hop"),
                "page_number": item.get("page_number"),
                "slide_number": item.get("slide_number"),
                "section_path": item.get("section_path") or [],
                "sheet_name": item.get("sheet_name"),
                "cell_range": item.get("cell_range"),
                "entity_id": item.get("entity_id") or "",
                "kb_id": kb_value,
                "owner_id": owner,
                "channel": item.get("channel") or "",
                "channel_rank": item.get("channel_rank"),
                "query_variant_id": item.get("query_variant_id") or "",
                "path_id": item.get("path_id") or "",
                "source_id": item.get("source_id") or item.get("chunk_id") or "",
            }
        )
    return public


async def _run_staged(
    rag,
    *,
    query: str,
    mode: str,
    top_k: int,
    ratio: float,
    enable_rerank: bool,
    kb_chunks: list[dict[str, Any]],
    file_in_kb: Callable[[str], bool],
    score_threshold: float | None,
    kb_id: str,
    owner_id: str,
    working_dir: Path | None,
    history: list[str] | None,
    query_force: str | None,
    expand_parents: bool,
    graph_rerank: str,
    rerank_func: Callable[..., Any] | None,
) -> dict[str, Any]:
    clock = time.perf_counter()
    from lightrag.product_storage import _kv
    from lightrag.query_plan import build_query_plan, is_concept, needs_graph
    from lightrag.retrieval_admission import annotate_admission
    from lightrag.retrieval_orchestrator import (
        BM25_TOP,
        DENSE_TOP,
        FINAL_PARENTS_MAX,
        GRAPH_TOP,
        RERANK_POOL,
        RRF_CAP,
        aggregate_parents,
        expand_admitted_bundles,
        final_child_limit,
        fuse_ranked_groups,
        path_bundles,
        rrf_merge,
    )

    async def llm(prompt: str) -> str:
        caller = getattr(rag, "llm_model_func", None)
        if caller is None:
            raise RuntimeError("llm_unavailable")
        raw = caller(prompt)
        if hasattr(raw, "__await__"):
            raw = await raw
        return str(raw or "")

    model_caller = llm if getattr(rag, "llm_model_func", None) else None
    plan = await build_query_plan(
        query, history=history, llm=model_caller, force=query_force
    )
    if not query_force and plan["route"] == "factual" and is_concept(query):
        probe = await _vector_hits(rag, plan["standalone_query"], 5, file_in_kb)
        weak = not any(float(item.get("score") or 0) >= 0.2 for item in probe)
        if weak:
            plan = await build_query_plan(
                query,
                history=history,
                llm=model_caller,
                first_pass_weak=True,
                force=query_force,
            )
    recall = []
    for text in [plan["standalone_query"], *plan["variants"], *plan["subquestions"]]:
        if text and text not in recall:
            recall.append(text)
    channels: dict[str, list[dict[str, Any]]] = {}
    use_dense = mode in {"vector", "hybrid", "mix", "tree"}
    use_bm25 = mode in {"keyword", "hybrid", "mix"} or plan["route"] == "exact"
    use_graph = mode == "graph" or (mode == "mix" and needs_graph(plan, query))
    if use_dense:
        for index, text in enumerate(recall):
            rows = _stamp(
                await _vector_hits(rag, text, DENSE_TOP, file_in_kb),
                "dense",
                f"v{index}",
            )
            if rows:
                channels[f"dense-{index}"] = rows
        if plan["route"] == "hyde" and plan.get("hyde_text"):
            rows = _stamp(
                await _vector_hits(rag, plan["hyde_text"], DENSE_TOP, file_in_kb),
                "dense",
                "hyde",
            )
            if rows:
                channels["hyde"] = rows
    if use_bm25:
        for index, text in enumerate(recall):
            rows = _stamp(_keyword_hits(text, kb_chunks, BM25_TOP), "bm25", f"v{index}")
            if rows:
                channels[f"bm25-{index}"] = rows
    graph_hits: list[dict[str, Any]] = []
    if use_graph:
        graph_hits = _stamp(
            await _graph_hits(
                rag, plan["standalone_query"], GRAPH_TOP, file_in_kb, kb_id, owner_id
            ),
            "graph",
            "v0",
        )
        if graph_hits:
            channels["graph"] = graph_hits
    text_channels = {key: value for key, value in channels.items() if key != "graph"}
    if mode == "hybrid" and text_channels:
        union: dict[str, dict[str, Any]] = {}
        for rows in text_channels.values():
            for hit in rows:
                current = union.get(hit["chunk_id"])
                if current is None:
                    current = dict(hit)
                    union[hit["chunk_id"]] = current
                if hit.get("vector_score") is not None:
                    current["vector_score"] = max(
                        float(current.get("vector_score") or 0),
                        float(hit["vector_score"]),
                    )
                if hit.get("keyword_score") is not None:
                    current["keyword_score"] = max(
                        float(current.get("keyword_score") or 0),
                        float(hit["keyword_score"]),
                    )
        blend_rows = list(union.values())
        for row in blend_rows:
            row["ranking_score"] = float(ratio) * float(
                row.get("vector_score") or 0
            ) + (1 - float(ratio)) * float(row.get("keyword_score") or 0)
        blend_rows.sort(
            key=lambda row: (
                -float(row.get("ranking_score") or 0),
                str(row.get("chunk_id") or ""),
            )
        )
        channels["blend"] = _stamp(blend_rows, "blend", "v0")
    merged = rrf_merge(channels, limit=RRF_CAP) if channels else []
    text_merged = rrf_merge(text_channels, limit=RRF_CAP) if text_channels else []
    candidate_units: list[str] = []
    for item in merged:
        unit_id = str(item.get("unit_id") or "")
        if unit_id and unit_id not in candidate_units:
            candidate_units.append(unit_id)
    rerank_info = {
        "enabled": bool(enable_rerank),
        "applied": False,
        "model": os.getenv("RERANK_MODEL") or "",
        "strategy": "cross-encoder",
        "warning": None,
        "pool_size": RERANK_POOL,
    }
    if mode == "graph" and enable_rerank and graph_rerank != "chunk":
        bundles = path_bundles(graph_hits or merged)
        bundle_rows = [
            {
                "chunk_id": bundle["path_id"],
                "content": bundle["content"],
                "index_text": bundle["content"],
                "doc_name": "",
                "raw_score": bundle.get("raw_score"),
                "source": "graph",
            }
            for bundle in bundles
        ]
        ordered, rerank_info = await _rerank(
            query, bundle_rows, top_k, pool_size=RERANK_POOL, rerank_func=rerank_func
        )
        by_path = {bundle["path_id"]: bundle for bundle in bundles}
        admitted = []
        for row in ordered:
            bundle = by_path.get(row.get("chunk_id"))
            if bundle is None:
                continue
            bundle["rerank_score"] = row.get("rerank_score")
            bundle["reranked"] = True
            bundle["score"] = row.get("score")
            annotate_admission(bundle, "graph", ratio)
            if filter_by_threshold([bundle], score_threshold):
                admitted.append(bundle)
        hits = expand_admitted_bundles(admitted)
    elif mode == "mix" and use_graph:
        if enable_rerank:
            text_ranked, rerank_info = await _rerank(
                query,
                text_merged,
                top_k,
                pool_size=RERANK_POOL,
                rerank_func=rerank_func,
            )
            bundles = path_bundles(graph_hits)
            bundle_rows = [
                {
                    "chunk_id": bundle["path_id"],
                    "content": bundle["content"],
                    "index_text": bundle["content"],
                    "doc_name": "",
                    "raw_score": bundle.get("raw_score"),
                    "source": "graph",
                }
                for bundle in bundles
            ]
            ordered, graph_info = await _rerank(
                query,
                bundle_rows,
                top_k,
                pool_size=min(GRAPH_TOP, RERANK_POOL),
                rerank_func=rerank_func,
            )
            if graph_info.get("warning") and not rerank_info.get("warning"):
                rerank_info["warning"] = graph_info["warning"]
            by_path = {bundle["path_id"]: bundle for bundle in bundles}
            admitted = []
            for row in ordered:
                bundle = by_path.get(row.get("chunk_id"))
                if bundle is None:
                    continue
                bundle["rerank_score"] = row.get("rerank_score")
                bundle["reranked"] = True
                bundle["score"] = row.get("score")
                annotate_admission(bundle, "graph", ratio)
                if filter_by_threshold([bundle], score_threshold):
                    admitted.append(bundle)
            expanded = expand_admitted_bundles(admitted)
            hits = fuse_ranked_groups([text_ranked, expanded])
        else:
            hits = rrf_merge({"text": text_merged, "graph": graph_hits}, limit=RRF_CAP)
    else:
        source = merged
        if enable_rerank and graph_rerank == "chunk" and mode == "graph":
            source = graph_hits or merged
        if enable_rerank:
            hits, rerank_info = await _rerank(
                query, source, top_k, pool_size=RERANK_POOL, rerank_func=rerank_func
            )
        else:
            hits = source
    if mode == "tree":
        for item in hits:
            item["source"] = "tree"
            if (
                item.get("query_variant_id") == "hyde"
                or item.get("vector_score") is None
            ):
                cosine = await _query_cosine(rag, query, item.get("content") or "")
                if cosine is not None:
                    cosine = calibrate_cosine(cosine)
                item["vector_score"] = cosine
                item["raw_score"] = cosine
                item["keyword_score"] = 0.0
    if kb_id or owner_id:
        hits = [
            item
            for item in hits
            if _record_in_kb(item, kb_id, owner_id) or not item.get("kb_id")
        ]
    fused_mix = mode == "mix" and use_graph
    for item in hits:
        if (
            item.get("reranked")
            and item.get("rerank_score") is not None
            and not fused_mix
        ):
            item["ranking_score"] = float(item["rerank_score"])
        elif item.get("rrf_score") is not None:
            item["ranking_score"] = float(item["rrf_score"])
        elif item.get("ranking_score") is None:
            item["ranking_score"] = item.get("score")
        if item.get("admission_score") is None:
            annotate_admission(item, mode, ratio)
    hits.sort(
        key=lambda item: (
            -float(item.get("ranking_score") or 0),
            str(item.get("chunk_id") or ""),
        )
    )
    hits = filter_by_threshold(hits, score_threshold)
    parents = _kv(Path(working_dir), "parent_chunks") if working_dir is not None else {}
    selected, contexts = aggregate_parents(
        hits,
        parents,
        max_children=final_child_limit(top_k),
        max_parents=FINAL_PARENTS_MAX,
        enabled=expand_parents,
    )
    for item in selected:
        item["parent_content"] = ""
    public = _public_chunks(selected, mode)
    return {
        "query": query,
        "mode": mode,
        "status": "ok",
        "total": len(public),
        "chunks": public,
        "candidates": candidate_units[:50],
        "ratio_applied": mode == "hybrid",
        "similarity_ratio": ratio if mode == "hybrid" else None,
        "hyde_text": plan.get("hyde_text") or None,
        "rerank": rerank_info,
        "query_plan": plan,
        "answer_context": contexts,
        "retrieval": {
            "dense_top": DENSE_TOP if use_dense else 0,
            "bm25_top": BM25_TOP if use_bm25 else 0,
            "graph_top": len(graph_hits),
            "rrf": len(merged),
            "rerank_pool": RERANK_POOL if enable_rerank else 0,
            "children": len(public),
            "parents": len(contexts),
        },
        "empty_reason": None if public else "没有达到相关度要求的内容",
        "timings": {"total": round((time.perf_counter() - clock) * 1000, 1)},
        **_runtime_fields(Path(working_dir) if working_dir is not None else None),
    }
