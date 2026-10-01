"""离线检索评测。calibration 只用于观察分数和推荐门槛，holdout 才用于验收。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

MODES = ("vector", "keyword", "hybrid", "mix", "tree", "graph")


def _ci_embed_one(text: str, dim: int = 16384) -> np.ndarray:
    from lightrag.search_strategies import tokenize

    vector = np.zeros(dim, dtype=np.float32)
    for token in tokenize(text):
        slot = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % dim
        vector[slot] += 1.0
    norm = float(np.linalg.norm(vector)) or 1.0
    return vector / norm


async def _ci_embed(texts: list[str]) -> np.ndarray:
    return np.vstack([_ci_embed_one(text) for text in texts])


def _percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return round(ordered[index], 4)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "min": round(min(values), 4),
        "p50": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
        "max": round(max(values), 4),
    }


class _Chunks:
    def __init__(self, rows: dict[str, dict[str, Any]]):
        self.rows = rows

    async def get_by_id(self, chunk_id: str):
        return self.rows.get(chunk_id)


class _VectorDB:
    def __init__(self, rows: list[dict[str, Any]], vectors: np.ndarray, embed_one: Callable[[str], np.ndarray]):
        self.rows = rows
        self.vectors = vectors
        self.embed_one = embed_one

    async def query(self, query: str, top_k: int = 10, query_embedding=None):
        if query_embedding is None:
            query_vector = self.embed_one(query)
        else:
            query_vector = np.asarray(query_embedding, dtype=np.float32)
            query_vector = query_vector / (float(np.linalg.norm(query_vector)) or 1.0)
        scored = []
        for row, vector in zip(self.rows, self.vectors):
            cosine = float(np.dot(query_vector, vector))
            scored.append({
                **row,
                "id": row.get("chunk_id") or row.get("entity_id"),
                "__id__": row.get("entity_id") or row.get("chunk_id"),
                "distance": cosine,
            })
        scored.sort(key=lambda item: item["distance"], reverse=True)
        return scored[:top_k]


class _Graph:
    def __init__(self, nodes: dict[str, dict[str, Any]], edges: dict[tuple[str, str], dict[str, Any]]):
        self.nodes = nodes
        self.edges = edges

    async def get_node(self, node_id: str):
        return self.nodes.get(node_id)

    async def get_edge(self, src: str, tgt: str):
        return self.edges.get((src, tgt))

    async def get_node_edges(self, node_id: str):
        if node_id not in self.nodes:
            return None
        return [(src, tgt) for src, tgt in self.edges if src == node_id or tgt == node_id]

    async def get_all_nodes(self):
        return [{"id": node_id, **attrs} for node_id, attrs in self.nodes.items()]


class _Rag:
    def __init__(self, chunks: list[dict[str, Any]], entities: list[dict[str, Any]], graph: _Graph, embed, embed_one):
        self.text_chunks = _Chunks({row["chunk_id"]: row for row in chunks})
        self.chunks_vdb = _VectorDB(chunks, np.vstack([row["vector"] for row in chunks]), embed_one)
        self.entities_vdb = _VectorDB(entities, np.vstack([row["vector"] for row in entities]), embed_one) if entities else None
        self.chunk_entity_relation_graph = graph
        self.embedding_func = embed

    async def llm_model_func(self, prompt: str, **kwargs):
        del kwargs
        return "假设说明：" + prompt[-80:]

    async def aquery_data(self, query: str, param=None):
        del query, param
        return {
            "chunks": [{key: value for key, value in row.items() if key != "vector"} for row in self.text_chunks.rows.values()],
            "entities": [],
            "relationships": [],
        }


def _entity_id(kb_id: str, name: str) -> str:
    return "ent-" + hashlib.sha256(f"{kb_id}\n概念\n{name}".encode("utf-8")).hexdigest()[:16]


def load_dataset(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_library(dataset: dict[str, Any], embed_one: Callable[[str], np.ndarray]) -> tuple[_Rag, list[dict[str, Any]]]:
    chunks: list[dict[str, Any]] = []
    by_unit: dict[tuple[str, str], str] = {}
    for doc in dataset["documents"]:
        for index, chunk in enumerate(doc["chunks"], start=1):
            chunk_id = f"{doc['id']}-c{index:04d}"
            by_unit[(doc["id"], chunk["unit_id"])] = chunk_id
            row = {
                "chunk_id": chunk_id,
                "content": chunk["text"],
                "doc_name": doc["name"],
                "file_path": doc["name"],
                "document_id": doc["id"],
                "doc_id": doc["id"],
                "unit_id": chunk["unit_id"],
                "kb_id": doc["kb_id"],
                "owner_id": doc["owner_id"],
                "user_id": doc["owner_id"],
                "evidence_refs": [{
                    "document_id": doc["id"],
                    "unit_id": chunk["unit_id"],
                    "chunk_id": chunk_id,
                }],
                "vector": embed_one(chunk["text"]),
            }
            chunks.append(row)
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str], dict[str, Any]] = {}
    for relation in dataset.get("relations") or []:
        src = _entity_id(relation["kb_id"], relation["src"])
        tgt = _entity_id(relation["kb_id"], relation["tgt"])
        chunk_id = by_unit[(relation["document_id"], relation["unit_id"])]
        for node_id, name in ((src, relation["src"]), (tgt, relation["tgt"])):
            current = nodes.setdefault(node_id, {
                "name": name,
                "entity_type": "概念",
                "kb_id": relation["kb_id"],
                "owner_id": relation["owner_id"],
                "user_id": relation["owner_id"],
                "source_id": "",
                "doc_id": "",
            })
            sources = [part for part in str(current.get("source_id") or "").split("<SEP>") if part]
            docs = [part for part in str(current.get("doc_id") or "").split("<SEP>") if part]
            if chunk_id not in sources:
                sources.append(chunk_id)
                docs.append(relation["document_id"])
            current["source_id"] = "<SEP>".join(sources)
            current["doc_id"] = "<SEP>".join(docs)
        edges[(src, tgt)] = {
            "relation_type": relation["relation"],
            "kb_id": relation["kb_id"],
            "owner_id": relation["owner_id"],
            "source_id": chunk_id,
            "doc_id": relation["document_id"],
        }
    entities = []
    for node_id, attrs in nodes.items():
        entities.append({
            "chunk_id": node_id,
            "entity_id": node_id,
            "entity_name": attrs["name"],
            "content": attrs["name"],
            "kb_id": attrs["kb_id"],
            "owner_id": attrs["owner_id"],
            "source_id": attrs["source_id"],
            "vector": embed_one(attrs["name"]),
        })
    async def embed_batch(texts: list[str], embed_one=embed_one):
        return np.vstack([embed_one(text) for text in texts])

    rag = _Rag(chunks, entities, _Graph(nodes, edges), embed_batch, embed_one)
    return rag, chunks


def _returned(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in chunks if isinstance(item, dict)]


def _citations(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _returned(chunks)


def _dcg(rels: list[int]) -> float:
    return sum(rel / math.log2(index + 2) for index, rel in enumerate(rels))


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row["answerable"]]
    empty_expected = [row for row in rows if not row["answerable"]]
    recall1 = []
    recall5 = []
    mrr = []
    ndcg = []
    citation_hit = []
    leaks = 0
    hop_hits = []
    for row in answerable:
        expected = set(row["expected_units"])
        cited = _citations(row["chunks"])
        ids = []
        for item in cited:
            unit_id = item.get("unit_id")
            if unit_id not in ids:
                ids.append(unit_id)
        recall1.append(1.0 if ids and ids[0] in expected else 0.0)
        recall5.append(1.0 if any(unit in expected for unit in ids[:5]) else 0.0)
        rank = next((index + 1 for index, unit in enumerate(ids[:5]) if unit in expected), None)
        mrr.append(0.0 if rank is None else 1.0 / rank)
        rels = [1 if unit in expected else 0 for unit in ids[:5]]
        ideal_count = min(5, len(expected))
        ideal = [1] * ideal_count
        ndcg.append(0.0 if not ideal else _dcg(rels) / _dcg(ideal))
        found_units = {item.get("unit_id") for item in cited}
        complete = bool(cited) and expected <= found_units
        scoped = all(
            item.get("owner_id") and item.get("kb_id") and item.get("document_id")
            and item.get("kb_id") == row["kb_id"]
            and (not row.get("owner_id") or item.get("owner_id") == row["owner_id"])
            for item in cited
        )
        citation_hit.append(1.0 if complete and scoped else 0.0)
        if row.get("multihop"):
            units = set(ids)
            path_ok = any(
                item.get("hop") is not None and int(item.get("hop") or 0) >= 2 and all(name in (item.get("path") or "") for name in row.get("path_names") or [])
                for item in row["chunks"]
            )
            hop_hits.append(1.0 if path_ok and expected <= units else 0.0)
    for row in rows:
        expected_owner = row.get("owner_id")
        for item in _returned(row["chunks"]):
            if item.get("kb_id") != row["kb_id"]:
                leaks += 1
            elif expected_owner and item.get("owner_id") != expected_owner:
                leaks += 1
    predicted_empty = [row for row in rows if not _returned(row["chunks"])]
    true_empty = [row for row in predicted_empty if not row["answerable"]]
    precision = 1.0 if not predicted_empty else len(true_empty) / len(predicted_empty)
    recall = 1.0 if not empty_expected else len(true_empty) / len(empty_expected)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    false_returns = 0 if not empty_expected else len([row for row in empty_expected if _returned(row["chunks"])]) / len(empty_expected)
    latencies = [row["latency_ms"] for row in rows]

    def _mean(values: list[float]) -> float | None:
        return None if not values else round(sum(values) / len(values), 4)

    return {
        "count": len(rows),
        "recall_at_1": _mean(recall1),
        "recall_at_5": _mean(recall5),
        "mrr_at_5": _mean(mrr),
        "ndcg_at_5": _mean(ndcg),
        "no_answer_precision": round(precision, 4),
        "no_answer_recall": round(recall, 4),
        "no_answer_f1": round(f1, 4),
        "false_return_rate": round(false_returns, 4),
        "cross_kb_leaks": leaks,
        "citation_accuracy": _mean(citation_hit),
        "multihop_hit_rate": _mean(hop_hits),
        "latency_ms": _summary(latencies),
        "candidate_recall_at_20": _mean([
            1.0 if any(unit in set(row.get("expected_units") or []) for unit in (row.get("candidates") or [])[:20]) else 0.0
            for row in answerable
        ] if any(row.get("candidates") for row in answerable) else []),
        "candidate_recall_at_50": _mean([
            1.0 if any(unit in set(row.get("expected_units") or []) for unit in (row.get("candidates") or [])[:50]) else 0.0
            for row in answerable
        ] if any(row.get("candidates") for row in answerable) else []),
        "query_rewrite_calls": sum(int(row.get("rewrite_calls") or 0) for row in rows),
        "rerank_calls": sum(int(row.get("rerank_calls") or 0) for row in rows),
    }


def _empty_groups() -> dict[str, list[float]]:
    return {"positive": [], "hard_negative": [], "irrelevant": []}


def _observe(question: dict[str, Any], scores: list[float], rows: list[dict[str, Any]], grouped: dict[str, list[float]]) -> None:
    expected = set(question.get("expected_units") or [])
    positives = [score for score, row in zip(scores, rows) if row["unit_id"] in expected]
    hard = [score for score, row in zip(scores, rows) if row["unit_id"] not in expected]
    if question.get("answerable") and positives:
        grouped["positive"].append(max(positives))
    if question.get("answerable"):
        grouped["hard_negative"].append(max(hard) if hard else 0.0)
    if not question.get("answerable"):
        grouped["irrelevant"].append(max(scores) if scores else 0.0)


def _fit_floor(grouped: dict[str, list[float]]) -> tuple[float, str]:
    positive = list(grouped["positive"])
    usable = [value for value in positive if value > 0]
    # 关键词分为 0 表示没有词重合。它不能把门槛拉到 0，否则无关结果会和“没有信号”一起通过。
    scored = usable or positive
    negative = grouped["hard_negative"] + grouped["irrelevant"]
    if scored and negative and min(scored) > max(negative):
        return round((min(scored) + max(negative)) / 2, 4), "验证集有信号的正样本最低分高于负样本最高分，门槛取两者中点。"
    if scored:
        return round(max(0.0, min(scored) - 0.01), 4), "验证集正负样本有重叠，门槛放在有信号的正样本最低分下方。"
    return 0.25, "验证集没有正样本，保留保守门槛。"


def _signal_scores(dataset: dict[str, Any], chunks: list[dict[str, Any]], questions: list[dict[str, Any]], embed_one: Callable[[str], np.ndarray]) -> dict[str, Any]:
    from lightrag.search_strategies import bm25_rank, calibrate_bm25, calibrate_cosine

    grouped = {name: _empty_groups() for name in ("vector", "keyword", "mix", "tree", "graph")}
    hybrid = {key: _empty_groups() for key in ("0", "0.25", "0.5", "0.75", "1")}
    ratios = (0.0, 0.25, 0.5, 0.75, 1.0)
    for question in questions:
        in_kb = [row for row in chunks if row["kb_id"] == question["kb_id"]]
        if not in_kb:
            continue
        query_vector = embed_one(question["query"])
        cosines = [calibrate_cosine(float(np.dot(query_vector, row["vector"]))) for row in in_kb]
        raw_keyword = bm25_rank(question["query"], [row["content"] for row in in_kb])
        keywords = [calibrate_bm25(score) for score in raw_keyword]
        _observe(question, cosines, in_kb, grouped["vector"])
        _observe(question, keywords, in_kb, grouped["keyword"])
        _observe(question, cosines, in_kb, grouped["mix"])
        _observe(question, cosines, in_kb, grouped["tree"])
        linked = [relation for relation in (dataset.get("relations") or []) if relation.get("kb_id") == question["kb_id"]]
        if linked:
            expected = set(question.get("expected_units") or [])
            signals = []
            for relation in linked:
                name_hit = 1.0 if relation["src"] in question["query"] or relation["tgt"] in question["query"] else 0.0
                name_cosine = max(
                    calibrate_cosine(float(np.dot(query_vector, embed_one(relation["src"])))),
                    calibrate_cosine(float(np.dot(query_vector, embed_one(relation["tgt"])))),
                )
                signals.append((max(name_hit, name_cosine), relation["unit_id"] in expected))
            positive_scores = [score for score, matched in signals if matched]
            hard_scores = [score for score, matched in signals if not matched]
            bucket = grouped["graph"]
            if question.get("answerable") and positive_scores:
                bucket["positive"].append(max(positive_scores))
            if question.get("answerable"):
                bucket["hard_negative"].append(max(hard_scores) if hard_scores else 0.0)
            if not question.get("answerable"):
                bucket["irrelevant"].append(max(score for score, _matched in signals))
        else:
            _observe(question, [max(cosine, keyword) for cosine, keyword in zip(cosines, keywords)], in_kb, grouped["graph"])
        for ratio, key in zip(ratios, hybrid):
            blend = [ratio * cosine + (1 - ratio) * keyword for cosine, keyword in zip(cosines, keywords)]
            _observe(question, blend, in_kb, hybrid[key])
    grouped["hybrid"] = hybrid
    return grouped


def recommend_floors(distributions: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rules = {}
    for mode, signal in (
        ("vector", "vector"),
        ("keyword", "keyword"),
        ("mix", "vector"),
        ("tree", "vector"),
        ("graph", "max"),
    ):
        floor, reason = _fit_floor(distributions[mode if mode != "mix" else "mix"])
        if mode == "tree":
            floor, reason = _fit_floor(distributions["tree"])
        if mode == "vector":
            floor, reason = _fit_floor(distributions["vector"])
        rules[mode] = {
            "floor": floor,
            "temperature": 0.06 if mode == "keyword" else 0.08,
            "signal": signal if mode != "keyword" else "keyword",
            "reason": reason + " 该门槛来自这个模式自己的运行信号。",
        }
    rules["keyword"]["signal"] = "keyword"
    rules["graph"]["signal"] = "max"
    floors = {}
    reasons = []
    for key in ("0", "0.25", "0.5", "0.75", "1"):
        floor, reason = _fit_floor(distributions["hybrid"][key])
        floors[key] = floor
        reasons.append(f"{key}={floor}")
    rules["hybrid"] = {
        "floor": floors["0.5"],
        "floors": floors,
        "temperature": 0.08,
        "signal": "blend",
        "reason": "各比例单独使用混合分标定，不用纯向量分布。" + " ".join(reasons),
    }
    return rules


def _profile_rule(rule: dict[str, Any]) -> dict[str, Any]:
    payload = {"floor": rule["floor"], "temperature": rule["temperature"], "signal": rule["signal"]}
    if "floors" in rule:
        payload["floors"] = rule["floors"]
    return payload


def _apply_profile(name: str, rules: dict[str, dict[str, Any]]) -> None:
    from lightrag.retrieval_admission import _profiles

    payload = _profiles()
    payload[name] = {mode: _profile_rule(rule) for mode, rule in rules.items()}


async def _run_split(
    rag,
    chunks,
    questions: list[dict[str, Any]],
    mode: str,
    ratio: float = 0.5,
    enable_rerank: bool = False,
    staged: bool = False,
    query_force: str | None = None,
    expand_parents: bool = True,
    graph_rerank: str = "bundle",
) -> list[dict[str, Any]]:
    from lightrag.search_runtime import run_search_test

    rows = []
    for question in questions:
        kb_id = question["kb_id"]
        visible = [row for row in chunks if row["kb_id"] == kb_id]

        def file_in_kb(file_path: str, record: dict | None = None, expected_kb: str = kb_id) -> bool:
            del file_path
            record = record or {}
            tagged = str(record.get("kb_id") or "")
            return expected_kb in tagged.split("<SEP>") if tagged else False

        started = time.perf_counter()
        result = await run_search_test(
            rag,
            query=question["query"],
            mode=mode,
            top_k=5,
            ratio=ratio,
            enable_rerank=enable_rerank,
            kb_chunks=visible,
            file_in_kb=file_in_kb,
            score_threshold=0.2,
            kb_id=kb_id,
            owner_id=str(question.get("owner_id") or "user-a"),
            staged=staged,
            query_force=query_force,
            expand_parents=expand_parents,
            graph_rerank=graph_rerank,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        plan = result.get("query_plan") or {}
        rows.append({
            **question,
            "chunks": result["chunks"],
            "candidates": result.get("candidates") or [],
            "latency_ms": latency_ms,
            "empty_reason": result.get("empty_reason"),
            "rewrite_calls": 1 if plan.get("llm_used") else 0,
            "rerank_calls": 1 if (result.get("rerank") or {}).get("applied") else 0,
        })
    return rows


def _distribution_summary(distributions: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for name, grouped in distributions.items():
        if name == "hybrid":
            summary[name] = {key: {bucket: _summary(values) for bucket, values in buckets.items()} for key, buckets in grouped.items()}
        else:
            summary[name] = {bucket: _summary(values) for bucket, values in grouped.items()}
    return summary


def evaluate(
    dataset: dict[str, Any],
    embed_one: Callable[[str], np.ndarray] | None = None,
    profile: str = "ci",
    fit_calibration: bool = False,
    tracks: set[str] | None = None,
    enable_rerank: bool = False,
    rerank_rule: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import copy

    from lightrag.retrieval_admission import _profiles

    previous_profile = os.environ.get("RETRIEVAL_PROFILE")
    os.environ["RETRIEVAL_PROFILE"] = profile
    snapshot = copy.deepcopy(_profiles())
    try:
        embed_one = embed_one or _ci_embed_one
        if hasattr(embed_one, "prefetch"):
            texts = [question["query"] for question in dataset["questions"]]
            for doc in dataset["documents"]:
                texts.extend(chunk["text"] for chunk in doc["chunks"])
            for relation in dataset.get("relations") or []:
                texts.extend([relation.get("src") or "", relation.get("tgt") or ""])
            embed_one.prefetch([text for text in texts if text])
        rag, chunks = build_library(dataset, embed_one)
        questions = dataset["questions"]
        if tracks is not None:
            questions = [item for item in questions if item.get("track") in tracks]
        calibration = [item for item in questions if item["split"] == "calibration"]
        holdout = [item for item in questions if item["split"] == "holdout"]
        distributions = _signal_scores(dataset, chunks, calibration, embed_one)
        recommendation = recommend_floors(distributions)
        if fit_calibration or rerank_rule:
            if fit_calibration:
                _apply_profile(profile, recommendation)
            if rerank_rule:
                _profiles().setdefault(profile, {})["rerank"] = {
                    "floor": rerank_rule["floor"],
                    "temperature": rerank_rule["temperature"],
                    "signal": "rerank",
                }
            elif "rerank" in snapshot.get(profile, {}):
                _profiles()[profile]["rerank"] = snapshot[profile]["rerank"]
        by_mode = {}
        hybrid_ratios = {}
        for mode in MODES:
            if mode == "hybrid":
                for key, ratio in (("0", 0.0), ("0.25", 0.25), ("0.5", 0.5), ("0.75", 0.75), ("1", 1.0)):
                    measured = asyncio.run(_run_split(rag, chunks, holdout, mode, ratio=ratio, enable_rerank=enable_rerank))
                    hybrid_ratios[key] = _metrics(measured)
                by_mode[mode] = hybrid_ratios["0.5"]
            else:
                measured = asyncio.run(_run_split(rag, chunks, holdout, mode, enable_rerank=enable_rerank))
                by_mode[mode] = _metrics(measured)
        return {
            "profile": profile,
            "fit_calibration": fit_calibration,
            "rerank": enable_rerank,
            "question_count": len(questions),
            "calibration_count": len(calibration),
            "holdout_count": len(holdout),
            "unanswerable_count": len([item for item in questions if not item["answerable"]]),
            "distributions": _distribution_summary(distributions),
            "recommended_admission": recommendation,
            "holdout": by_mode,
            "hybrid_ratios": hybrid_ratios,
        }
    finally:
        current = _profiles()
        current.clear()
        current.update(snapshot)
        if previous_profile is None:
            os.environ.pop("RETRIEVAL_PROFILE", None)
        else:
            os.environ["RETRIEVAL_PROFILE"] = previous_profile


def _real_embed_one_factory():
    from dotenv import load_dotenv

    load_dotenv(".env", override=False)
    if os.getenv("RUN_REAL_RETRIEVAL_EVAL") != "1":
        raise RuntimeError("真实模型评测需要显式设置 RUN_REAL_RETRIEVAL_EVAL=1")
    key = os.getenv("EMBEDDING_BINDING_API_KEY") or os.getenv("LLM_BINDING_API_KEY") or ""
    model = os.getenv("EMBEDDING_MODEL") or "BAAI/bge-m3"
    host = os.getenv("EMBEDDING_BINDING_HOST") or os.getenv("LLM_BINDING_HOST") or ""
    if not key or not host:
        raise RuntimeError("真实向量模型配置不完整")
    cache: dict[str, np.ndarray] = {}
    dimensions = int(os.getenv("EMBEDDING_DIM", "1024") or "1024")

    def _fetch(texts: list[str]):
        from lightrag.llm.openai import openai_embed

        async def _call():
            return await openai_embed(texts, model=model, base_url=host, api_key=key, dimensions=dimensions)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_call())
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(_call())).result()

    def embed_many(texts: list[str]) -> None:
        missing = [text for text in dict.fromkeys(texts) if text not in cache]
        for start in range(0, len(missing), 16):
            batch = missing[start:start + 16]
            matrix = _fetch(batch)
            for text, row in zip(batch, matrix):
                vector = np.asarray(row, dtype=np.float32)
                cache[text] = vector / (float(np.linalg.norm(vector)) or 1.0)

    def embed_one(text: str) -> np.ndarray:
        if text not in cache:
            embed_many([text])
        return cache[text]

    embed_one.prefetch = embed_many
    return embed_one, model


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 检索评测",
        "",
        f"- 画像：{report['profile']}",
        f"- 问题数：{report['question_count']}，其中验证集 {report['calibration_count']}，留出集 {report['holdout_count']}",
        f"- 无法回答：{report['unanswerable_count']}",
        "",
        "## 验证集分数分布",
        "",
    ]
    for name, grouped in report["distributions"].items():
        lines.append(f"### {name}")
        for key, summary in grouped.items():
            lines.append(f"- {key}：{summary}")
        lines.append("")
    lines.append("## 推荐准入")
    lines.append("")
    for mode, rule in report["recommended_admission"].items():
        lines.append(f"- {mode}：floor {rule['floor']}，temperature {rule['temperature']}，signal {rule['signal']}。{rule['reason']}")
    lines.append("")
    lines.append("## 留出集")
    lines.append("")
    for mode, metrics in report["holdout"].items():
        lines.append(f"### {mode}")
        for key, value in metrics.items():
            lines.append(f"- {key}：{value}")
        lines.append("")
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")


def profile_name_for_embedding(model: str) -> str:
    from lightrag.retrieval_admission import profile_name_for_model

    matched = profile_name_for_model(model)
    if matched:
        return matched
    if "qwen3-embedding-4b" in model.lower():
        return "qwen3-embedding-4b"
    if "bge-m3" in model.lower():
        return "bge-m3"
    return "real"


def _install_rerank_cache():
    import lightrag.rerank as rerank_module
    import lightrag.search_runtime as runtime_module

    original = rerank_module.siliconflow_rerank
    cache: dict[tuple, list] = {}

    async def cached(**kwargs):
        key = (kwargs.get("query"), tuple(kwargs.get("documents") or []))
        if key not in cache:
            cache[key] = await original(**kwargs)
        return cache[key]

    rerank_module.siliconflow_rerank = cached
    runtime_module.siliconflow_rerank = cached


def fit_rerank_rule(dataset: dict[str, Any], questions: list[dict[str, Any]]) -> dict[str, Any]:
    from lightrag.rerank import siliconflow_rerank

    grouped = _empty_groups()

    async def _score(query: str, documents: list[str]) -> list[float]:
        ranked = await siliconflow_rerank(query=query, documents=documents, top_n=len(documents))
        scores = [0.0] * len(documents)
        for item in ranked:
            index = item.get("index")
            if isinstance(index, int) and 0 <= index < len(scores):
                scores[index] = float(item.get("relevance_score") or 0)
        return scores

    for question in questions:
        documents = []
        rows = []
        for doc in dataset["documents"]:
            if doc["kb_id"] != question["kb_id"]:
                continue
            for chunk in doc["chunks"]:
                documents.append(chunk["text"])
                rows.append({"unit_id": chunk["unit_id"]})
        if not documents:
            continue
        scores = asyncio.run(_score(question["query"], documents))
        _observe(question, scores, rows, grouped)
    floor, reason = _fit_floor(grouped)
    return {
        "floor": floor,
        "temperature": 0.05,
        "signal": "rerank",
        "reason": "重排分使用验证集单独标定。" + reason,
        "distributions": {key: _summary(values) for key, values in grouped.items()},
    }


def _safe_error(exc: BaseException) -> str:
    text = str(exc)
    return text.replace(os.getenv("RERANK_BINDING_API_KEY", " "), "***").replace(os.getenv("EMBEDDING_BINDING_API_KEY", " "), "***")[:300]


def evaluate_ablation(
    dataset: dict[str, Any],
    embed_one: Callable[[str], np.ndarray] | None = None,
    profile: str = "ci",
    questions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """同一批问题的消融。重排关闭时不会访问重排服务。"""
    from lightrag.search_strategies import tokenize
    import lightrag.rerank as rerank_module
    import lightrag.search_runtime as runtime_module

    previous = os.environ.get("RETRIEVAL_PROFILE")
    os.environ["RETRIEVAL_PROFILE"] = profile
    original_rerank = runtime_module.siliconflow_rerank

    async def _local_rerank(query: str, documents: list[str], top_n: int | None = None, **kwargs):
        del kwargs
        query_tokens = set(tokenize(query))
        scored = []
        for index, document in enumerate(documents):
            tokens = set(tokenize(document))
            score = 0.0 if not query_tokens else len(query_tokens & tokens) / len(query_tokens)
            scored.append({"index": index, "relevance_score": score})
        scored.sort(key=lambda item: (-item["relevance_score"], item["index"]))
        if top_n:
            scored = scored[:top_n]
        return scored

    runtime_module.siliconflow_rerank = _local_rerank
    rerank_module.siliconflow_rerank = _local_rerank
    try:
        embed_one = embed_one or _ci_embed_one
        rag, chunks = build_library(dataset, embed_one)
        selected = questions if questions is not None else [item for item in dataset["questions"] if item["split"] == "holdout"]
        specs = [
            ("single_query", {"mode": "vector", "staged": True, "query_force": "single", "enable_rerank": False}),
            ("multi_query", {"mode": "vector", "staged": True, "query_force": "multi", "enable_rerank": False}),
            ("hyde", {"mode": "vector", "staged": True, "query_force": "hyde", "enable_rerank": False}),
            ("subquestions", {"mode": "vector", "staged": True, "query_force": "subquestions", "enable_rerank": False}),
            ("rerank_off", {"mode": "mix", "staged": True, "query_force": "single", "enable_rerank": False}),
            ("rerank_on", {"mode": "mix", "staged": True, "query_force": "single", "enable_rerank": True}),
            ("parent_off", {"mode": "vector", "staged": True, "query_force": "single", "expand_parents": False}),
            ("parent_on", {"mode": "vector", "staged": True, "query_force": "single", "expand_parents": True}),
            ("graph_chunk", {"mode": "graph", "staged": True, "query_force": "single", "enable_rerank": True, "graph_rerank": "chunk"}),
            ("graph_bundle", {"mode": "graph", "staged": True, "query_force": "single", "enable_rerank": True, "graph_rerank": "bundle"}),
            ("vector", {"mode": "vector", "staged": True, "query_force": "single", "enable_rerank": False}),
            ("hybrid", {"mode": "hybrid", "staged": True, "query_force": "single", "enable_rerank": False}),
            ("tree", {"mode": "tree", "staged": True, "query_force": "single", "enable_rerank": False}),
        ]
        report = {}
        for name, options in specs:
            rows = asyncio.run(_run_split(rag, chunks, selected, **options))
            report[name] = _metrics(rows)
        return report
    finally:
        runtime_module.siliconflow_rerank = original_rerank
        rerank_module.siliconflow_rerank = original_rerank
        if previous is None:
            os.environ.pop("RETRIEVAL_PROFILE", None)
        else:
            os.environ["RETRIEVAL_PROFILE"] = previous


def main() -> None:
    parser = argparse.ArgumentParser(description="个人知识库检索评测")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--real-model", action="store_true")
    parser.add_argument("--output", default="")
    parser.add_argument("--write-profile", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()
    dataset = load_dataset(Path(args.dataset))
    if args.ablation and not args.real_model:
        report = {"profile": "ci", "ablation": evaluate_ablation(dataset, profile="ci")}
        output = Path(args.output) if args.output else Path("eval_report")
        output.parent.mkdir(parents=True, exist_ok=True)
        destination = output if output.suffix == ".json" else output.with_suffix(".json")
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(destination)
        return
    if args.real_model:
        embed_one, model = _real_embed_one_factory()
        _install_rerank_cache()
        profile = profile_name_for_embedding(model)
        calibration = [item for item in dataset["questions"] if item["split"] == "calibration"]
        try:
            rerank_rule = fit_rerank_rule(dataset, calibration)
        except Exception as exc:
            raise RuntimeError("重排标定失败：" + _safe_error(exc)) from exc
        report = evaluate(dataset, embed_one=embed_one, profile=profile, fit_calibration=True, rerank_rule=rerank_rule)
        report_on = evaluate(
            dataset,
            embed_one=embed_one,
            profile=profile,
            fit_calibration=True,
            enable_rerank=True,
            rerank_rule=rerank_rule,
        )
        report["embedding_model"] = model
        report["rerank_model"] = os.getenv("RERANK_MODEL", "")
        report["rerank_rule"] = rerank_rule
        report["rerank_off"] = {
            "holdout": report["holdout"],
            "hybrid_ratios": report["hybrid_ratios"],
        }
        report["rerank_on"] = {
            "holdout": report_on["holdout"],
            "hybrid_ratios": report_on["hybrid_ratios"],
        }
        report["recommended_admission"]["rerank"] = rerank_rule
    else:
        report = evaluate(dataset, profile="ci", fit_calibration=True)
    if args.write_profile:
        from lightrag.retrieval_admission import _PROFILE_PATH, _profiles

        payload = _profiles()
        payload[report["profile"]] = {
            mode: _profile_rule(rule)
            for mode, rule in report["recommended_admission"].items()
            if mode != "rerank"
        }
        if "rerank" in report.get("recommended_admission", {}):
            payload[report["profile"]]["rerank"] = _profile_rule(report["recommended_admission"]["rerank"])
        _PROFILE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _profiles.cache_clear()
    output = Path(args.output) if args.output else Path("eval_report")
    write_report(report, output)
    print(output.with_suffix(".json"))


if __name__ == "__main__":
    main()
