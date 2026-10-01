"""检索测试用的纯函数：分词、BM25、加权融合、图谱多跳、重排回写。"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


_LATIN = re.compile(r"[A-Za-z0-9_]+")
_CJK = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """英文按词，中文按二字切。不依赖分词器。"""
    if not text:
        return []
    tokens: list[str] = []
    cjk: list[str] = []

    def flush_cjk() -> None:
        if not cjk:
            return
        if len(cjk) == 1:
            tokens.append(cjk[0])
        else:
            tokens.extend(a + b for a, b in zip(cjk, cjk[1:]))
        cjk.clear()

    for piece in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower()):
        if _LATIN.fullmatch(piece):
            flush_cjk()
            tokens.append(piece)
        elif _CJK.fullmatch(piece):
            cjk.append(piece)
    flush_cjk()
    return tokens


def bm25_rank(query: str, documents: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Okapi BM25。文档为空时对应分数为 0。"""
    if not documents:
        return []
    tokenized = [tokenize(doc) for doc in documents]
    query_terms = tokenize(query)
    if not query_terms:
        return [0.0] * len(documents)
    total = len(documents)
    avgdl = sum(len(tokens) for tokens in tokenized) / total
    doc_freq: Counter[str] = Counter()
    for tokens in tokenized:
        doc_freq.update(set(tokens))
    scores: list[float] = []
    for tokens in tokenized:
        tf = Counter(tokens)
        length = len(tokens) or 1
        score = 0.0
        for term in query_terms:
            freq = tf.get(term, 0)
            if not freq:
                continue
            df = doc_freq[term]
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            denom = freq + k1 * (1 - b + b * length / (avgdl or 1))
            score += idf * (freq * (k1 + 1)) / denom
        scores.append(score)
    return scores


# BM25 没有上界。固定尺度从 0 开始，全程单调递增，不看这一批里的最高分和最低分。
_KEYWORD_SCALE = 4.0


def calibrate_cosine(value: float) -> float:
    """向量余弦校准。负相关记为 0，[0, 1] 保持原值，超出 1 的数值钳到 1。"""
    if value <= 0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def calibrate_bm25(value: float) -> float:
    """BM25 固定转换。0 对应 0，任意更大的原始分都严格更大。"""
    if value <= 0:
        return 0.0
    score = float(value)
    return score / (score + _KEYWORD_SCALE)


def absolute_relevance(value: float) -> float:
    """未标明来源时：不超过 1 的值按余弦保留，更大的值按 BM25 转换。"""
    if value <= 1.0:
        return calibrate_cosine(value)
    return calibrate_bm25(value)


def _normalize(values: list[float]) -> list[float]:
    """只做绝对刻度。不再用候选集合内部的 min-max。"""
    return [absolute_relevance(float(value or 0)) for value in values]


def admission_value(item: dict[str, Any]) -> float | None:
    """是否返回这条结果。没有可信相关度时返回 None，调用方必须丢掉。"""
    if item.get("untrusted"):
        return None
    decision = item.get("admission_score")
    if decision is not None and decision != "":
        try:
            return float(decision)
        except (TypeError, ValueError):
            return None
    calibrated = item.get("calibrated_score")
    if calibrated is not None and calibrated != "":
        try:
            return float(calibrated)
        except (TypeError, ValueError):
            return None
    if item.get("reranked"):
        rerank = item.get("rerank_score")
        if rerank is None or rerank == "":
            return None
        try:
            return float(rerank)
        except (TypeError, ValueError):
            return None
    raw = item.get("score")
    if raw is None or raw == "":
        return None
    try:
        return absolute_relevance(float(raw))
    except (TypeError, ValueError):
        return None


def filter_by_threshold(hits: list[dict[str, Any]], threshold: float | None) -> list[dict[str, Any]]:
    """用绝对相关度决定是否返回。排序分不能代替准入分。"""
    if threshold is None or not hits:
        return hits
    limit = min(1.0, max(0.0, float(threshold)))
    if limit <= 0:
        return hits
    kept: list[dict[str, Any]] = []
    for item in hits:
        score = admission_value(item)
        if score is None:
            continue
        cutoff = limit
        if item.get("admission_score") is not None and item.get("admission_score") != "":
            cutoff = 0.5 if limit <= 0.5 else limit
        if score < cutoff:
            continue
        kept.append(item)
    return kept


def weighted_fuse(
    vector_hits: list[dict[str, Any]],
    keyword_hits: list[dict[str, Any]],
    ratio: float,
    id_key: str = "chunk_id",
) -> list[dict[str, Any]]:
    """按「向量:关键词」比例融合两边的绝对分数。ratio 是向量权重。"""
    ratio = min(1.0, max(0.0, float(ratio)))

    def _vector_of(item: dict[str, Any]) -> float:
        if item.get("vector_score") is not None and item.get("vector_score") != "":
            return calibrate_cosine(float(item.get("vector_score") or 0))
        return calibrate_cosine(float(item.get("score") or 0))

    def _keyword_of(item: dict[str, Any]) -> float:
        if item.get("keyword_score") is not None and item.get("keyword_score") != "":
            return float(item.get("keyword_score") or 0)
        return calibrate_bm25(float(item.get("score") or 0))

    merged: dict[str, dict[str, Any]] = {}
    for item in vector_hits:
        key = str(item.get(id_key) or "")
        if not key:
            continue
        row = dict(item)
        row["vector_score"] = _vector_of(item)
        row["keyword_score"] = 0.0
        row["source"] = "vector"
        row["score_type"] = "vector"
        merged[key] = row
    for item in keyword_hits:
        key = str(item.get(id_key) or "")
        if not key:
            continue
        keyword = _keyword_of(item)
        if key in merged:
            merged[key]["keyword_score"] = keyword
            merged[key]["source"] = "hybrid"
            merged[key]["score_type"] = "hybrid"
            if item.get("raw_score") is not None:
                merged[key]["keyword_raw"] = item.get("raw_score")
        else:
            row = dict(item)
            row["vector_score"] = 0.0
            row["keyword_score"] = keyword
            row["source"] = "keyword"
            row["score_type"] = "keyword"
            merged[key] = row
    ranked: list[dict[str, Any]] = []
    for row in merged.values():
        vector = float(row.get("vector_score") or 0)
        keyword = float(row.get("keyword_score") or 0)
        blended = ratio * vector + (1 - ratio) * keyword
        row["score"] = blended
        row["calibrated_score"] = blended
        if row.get("raw_score") is None:
            row["raw_score"] = blended
        ranked.append(row)
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def expand_paths(
    seeds: list[tuple[str, float]],
    adjacency: dict[str, list[tuple[str, str]]],
    hops: int = 2,
    decay: float = 0.72,
) -> list[dict[str, Any]]:
    """从种子实体沿边扩展。adjacency[节点] = [(邻居, 关系名)]。"""
    best: dict[str, dict[str, Any]] = {}
    queue: list[tuple[str, int, float, list[str]]] = [
        (name, 0, score, [name]) for name, score in seeds if name
    ]
    while queue:
        name, hop, score, path = queue.pop(0)
        current = best.get(name)
        if current is not None and current["hop"] <= hop:
            continue
        best[name] = {
            "name": name,
            "hop": hop,
            "score": score * (decay ** hop),
            "path": path,
        }
        if hop >= hops:
            continue
        for neighbor, label in adjacency.get(name) or []:
            if not neighbor or neighbor in path:
                continue
            queue.append((neighbor, hop + 1, score, path + [label or "相关", neighbor]))
    rows = list(best.values())
    rows.sort(key=lambda item: (-item["score"], item["hop"], item["name"]))
    return rows


def reorder_by_rerank(
    hits: list[dict[str, Any]], ranked: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """按交叉编码器返回的 index / relevance_score 重排。"""
    ordered: list[dict[str, Any]] = []
    for item in ranked:
        index = item.get("index")
        if not isinstance(index, int) or not 0 <= index < len(hits):
            continue
        relevance = item.get("relevance_score")
        if relevance is None or relevance == "":
            continue
        row = dict(hits[index])
        row["rerank_score"] = float(relevance)
        row["calibrated_score"] = float(relevance)
        row["score"] = float(relevance)
        row["score_type"] = "rerank"
        if row.get("raw_score") is None:
            row["raw_score"] = float(relevance)
        row["reranked"] = True
        row["untrusted"] = False
        ordered.append(row)
    return ordered
