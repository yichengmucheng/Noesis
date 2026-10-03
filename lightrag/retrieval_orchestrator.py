"""两阶段检索的纯函数：RRF、路径包、父块聚合。

渠道原始分不直接相加。路径准入失败时，整条路径的证据一起丢弃。
"""

from __future__ import annotations

from typing import Any

from lightrag.chunk_hierarchy import count_tokens

DENSE_TOP = 40
BM25_TOP = 30
GRAPH_TOP = 20
RRF_CAP = 60
RERANK_POOL = 40
FINAL_CHILDREN_MIN = 5
FINAL_CHILDREN_MAX = 8
FINAL_PARENTS_MIN = 3
FINAL_PARENTS_MAX = 6
TOKEN_BUDGET = 4500
RRF_K = 60


def final_child_limit(top_k: int) -> int:
    limit = max(1, int(top_k or FINAL_CHILDREN_MIN))
    if limit < FINAL_CHILDREN_MIN:
        return limit
    return min(FINAL_CHILDREN_MAX, limit)


def rerank_document(hit: dict[str, Any]) -> str:
    ready = str(hit.get("index_text") or "").strip()
    if ready:
        return ready[:1800]
    path = hit.get("section_path") or []
    if isinstance(path, str):
        heading = path
    else:
        heading = " / ".join(str(part) for part in path if str(part).strip())
    return "\n".join(
        part
        for part in (
            str(hit.get("doc_name") or ""),
            heading,
            str(hit.get("content") or ""),
        )
        if part
    ).strip()[:1800]


def rrf_merge(
    channels: dict[str, list[dict[str, Any]]],
    *,
    limit: int = RRF_CAP,
    id_key: str = "chunk_id",
) -> list[dict[str, Any]]:
    """按名次融合。不把各渠道的 raw_score 加在一起。"""
    scores: dict[str, float] = {}
    best: dict[str, dict[str, Any]] = {}
    for channel, hits in channels.items():
        for rank, hit in enumerate(hits, start=1):
            identity = str(hit.get(id_key) or "")
            if not identity:
                continue
            scores[identity] = scores.get(identity, 0.0) + 1.0 / (RRF_K + rank)
            row = best.get(identity)
            if row is None:
                row = dict(hit)
                row["channel"] = hit.get("channel") or channel
                row["channel_rank"] = hit.get("channel_rank") or rank
                row["channels"] = []
                best[identity] = row
            else:
                for field in ("vector_score", "keyword_score"):
                    incoming = hit.get(field)
                    if incoming is None:
                        continue
                    current = row.get(field)
                    row[field] = (
                        float(incoming)
                        if current is None
                        else max(float(current), float(incoming))
                    )
            row["channels"].append(
                {
                    "channel": channel,
                    "channel_rank": rank,
                    "raw_score": hit.get("raw_score"),
                    "query_variant_id": hit.get("query_variant_id") or "",
                }
            )
            row["rrf_score"] = scores[identity]
    ordered = sorted(
        best.values(),
        key=lambda item: (
            -float(item.get("rrf_score") or 0),
            str(item.get(id_key) or ""),
        ),
    )
    return ordered[:limit]


def path_bundles(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """一条路径上的证据块组成一个包。短路径若已被更长路径覆盖，不再单独成包。"""
    if not hits:
        return []
    ordered = sorted(hits, key=lambda item: -int(item.get("hop") or 0))
    used: set[str] = set()
    bundles: list[dict[str, Any]] = []
    for anchor in ordered:
        path = str(anchor.get("path") or anchor.get("path_id") or "")
        if not path:
            continue
        members = []
        for hit in hits:
            chunk_id = str(hit.get("chunk_id") or "")
            hit_path = str(hit.get("path") or hit.get("path_id") or "")
            if not chunk_id or chunk_id in used:
                continue
            if hit_path and (path == hit_path or path.startswith(hit_path)):
                members.append(hit)
        if not members:
            continue
        for item in members:
            used.add(str(item.get("chunk_id") or ""))
        bundles.append(_bundle(path, members))
    for hit in hits:
        chunk_id = str(hit.get("chunk_id") or "")
        if not chunk_id or chunk_id in used:
            continue
        used.add(chunk_id)
        bundles.append(_bundle(str(hit.get("path_id") or chunk_id), [hit]))
    return bundles


def _bundle(path_id: str, members: list[dict[str, Any]]) -> dict[str, Any]:
    text = "\n".join(rerank_document(item) for item in members)
    return {
        "path_id": path_id,
        "members": [dict(item) for item in members],
        "content": text[:1800],
        "channel": "graph",
        "raw_score": max(float(item.get("raw_score") or 0) for item in members),
        "chunk_id": path_id,
    }


def expand_admitted_bundles(bundles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for bundle in bundles:
        for item in bundle.get("members") or []:
            chunk_id = str(item.get("chunk_id") or "")
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            row = dict(item)
            row["path_id"] = bundle.get("path_id") or row.get("path_id") or ""
            row["channel"] = row.get("channel") or "graph"
            if bundle.get("rerank_score") is not None:
                row["rerank_score"] = bundle.get("rerank_score")
                row["reranked"] = True
            if bundle.get("admission_score") is not None:
                row["admission_score"] = bundle.get("admission_score")
                row["calibrated_score"] = bundle.get("admission_score")
                row["untrusted"] = False
            rows.append(row)
    return rows


def fuse_ranked_groups(
    groups: list[list[dict[str, Any]]], *, id_key: str = "chunk_id"
) -> list[dict[str, Any]]:
    named = {f"group-{index}": rows for index, rows in enumerate(groups) if rows}
    return rrf_merge(named, limit=RRF_CAP, id_key=id_key)


def aggregate_parents(
    hits: list[dict[str, Any]],
    parents: dict[str, dict[str, Any]],
    *,
    max_children: int,
    max_parents: int = FINAL_PARENTS_MAX,
    token_budget: int = TOKEN_BUDGET,
    enabled: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """同一父块的正文只进入上下文一次。引用仍留在命中的小块上。"""
    if not enabled:
        return hits[:max_children], []
    selected: list[dict[str, Any]] = []
    parent_order: list[str] = []
    used_tokens = 0
    per_doc: dict[str, int] = {}
    documents = {
        str(item.get("document_id") or item.get("doc_id") or "")
        for item in hits
        if item.get("document_id") or item.get("doc_id")
    }
    for hit in hits:
        parent_id = str(hit.get("parent_id") or "")
        invented = False
        if not parent_id:
            parent_id = str(hit.get("chunk_id") or "")
            invented = True
        if parent_id not in parent_order:
            if len(parent_order) >= max_parents:
                continue
            parent = None if invented else parents.get(parent_id)
            extra = (
                count_tokens(str(parent.get("content") or ""))
                if parent
                else count_tokens(str(hit.get("content") or ""))
            )
            if parent_order and used_tokens + extra > token_budget:
                continue
            parent_order.append(parent_id)
            used_tokens += extra
        if len(selected) >= max_children:
            break
        doc_id = str(hit.get("document_id") or hit.get("doc_id") or "")
        if doc_id and per_doc.get(doc_id, 0) >= 3 and len(documents) > 1:
            continue
        if doc_id:
            per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
        selected.append(hit)
    contexts: list[dict[str, Any]] = []
    for parent_id in parent_order:
        members = [
            item
            for item in selected
            if str(item.get("parent_id") or item.get("chunk_id") or "") == parent_id
        ]
        if not members:
            continue
        parent = parents.get(parent_id)
        if parent and str(members[0].get("parent_id") or "") == parent_id:
            content = str(parent.get("content") or "")
        else:
            content = str(members[0].get("content") or "")
        contexts.append(
            {
                "parent_id": parent_id,
                "content": content,
                "chunk_ids": [str(item.get("chunk_id") or "") for item in members],
            }
        )
    return selected, contexts
