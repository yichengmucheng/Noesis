"""开放三元组抽取、同义边和个性化 PageRank。

不规定实体类型和关系名。每个文本块抽出原文中的短语和它们之间的说法，
向量足够近且图上还没有边的短语补一条「同义」边。提问时用向量找到入口短语，
再在整张图上做 PageRank，按分数把相关原文块带回来。
"""

from __future__ import annotations

import os
from typing import Any

from lightrag.utils import compute_mdhash_id, logger

SYNONYM_KEYWORD = "同义"
DEFAULT_SYNONYM_THRESHOLD = 0.8
DEFAULT_PPR_ALPHA = 0.85
DEFAULT_PPR_STEPS = 20


def open_ie_enabled(global_config: dict | None = None) -> bool:
    if global_config:
        addon = global_config.get("addon_params") or {}
        if isinstance(addon, dict) and addon.get("kg_schema"):
            return str(addon["kg_schema"]).lower() == "open"
    return os.getenv("KG_SCHEMA", "open").lower() != "fault"


def synonym_threshold() -> float:
    raw = os.getenv("OPEN_IE_SYNONYM_THRESHOLD", str(DEFAULT_SYNONYM_THRESHOLD))
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_SYNONYM_THRESHOLD


def synonym_targets(
    entity_name: str,
    hits: list[tuple[str, float]],
    threshold: float,
    limit: int = 3,
) -> list[tuple[str, float]]:
    """从向量近邻里挑出可以连同义边的短语。相似度越大越近。"""
    ranked = sorted(hits, key=lambda item: item[1], reverse=True)
    own = next((score for name, score in ranked if name == entity_name), None)
    if own is not None and own < 0.5:
        ranked = [(name, 1.0 - score) for name, score in ranked]
        ranked.sort(key=lambda item: item[1], reverse=True)

    picked: list[tuple[str, float]] = []
    seen: set[str] = set()
    for name, score in ranked:
        if not name or name == entity_name or name in seen:
            continue
        if score < threshold:
            continue
        seen.add(name)
        picked.append((name, score))
        if len(picked) >= limit:
            break
    return picked


def personalized_pagerank(
    neighbors: dict[str, list[str]],
    personalization: dict[str, float],
    alpha: float = DEFAULT_PPR_ALPHA,
    steps: int = DEFAULT_PPR_STEPS,
) -> dict[str, float]:
    """沿边扩散入口短语的权重。alpha 是继续走向邻居的概率。"""
    nodes: set[str] = set(personalization)
    for source, targets in neighbors.items():
        nodes.add(source)
        nodes.update(targets)
    if not nodes:
        return {}

    total = sum(max(weight, 0.0) for weight in personalization.values())
    if total <= 0:
        seed = {name: 1.0 / len(personalization) for name in personalization}
    else:
        seed = {
            name: max(weight, 0.0) / total for name, weight in personalization.items()
        }

    score = {name: seed.get(name, 0.0) for name in nodes}
    teleport = 1.0 - alpha
    for _ in range(steps):
        nxt = {name: teleport * seed.get(name, 0.0) for name in nodes}
        for source, targets in neighbors.items():
            if source not in score or not targets:
                continue
            share = alpha * score[source] / len(targets)
            for target in targets:
                if target in nxt:
                    nxt[target] += share
        dangling = sum(score[name] for name in nodes if not neighbors.get(name))
        if dangling:
            dangling_share = alpha * dangling
            for name in nodes:
                nxt[name] += dangling_share * seed.get(name, 0.0)
        score = nxt
    return score


OPEN_PROMPTS: dict[str, Any] = {
    "entity_extraction_system_prompt": """---角色---
你从个人资料中抽取开放三元组。不要使用预设类型表。

---规则---
1. 实体名称必须沿用原文，不改写成近义词，不使用代词。
2. 实体类型一律写「短语」。
3. 关系词使用原文里的说法，例如「就职于」「计划在」「提到」。不要换成统一词表。
4. 只抽取原文里写明的关系。原文没有写出的关系不要补。
5. 输出格式：
实体：entity{tuple_delimiter}实体名称{tuple_delimiter}短语{tuple_delimiter}实体描述
关系：relation{tuple_delimiter}源实体{tuple_delimiter}目标实体{tuple_delimiter}关系词{tuple_delimiter}关系说明
6. 先输出全部实体，再输出全部关系。语言为{language}。
7. 完成后单独一行输出 {completion_delimiter}。

---示例---
{examples}

---文本---
{input_text}
""",
    "entity_extraction_user_prompt": """请从下面文本抽取开放三元组。实体类型只写「短语」。关系词沿用原文说法。

{input_text}

先实体后关系。最后一行是 {completion_delimiter}。
""",
    "entity_continue_extraction_user_prompt": """只补充上一轮漏掉的实体和关系。实体类型只写「短语」。不要重复已经正确的条目。

最后一行是 {completion_delimiter}。
""",
    "entity_extraction_examples": [
        """entity{tuple_delimiter}文成{tuple_delimiter}短语{tuple_delimiter}笔记作者，天津财经大学大数据专业
entity{tuple_delimiter}百事通{tuple_delimiter}短语{tuple_delimiter}集成在飞书里的助手
relation{tuple_delimiter}文成{tuple_delimiter}百事通{tuple_delimiter}负责{tuple_delimiter}笔记写明文成负责百事通的记忆和评测
{completion_delimiter}"""
    ],
}


async def link_synonym_edges(
    entity_names: list[str],
    knowledge_graph_inst,
    entity_vdb,
    relationships_vdb,
    threshold: float | None = None,
) -> int:
    """给新短语和已有近义短语补边。已有其他关系的两点不再覆盖。"""
    if entity_vdb is None or not entity_names:
        return 0
    cutoff = synonym_threshold() if threshold is None else threshold
    added = 0
    for entity_name in entity_names:
        if not entity_name or len(entity_name.strip()) < 2:
            continue
        try:
            hits = await entity_vdb.query(entity_name, top_k=8)
        except Exception as exc:
            logger.warning(f"同义检索失败 {entity_name}: {exc}")
            continue
        pairs = [
            (hit.get("entity_name") or "", float(hit.get("distance") or 0.0))
            for hit in hits
        ]
        for other, score in synonym_targets(entity_name, pairs, cutoff):
            if not await knowledge_graph_inst.has_node(other):
                continue
            if await knowledge_graph_inst.has_edge(entity_name, other):
                continue
            description = f"{entity_name} 与 {other} 在向量空间中指代接近"
            await knowledge_graph_inst.upsert_edge(
                entity_name,
                other,
                edge_data={
                    "weight": float(score),
                    "description": description,
                    "keywords": SYNONYM_KEYWORD,
                    "source_id": "synonym",
                    "file_path": "synonym",
                },
            )
            if relationships_vdb is not None:
                rel_id = compute_mdhash_id(
                    f"{entity_name}|{other}|{SYNONYM_KEYWORD}", prefix="rel-"
                )
                await relationships_vdb.upsert(
                    {
                        rel_id: {
                            "src_id": entity_name,
                            "tgt_id": other,
                            "source_id": "synonym",
                            "content": f"{SYNONYM_KEYWORD}\t{entity_name}\n{other}\n{description}",
                            "keywords": SYNONYM_KEYWORD,
                            "description": description,
                            "weight": float(score),
                            "file_path": "synonym",
                        }
                    }
                )
            added += 1
            logger.info(f"同义边 {entity_name} — {other} ({score:.3f})")
    return added


async def retrieve_by_pagerank(
    query: str,
    knowledge_graph_inst,
    entities_vdb,
    query_embedding,
    top_k: int,
) -> tuple[list[dict], list[dict]]:
    """向量找到入口短语，PageRank 把图上相连的短语和关系排出来。"""
    hits = await entities_vdb.query(
        query, top_k=max(top_k, 10), query_embedding=query_embedding
    )
    personalization: dict[str, float] = {}
    for hit in hits:
        name = hit.get("entity_name")
        if not name:
            continue
        personalization[name] = max(
            personalization.get(name, 0.0), float(hit.get("distance") or 0.0)
        )
    if not personalization:
        return [], []

    graph_edges = await knowledge_graph_inst.get_all_edges()
    neighbors: dict[str, list[str]] = {}
    for edge in graph_edges:
        source = edge.get("source")
        target = edge.get("target")
        if not source or not target:
            continue
        neighbors.setdefault(source, []).append(target)
        neighbors.setdefault(target, []).append(source)

    scores = personalized_pagerank(neighbors, personalization)
    ranked_names = sorted(scores, key=lambda name: scores[name], reverse=True)[:top_k]
    nodes = await knowledge_graph_inst.get_nodes_batch(ranked_names)
    entities = []
    for name in ranked_names:
        node = nodes.get(name)
        if not node:
            continue
        entities.append(
            {
                **node,
                "entity_name": name,
                "rank": scores[name],
                "source_id": node.get("source_id") or "",
            }
        )

    keep = {item["entity_name"] for item in entities}
    relations = []
    for edge in graph_edges:
        source = edge.get("source")
        target = edge.get("target")
        if source not in keep and target not in keep:
            continue
        relations.append(
            {
                "src_id": source,
                "tgt_id": target,
                "src_tgt": tuple(sorted([source, target])),
                "description": edge.get("description") or "",
                "keywords": edge.get("keywords") or "",
                "weight": float(edge.get("weight") or 1.0),
                "source_id": edge.get("source_id") or "",
                "file_path": edge.get("file_path") or "",
                "rank": scores.get(source, 0.0) + scores.get(target, 0.0),
            }
        )
    relations.sort(key=lambda item: item["rank"], reverse=True)
    return entities, relations[: max(top_k * 2, top_k)]
