"""三套存储共用的标识。

Neo4j 保存实体和关系，用 entity_id、relation_id 做多跳。
Milvus 保存文本块、实体和关系的向量，主键分别是 chunk_id、entity_id、relation_id。
PostgreSQL 保存正文、目录和来源对照；原始文件路径使用 user_id/kb_id/doc_id。
source_id 等于产生这条证据的 chunk_id。显示名称不是物理主键。
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

_SPACE = re.compile(r"\s+")


def _norm(value: str) -> str:
    return _SPACE.sub(" ", str(value or "").strip().lower())


def _digest(*parts: str) -> str:
    raw = "\n".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def require_scope(user_id: str, kb_id: str) -> tuple[str, str]:
    user = str(user_id or "").strip()
    kb = str(kb_id or "").strip()
    if not user or not kb:
        raise ValueError("user_id 和 kb_id 必须同时存在")
    return user, kb


def chunk_id(doc_id: str, index: int) -> str:
    doc = str(doc_id or "").strip()
    if not doc:
        raise ValueError("doc_id 不能为空")
    if index < 1:
        raise ValueError("chunk 序号从 1 开始")
    return f"{doc}-c{index:04d}"


def entity_id(kb_id: str, entity_type: str, name: str, model: str = "") -> str:
    kb = str(kb_id or "").strip()
    kind = _norm(entity_type)
    label = _norm(name)
    if not kb or not kind or not label:
        raise ValueError("entity_id 需要 kb_id、实体类型和名称")
    return "ent-" + _digest(kb, kind, label, _norm(model))


def relation_id(
    kb_id: str, src_entity_id: str, relation_type: str, tgt_entity_id: str
) -> str:
    kb = str(kb_id or "").strip()
    src = str(src_entity_id or "").strip()
    tgt = str(tgt_entity_id or "").strip()
    kind = _norm(relation_type)
    if not kb or not src or not tgt or not kind:
        raise ValueError("relation_id 需要 kb_id、两端实体和关系类型")
    return "rel-" + _digest(kb, src, kind, tgt)


def source_id(chunk: str) -> str:
    text = str(chunk or "").strip()
    if not text:
        raise ValueError("source_id 必须是 chunk_id")
    return text


def object_key(user_id: str, kb_id: str, doc_id: str, file_name: str) -> str:
    user, kb = require_scope(user_id, kb_id)
    doc = str(doc_id or "").strip()
    name = str(file_name or "").replace("\\", "/").split("/")[-1]
    if not doc or not name or name in {".", ".."}:
        raise ValueError("对象路径需要 doc_id 和文件名")
    return f"{user}/{kb}/{doc}/{name}"


def chunk_record(
    user_id: str,
    kb_id: str,
    doc_id: str,
    index: int,
    content: str,
) -> dict[str, str]:
    user, kb = require_scope(user_id, kb_id)
    cid = chunk_id(doc_id, index)
    return {
        "user_id": user,
        "kb_id": kb,
        "doc_id": str(doc_id),
        "chunk_id": cid,
        "source_id": source_id(cid),
        "content": content,
    }


def entity_record(
    user_id: str,
    kb_id: str,
    entity_type: str,
    name: str,
    chunk: str,
    doc_id: str,
    model: str = "",
) -> dict[str, str]:
    user, kb = require_scope(user_id, kb_id)
    eid = entity_id(kb, entity_type, name, model)
    return {
        "user_id": user,
        "kb_id": kb,
        "doc_id": str(doc_id),
        "entity_id": eid,
        "entity_type": entity_type,
        "name": name,
        "model": model,
        "source_id": source_id(chunk),
    }


def relation_record(
    user_id: str,
    kb_id: str,
    src_entity_id: str,
    relation_type: str,
    tgt_entity_id: str,
    chunk: str,
    doc_id: str,
) -> dict[str, str]:
    user, kb = require_scope(user_id, kb_id)
    rid = relation_id(kb, src_entity_id, relation_type, tgt_entity_id)
    return {
        "user_id": user,
        "kb_id": kb,
        "doc_id": str(doc_id),
        "relation_id": rid,
        "src_entity_id": src_entity_id,
        "tgt_entity_id": tgt_entity_id,
        "relation_type": relation_type,
        "source_id": source_id(chunk),
    }


# 每个标识由谁保存、谁只拿来过滤或跳转。
STORE_OF = {
    "user_id": {
        "postgres": "scope",
        "milvus": "filter",
        "neo4j": "property",
        "object": "path",
    },
    "kb_id": {
        "postgres": "scope",
        "milvus": "filter",
        "neo4j": "property",
        "object": "path",
    },
    "doc_id": {
        "postgres": "documents",
        "milvus": "filter",
        "neo4j": "property",
        "object": "path",
    },
    "chunk_id": {
        "postgres": "chunks",
        "milvus": "chunks",
        "neo4j": "via source_id",
        "object": "",
    },
    "entity_id": {
        "postgres": "entities",
        "milvus": "entities",
        "neo4j": "node",
        "object": "",
    },
    "relation_id": {
        "postgres": "relations",
        "milvus": "relations",
        "neo4j": "relationship",
        "object": "",
    },
    "source_id": {
        "postgres": "sources",
        "milvus": "filter",
        "neo4j": "evidence",
        "object": "",
    },
}


def check_bundle(rows: list[dict[str, Any]]) -> None:
    """确认这些记录能靠同一套 ID 在三套存储之间跳转。"""
    chunks = {row["chunk_id"] for row in rows if row.get("chunk_id")}
    entities = {row["entity_id"] for row in rows if row.get("entity_id")}
    for row in rows:
        if row.get("chunk_id") and row.get("source_id") not in {None, row["chunk_id"]}:
            raise ValueError("文本块的 source_id 必须等于 chunk_id")
        if row.get("entity_id") and row.get("source_id") not in chunks:
            raise ValueError("实体的 source_id 必须指向已有 chunk_id")
        if row.get("relation_id"):
            if row.get("source_id") not in chunks:
                raise ValueError("关系的 source_id 必须指向已有 chunk_id")
            if (
                row.get("src_entity_id") not in entities
                or row.get("tgt_entity_id") not in entities
            ):
                raise ValueError("关系两端必须是已有 entity_id")
        for key in ("user_id", "kb_id", "doc_id"):
            if key in row and not str(row.get(key) or "").strip():
                raise ValueError(f"{key} 不能为空")
        name = str(row.get("name") or "")
        if name and name in {
            row.get("entity_id"),
            row.get("chunk_id"),
            row.get("doc_id"),
        }:
            raise ValueError("显示名称不能充当物理主键")
