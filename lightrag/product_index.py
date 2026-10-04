"""产品上传和 Worker 共用的入库。

向量只来自当前配置的 embedding 函数。
实体和关系只来自 LightRAG 的 extract_entities，并且名称必须出现在本文档正文里。
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.product_ids import check_bundle, entity_record, relation_record
from lightrag.product_updates import current_time

EmbedFunc = Callable[[list[str]], Awaitable[Any]]
LlmFunc = Callable[..., Awaitable[Any]]


@dataclass
class IndexRuntime:
    embed: EmbedFunc
    llm: LlmFunc
    gleaning: int = 1


def current_runtime() -> IndexRuntime:
    spec = os.getenv("INGEST_RUNTIME", "").strip()
    if spec:
        module_name, _, attr = spec.partition(":")
        module = importlib.import_module(module_name)
        value = getattr(module, attr or "runtime")
        runtime = value() if callable(value) else value
        if not isinstance(runtime, IndexRuntime):
            raise TypeError("INGEST_RUNTIME 必须返回 IndexRuntime")
        return runtime
    return IndexRuntime(
        embed=_configured_embed,
        llm=_configured_llm,
        gleaning=int(os.getenv("MAX_GLEANING", "1") or 1),
    )


async def embed_texts(texts: list[str]) -> np.ndarray:
    raw = await current_runtime().embed(list(texts))
    matrix = np.asarray(raw, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.shape[0] != len(texts) or matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("embedding 结果数量或维度与切块不一致")
    if _is_index_one_hot(matrix):
        raise ValueError("拒绝写入按序号生成的占位向量")
    return matrix


def write_chunk_vectors(
    working: Path,
    doc_id: str,
    kb_id: str,
    owner_id: str,
    chunks: list[dict[str, Any]],
    matrix: np.ndarray,
) -> None:
    from lightrag.product_storage import update_vector_store

    path = Path(working) / "vdb_chunks.json"
    dim = int(matrix.shape[1])
    _require_dim(path, dim)
    payload = []
    for item, vector in zip(chunks, matrix):
        row = {
            "__id__": item["chunk_id"],
            "__vector__": np.asarray(vector, dtype=np.float32),
            "chunk_id": item["chunk_id"],
            "source_id": item["source_id"],
            "doc_id": doc_id,
            "kb_id": kb_id,
            "user_id": owner_id,
            "owner_id": owner_id,
            "content": item["content"],
            "parent_id": item.get("parent_id") or "",
            "index_text": item.get("index_text") or "",
            "version_id": item.get("version_id") or "",
            "stable_chunk_id": item.get("stable_chunk_id") or "",
            "content_hash": item.get("content_hash") or "",
            "structural_anchor": item.get("structural_anchor") or "",
        }
        if item.get("evidence_ids"):
            row["evidence_ids"] = ",".join(str(part) for part in item["evidence_ids"])
        payload.append(row)

    def editor(db) -> None:
        if payload:
            _merge_vector_rows(db, payload)
            db.upsert(payload)

    update_vector_store(path, dim, editor)


async def extract_and_store(
    working: Path,
    doc_id: str,
    kb_id: str,
    owner_id: str,
    document_text: str,
    chunks: list[dict[str, Any]],
    file_path: str,
) -> None:
    if not chunks:
        return
    from lightrag.constants import DEFAULT_ENTITY_TYPES, DEFAULT_SUMMARY_LANGUAGE
    from lightrag.operate import extract_entities

    runtime = current_runtime()
    version_id = str(chunks[0].get("version_id") or "")
    chunk_map = {
        item["chunk_id"]: {
            "content": item["content"],
            "tokens": max(1, len(item["content"])),
            "full_doc_id": doc_id,
            "chunk_order_index": index,
            "file_path": file_path,
        }
        for index, item in enumerate(chunks)
    }
    results = await extract_entities(
        chunk_map,
        global_config={
            "llm_model_func": runtime.llm,
            "entity_extract_max_gleaning": max(0, int(runtime.gleaning)),
            "addon_params": {
                "language": DEFAULT_SUMMARY_LANGUAGE,
                "entity_types": list(DEFAULT_ENTITY_TYPES),
            },
            "llm_model_max_async": 2,
        },
    )
    known = {item["chunk_id"] for item in chunks}
    entities: list[dict[str, str]] = []
    relations: list[dict[str, str]] = []
    seen_entities: set[str] = set()
    for maybe_nodes, maybe_edges in results or []:
        for name, rows in (maybe_nodes or {}).items():
            row = rows[0] if rows else {}
            entity_name = str(row.get("entity_name") or name or "").strip()
            source = str(row.get("source_id") or "")
            if source not in known or not _name_in_text(
                entity_name, document_text, doc_id
            ):
                continue
            record = entity_record(
                owner_id,
                kb_id,
                str(row.get("entity_type") or "部件"),
                entity_name,
                source,
                doc_id,
            )
            record["source_version_id"] = version_id
            record["valid_at"] = current_time()
            record["invalid_at"] = ""
            if record["entity_id"] in seen_entities:
                continue
            seen_entities.add(record["entity_id"])
            entities.append(record)
        by_name = {item["name"]: item for item in entities}
        for edge_key, rows in (maybe_edges or {}).items():
            row = rows[0] if rows else {}
            src_name = str(
                row.get("src_id") or (edge_key[0] if edge_key else "")
            ).strip()
            tgt_name = str(
                row.get("tgt_id") or (edge_key[1] if edge_key else "")
            ).strip()
            keyword = str(row.get("keywords") or row.get("relation_type") or "").strip()
            description = str(row.get("description") or "").strip()
            source = str(row.get("source_id") or "")
            if source not in known:
                continue
            if not _relation_grounded(
                src_name, tgt_name, keyword, description, document_text, doc_id
            ):
                continue
            left = by_name.get(src_name)
            right = by_name.get(tgt_name)
            if left is None or right is None or left["entity_id"] == right["entity_id"]:
                continue
            relations.append(
                relation_record(
                    owner_id,
                    kb_id,
                    left["entity_id"],
                    keyword or description,
                    right["entity_id"],
                    source,
                    doc_id,
                )
            )
            relations[-1]["source_version_id"] = version_id
            relations[-1]["valid_at"] = current_time()
            relations[-1]["invalid_at"] = ""
    if not entities:
        return
    check_bundle([*chunks, *entities, *relations])
    _write_graph(Path(working), entities, relations)
    try:
        from lightrag.product_appdb import open_appdb

        appdb = open_appdb(Path(working))
        for relation in relations:
            appdb.activate_knowledge_fact(
                {
                    "fact_id": relation["relation_id"],
                    "owner_id": owner_id,
                    "kb_id": kb_id,
                    "subject_id": relation["src_entity_id"],
                    "predicate": relation["relation_type"],
                    "object_id": relation["tgt_entity_id"],
                    "source_version_id": relation.get("source_version_id") or version_id,
                    "source_chunk_id": relation.get("source_id") or "",
                    "valid_at": relation.get("valid_at") or "",
                }
            )
    except Exception:
        # The graph/vector transaction remains authoritative for legacy stores;
        # app database facts are an additional audit index.
        pass
    await _write_graph_vectors(
        Path(working), doc_id, kb_id, owner_id, entities, relations
    )


def _name_in_text(name: str, document_text: str, doc_id: str) -> bool:
    if not name or name in {f"{doc_id}-a", f"{doc_id}-b"}:
        return False
    if doc_id and (name == f"{doc_id}-a" or name == f"{doc_id}-b"):
        return False
    return name in document_text


def _relation_grounded(
    src: str, tgt: str, keyword: str, description: str, document_text: str, doc_id: str
) -> bool:
    if not _name_in_text(src, document_text, doc_id) or not _name_in_text(
        tgt, document_text, doc_id
    ):
        return False
    if "导致" in keyword and "导致" not in document_text:
        return False
    if keyword and keyword in document_text:
        return True
    return len(description) >= 2 and description in document_text


def _merge_values(existing: Any, incoming: Any) -> str:
    from lightrag.constants import GRAPH_FIELD_SEP
    from lightrag.product_scope import split_values

    merged: list[str] = []
    for value in (existing, incoming):
        for item in split_values(value):
            if item not in merged:
                merged.append(item)
    return GRAPH_FIELD_SEP.join(merged)


def _merge_evidence(
    existing_source: Any, existing_doc: Any, source: str, doc_id: str
) -> tuple[str, str]:
    from lightrag.product_scope import split_values

    sources = split_values(existing_source)
    docs = split_values(existing_doc)
    if len(docs) != len(sources):
        docs = [""] * len(sources)
    for part in split_values(source):
        if part in sources:
            continue
        sources.append(part)
        docs.append(doc_id)
    from lightrag.constants import GRAPH_FIELD_SEP

    return GRAPH_FIELD_SEP.join(sources), GRAPH_FIELD_SEP.join(docs)


def _merge_vector_rows(db, payload: list[dict[str, Any]]) -> None:
    storage = getattr(db, "_NanoVectorDB__storage")
    existing = {str(row.get("__id__")): row for row in storage.get("data") or []}
    for row in payload:
        old = existing.get(str(row.get("__id__")))
        if not old:
            continue
        source, doc = _merge_evidence(
            old.get("source_id"),
            old.get("doc_id"),
            str(row.get("source_id") or ""),
            str(row.get("doc_id") or ""),
        )
        row["source_id"] = source
        row["doc_id"] = doc
        row["kb_id"] = _merge_values(old.get("kb_id"), row.get("kb_id"))


def _write_graph(
    working: Path, entities: list[dict[str, str]], relations: list[dict[str, str]]
) -> None:
    from lightrag.product_storage import rewrite_graph

    path = working / "graph_chunk_entity_relation.graphml"

    def editor(graph) -> None:
        for item in entities:
            node_id = item["entity_id"]
            current = dict(graph.nodes[node_id]) if graph.has_node(node_id) else {}
            source, doc = _merge_evidence(
                current.get("source_id"),
                current.get("doc_id"),
                item["source_id"],
                item["doc_id"],
            )
            merged = dict(item)
            merged["source_id"] = source
            merged["doc_id"] = doc
            merged["kb_id"] = _merge_values(current.get("kb_id"), item.get("kb_id"))
            for key in ("source_version_id", "valid_at", "invalid_at", "superseded_by"):
                if item.get(key) is not None:
                    merged[key] = item.get(key) or current.get(key) or ""
            if current.get("file_path") or item.get("file_path"):
                merged["file_path"] = _merge_values(
                    current.get("file_path"), item.get("file_path")
                )
            graph.add_node(node_id, **merged)
        for item in relations:
            src = item["src_entity_id"]
            tgt = item["tgt_entity_id"]
            # Preserve conflicting historical edges, but mark them inactive so
            # current retrieval can ignore them without losing auditability.
            for old_src, old_tgt, old_attrs in list(graph.edges(data=True)):
                if old_src != src or old_tgt == tgt:
                    continue
                old_type = str(old_attrs.get("relation_type") or old_attrs.get("keywords") or "")
                if old_type.split(GRAPH_FIELD_SEP)[0] != str(item.get("relation_type") or ""):
                    continue
                old_attrs["invalid_at"] = current_time()
                old_attrs["status"] = "superseded"
                old_attrs["superseded_by"] = item.get("relation_id") or ""
                graph.edges[old_src, old_tgt].update(old_attrs)
            current = dict(graph.edges[src, tgt]) if graph.has_edge(src, tgt) else {}
            source, doc = _merge_evidence(
                current.get("source_id"),
                current.get("doc_id"),
                item["source_id"],
                item["doc_id"],
            )
            merged = dict(item)
            merged["source_id"] = source
            merged["doc_id"] = doc
            merged["kb_id"] = _merge_values(current.get("kb_id"), item.get("kb_id"))
            for key in ("source_version_id", "valid_at", "invalid_at", "superseded_by"):
                if item.get(key) is not None:
                    merged[key] = item.get(key) or current.get(key) or ""
            graph.add_edge(src, tgt, **merged)

    rewrite_graph(path, editor)


async def _write_graph_vectors(
    working, doc_id, kb_id, owner_id, entities, relations
) -> None:
    from lightrag.product_storage import update_vector_store

    async def store(path: Path, rows: list[tuple[str, str, dict[str, str]]]) -> None:
        if not rows:
            return
        matrix = await embed_texts([text for _id, text, _extra in rows])
        dim = int(matrix.shape[1])
        _require_dim(path, dim)
        payload = []
        for (row_id, _text, extra), vector in zip(rows, matrix):
            payload.append(
                {
                    "__id__": row_id,
                    "__vector__": np.asarray(vector, dtype=np.float32),
                    "doc_id": doc_id,
                    "kb_id": kb_id,
                    "user_id": owner_id,
                    "owner_id": owner_id,
                    **extra,
                }
            )

        def editor(db) -> None:
            _merge_vector_rows(db, payload)
            db.upsert(payload)

        update_vector_store(path, dim, editor)

    await store(
        working / "vdb_entities.json",
        [
            (
                item["entity_id"],
                item["name"],
                {
                    "entity_id": item["entity_id"],
                    "entity_name": item["name"],
                    "source_id": item["source_id"],
                    "content": item["name"],
                    "source_version_id": item.get("source_version_id") or "",
                    "valid_at": item.get("valid_at") or "",
                    "invalid_at": item.get("invalid_at") or "",
                },
            )
            for item in entities
        ],
    )
    await store(
        working / "vdb_relationships.json",
        [
            (
                item["relation_id"],
                item["relation_type"],
                {
                    "relation_id": item["relation_id"],
                    "src_id": item["src_entity_id"],
                    "tgt_id": item["tgt_entity_id"],
                    "source_id": item["source_id"],
                    "content": item["relation_type"],
                    "source_version_id": item.get("source_version_id") or "",
                    "valid_at": item.get("valid_at") or "",
                    "invalid_at": item.get("invalid_at") or "",
                },
            )
            for item in relations
        ],
    )


def _require_dim(path: Path, dim: int) -> None:
    if not path.exists():
        return
    from lightrag.product_storage import _read_json

    payload = _read_json(path) or {}
    existing = int(payload.get("embedding_dim") or 0)
    if existing and existing != dim:
        raise ValueError(f"向量维度 {dim} 与已有存储 {existing} 不一致")


def _is_index_one_hot(matrix: np.ndarray) -> bool:
    if matrix.size == 0:
        return False
    for index, row in enumerate(matrix):
        hot = int(index % row.shape[0])
        for position, value in enumerate(row):
            expected = 1.0 if position == hot else 0.0
            if abs(float(value) - expected) > 1e-5:
                return False
    return True


async def _configured_embed(texts: list[str]):
    binding = os.getenv("EMBEDDING_BINDING", "ollama").strip() or "ollama"
    model = os.getenv("EMBEDDING_MODEL", "").strip()
    host = os.getenv("EMBEDDING_BINDING_HOST", "").strip()
    api_key = os.getenv("EMBEDDING_BINDING_API_KEY", "").strip()
    if binding == "ollama":
        from lightrag.llm.ollama import ollama_embed

        return await ollama_embed(
            texts,
            embed_model=model or "bge-m3",
            host=host or "http://localhost:11434",
            api_key=api_key or None,
        )
    from lightrag.llm.openai import openai_embed

    try:
        configured_dimension = int(os.getenv("EMBEDDING_DIM", "0") or 0)
    except ValueError:
        configured_dimension = 0
    selected_model = model or "text-embedding-3-small"
    dimensions = (
        configured_dimension
        if configured_dimension > 0 and "qwen" in selected_model.lower()
        else None
    )
    return await openai_embed(
        texts,
        model=selected_model,
        base_url=host or None,
        api_key=api_key or None,
        dimensions=dimensions,
    )


async def _configured_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    history = history_messages or []
    binding = os.getenv("LLM_BINDING", "ollama").strip() or "ollama"
    model = os.getenv("LLM_MODEL", "").strip()
    host = os.getenv("LLM_BINDING_HOST", "").strip()
    api_key = os.getenv("LLM_BINDING_API_KEY", "").strip()
    if binding == "ollama":
        from lightrag.llm.ollama import _ollama_model_if_cache

        return await _ollama_model_if_cache(
            model or "qwen2.5",
            prompt,
            system_prompt=system_prompt,
            history_messages=history,
            host=host or "http://localhost:11434",
            api_key=api_key or None,
        )
    from lightrag.llm.openai import openai_complete_if_cache

    return await openai_complete_if_cache(
        model or "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history,
        base_url=host or None,
        api_key=api_key or None,
    )
