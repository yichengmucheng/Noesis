"""读取并清理 LightRAG 落在磁盘上的 JSON、向量和图。

检查器会解析文件内容。删除按 doc_id 和 chunk_id 进行。
实体或关系还有其他文档的 source_id 时，只摘掉被删来源，不按名称整条删除。
"""

from __future__ import annotations

import json
import threading
import weakref
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.product_scope import split_values

CHECKER_VERSION = "deep-check-1"
_STORE_FILES = {
    "doc_status": "kv_store_doc_status.json",
    "full_docs": "kv_store_full_docs.json",
    "text_chunks": "kv_store_text_chunks.json",
    "parent_chunks": "kv_store_parent_chunks.json",
    "full_entities": "kv_store_full_entities.json",
    "full_relations": "kv_store_full_relations.json",
    "chunks_vdb": "vdb_chunks.json",
    "entities_vdb": "vdb_entities.json",
    "relationships_vdb": "vdb_relationships.json",
}
_GRAPH_FILE = "graph_chunk_entity_relation.graphml"
_FAIL_STEP = {"name": ""}
_COMMIT = threading.local()


class CommitGate:
    """超时后关闭。已经在写的线程必须在替换文件前再看一次。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.closed = False

    def close(self) -> None:
        with self._lock:
            self.closed = True

    def allow(self) -> bool:
        with self._lock:
            return not self.closed


def bind_commit_gate(gate: CommitGate | None) -> None:
    _COMMIT.gate = gate


def commit_allowed() -> bool:
    gate = getattr(_COMMIT, "gate", None)
    if gate is None:
        return True
    return gate.allow()


class ProcessCrash(RuntimeError):
    """模拟进程在删除中途退出。任务保持 running，下次可以继续。"""


def arm_failure(step: str) -> None:
    _FAIL_STEP["name"] = step


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _file_lock(path: Path):
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+b")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield handle
    finally:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()


def _storage_lock(path: Path):
    return _file_lock(path.with_name(path.name + ".lock"))


def update_vector_store(path: Path, dim: int, editor: Callable[[Any], None]) -> bool:
    """锁住整次读-改-写，先写临时文件再替换。超时关闭后不再落盘。"""
    from nano_vectordb import NanoVectorDB

    path = Path(path)
    with _storage_lock(path):
        if not commit_allowed():
            return False
        db = NanoVectorDB(embedding_dim=int(dim), storage_file=str(path))
        editor(db)
        if not commit_allowed():
            return False
        tmp = path.with_name(path.name + ".tmp")
        db.storage_file = str(tmp)
        db.save()
        if not commit_allowed():
            if tmp.exists():
                tmp.unlink()
            return False
        tmp.replace(path)
        return True


def rewrite_graph(path: Path, editor: Callable[[Any], None]) -> bool:
    """锁住 GraphML 的整次读-改-写，并用临时文件替换。"""
    import networkx as nx

    path = Path(path)
    with _storage_lock(path):
        if not commit_allowed():
            return False
        graph = nx.read_graphml(path) if path.exists() else nx.Graph()
        editor(graph)
        if not commit_allowed():
            return False
        if graph.number_of_nodes() == 0 and graph.number_of_edges() == 0:
            if path.exists():
                path.unlink()
            return True
        tmp = path.with_name(path.name + ".tmp")
        nx.write_graphml(graph, tmp)
        if not commit_allowed():
            if tmp.exists():
                tmp.unlink()
            return False
        tmp.replace(path)
        return True


_shell_snapshots: dict[str, dict[str, Any]] = {}
_snapshot_guard = threading.Lock()
_MISSING = object()


class ShellData(dict):
    """外壳副本。用对象 id 找回读取时的基线，保存后立刻丢掉。"""


_shell_bases: dict[int, dict[str, Any]] = {}


def _shell_defaults(data: dict[str, Any]) -> dict[str, Any]:
    data.setdefault("kbs", [])
    data.setdefault("file_bindings", {})
    data.setdefault("doc_index", {})
    data.setdefault("qa_pairs", [])
    data.setdefault("comparisons", [])
    data.setdefault("audits", {})
    data.setdefault("sessions", {})
    data.setdefault("purge_jobs", [])
    data.setdefault("upload_jobs", {})
    return data


def _apply_diff(base: dict[str, Any], new: dict[str, Any], current: dict[str, Any]) -> None:
    """把 base 到 new 的差异写进 current。未改动的字段保留 current 里的并发更新。"""
    base = base if isinstance(base, dict) else {}
    new = new if isinstance(new, dict) else {}
    for key in list(base):
        if key not in new and key in current and current.get(key) == base.get(key):
            current.pop(key, None)
    for key, value in new.items():
        old = base.get(key, _MISSING)
        if old is _MISSING:
            current[key] = json.loads(json.dumps(value))
            continue
        if value == old:
            continue
        if isinstance(value, dict) and isinstance(old, dict) and isinstance(current.get(key), dict):
            _apply_diff(old, value, current[key])
        else:
            current[key] = json.loads(json.dumps(value))


def _remember_shell(path: Path, payload: dict[str, Any]) -> None:
    with _snapshot_guard:
        _shell_snapshots[str(path)] = json.loads(json.dumps(payload))


def _recall_shell(path: Path) -> dict[str, Any] | None:
    with _snapshot_guard:
        cached = _shell_snapshots.get(str(path))
    return None if cached is None else json.loads(json.dumps(cached))


def _parse_complete(candidate: Path) -> Any:
    text = candidate.read_text(encoding="utf-8")
    if not text.strip():
        raise json.JSONDecodeError("empty", text, 0)
    return json.loads(text)


def _read_json(path: Path) -> Any:
    for candidate in (path, path.with_name(path.name + ".bak")):
        if not candidate.exists():
            continue
        try:
            return _parse_complete(candidate)
        except (json.JSONDecodeError, PermissionError, OSError):
            continue
    return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, ensure_ascii=False, indent=2)
    tmp = path.with_name(path.name + ".tmp")
    bak = path.with_name(path.name + ".bak")
    lock = path.with_name(path.name + ".lock")
    last_error: OSError | None = None
    for _attempt in range(8):
        try:
            with _file_lock(lock):
                tmp.write_text(raw, encoding="utf-8")
                if path.exists():
                    try:
                        bak.write_bytes(path.read_bytes())
                    except PermissionError:
                        pass
                tmp.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.02)
    if last_error is not None:
        raise last_error


def _read_shell_locked(path: Path) -> Any:
    lock = path.with_name(path.name + ".lock")
    last_error: OSError | None = None
    for _attempt in range(8):
        try:
            with _file_lock(lock):
                for candidate in (path, path.with_name(path.name + ".bak")):
                    if not candidate.exists():
                        continue
                    try:
                        return _parse_complete(candidate)
                    except json.JSONDecodeError:
                        continue
            return None
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.02)
    cached = _recall_shell(path)
    if cached is not None:
        return cached
    if last_error is not None:
        raise last_error
    return None


def _shell_path(working_dir: Path) -> Path:
    return working_dir / "product_shell.json"


def _cache_path(working_dir: Path) -> Path:
    return working_dir / "semantic_cache.json"


def load_shell(working_dir: Path) -> dict[str, Any]:
    path = _shell_path(working_dir)
    try:
        payload = _read_shell_locked(path)
    except PermissionError:
        payload = _recall_shell(path)
    data = ShellData(payload if isinstance(payload, dict) else {})
    _shell_defaults(data)
    base = json.loads(json.dumps(data))
    _shell_bases[id(data)] = base
    weakref.finalize(data, _shell_bases.pop, id(data), None)
    if payload:
        _remember_shell(path, data)
    return data


def mutate_shell(working_dir: Path, editor: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """同一把跨进程锁里完成读取、修改、写临时文件和替换。"""
    path = _shell_path(working_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    last_error: OSError | None = None
    for _attempt in range(8):
        try:
            with _file_lock(lock):
                payload = None
                for candidate in (path, path.with_name(path.name + ".bak")):
                    if not candidate.exists():
                        continue
                    try:
                        payload = _parse_complete(candidate)
                        break
                    except json.JSONDecodeError:
                        continue
                data = _shell_defaults(payload if isinstance(payload, dict) else {})
                editor(data)
                raw = json.dumps(data, ensure_ascii=False, indent=2)
                tmp = path.with_name(path.name + ".tmp")
                bak = path.with_name(path.name + ".bak")
                tmp.write_text(raw, encoding="utf-8")
                if path.exists():
                    try:
                        bak.write_bytes(path.read_bytes())
                    except PermissionError:
                        pass
                tmp.replace(path)
                stored = json.loads(raw)
                _remember_shell(path, stored)
                return stored
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.02)
    if last_error is not None:
        raise last_error
    raise PermissionError(str(path))


def save_shell(working_dir: Path, data: dict[str, Any]) -> None:
    public = json.loads(json.dumps(data))
    base = _shell_bases.pop(id(data), None)

    def editor(current: dict[str, Any]) -> None:
        if not isinstance(base, dict):
            current.clear()
            current.update(public)
            _shell_defaults(current)
            return
        _apply_diff(base, public, current)

    mutate_shell(working_dir, editor)


def _load_cache(working_dir: Path) -> list[dict[str, Any]]:
    payload = _read_json(_cache_path(working_dir))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _save_cache(working_dir: Path, entries: list[dict[str, Any]]) -> None:
    _write_json(_cache_path(working_dir), entries)


def _inside(root: Path, path: Path) -> bool:
    root = root.resolve()
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return resolved == root or root in resolved.parents


def _kv(working_dir: Path, name: str) -> dict[str, Any]:
    payload = _read_json(working_dir / _STORE_FILES[name])
    if not isinstance(payload, dict):
        return {}
    return {str(key): value for key, value in payload.items() if isinstance(value, dict)}


def _save_kv(working_dir: Path, name: str, rows: dict[str, Any]) -> None:
    _write_json(working_dir / _STORE_FILES[name], rows)


def _record_id(key: str, row: dict[str, Any]) -> str:
    return str(row.get("__id__") or row.get("_id") or row.get("id") or key)


def _hits(row: dict[str, Any], key: str, kb_id: str, doc_ids: set[str], chunk_ids: set[str]) -> bool:
    if kb_id and kb_id in split_values(row.get("kb_id")):
        return True
    doc_id = str(row.get("doc_id") or row.get("full_doc_id") or "")
    if doc_id and doc_id in doc_ids:
        return True
    identity = _record_id(key, row)
    if identity in doc_ids or identity in chunk_ids:
        return True
    if set(split_values(row.get("source_id"))) & chunk_ids:
        return True
    return False


def _orphan(row: dict[str, Any]) -> bool:
    if split_values(row.get("kb_id")):
        return False
    if str(row.get("doc_id") or row.get("full_doc_id") or "").strip():
        return False
    if split_values(row.get("source_id")):
        return False
    if str(row.get("file_path") or "").strip():
        return False
    return True


def _known_ids(data: dict[str, Any], kb_id: str) -> tuple[set[str], set[str]]:
    docs: set[str] = set()
    chunks: set[str] = set()
    for row in (data.get("doc_index") or {}).values():
        if not isinstance(row, dict) or row.get("kb_id") != kb_id:
            continue
        if row.get("doc_id"):
            docs.add(str(row["doc_id"]))
        chunks.update(str(item) for item in row.get("chunk_ids") or [] if item)
    for job in data.get("purge_jobs") or []:
        if job.get("kb_id") != kb_id:
            continue
        docs.update(str(item) for item in job.get("checked_doc_ids") or [] if item)
        chunks.update(str(item) for item in job.get("checked_chunk_ids") or [] if item)
    return docs, chunks


def _samples(ids: list[str]) -> list[str]:
    return ids[:5]


def _graph(working_dir: Path):
    import networkx as nx

    path = working_dir / _GRAPH_FILE
    if not path.exists():
        return nx.Graph(), path
    return nx.read_graphml(path), path


def _vdb_rows(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return []
    rows = payload.get("data") or []
    return [row for row in rows if isinstance(row, dict)]


def deep_check(
    working_dir: Path,
    kb_id: str,
    input_dir: Path | None = None,
    *,
    doc_ids: list[str] | None = None,
    chunk_ids: list[str] | None = None,
) -> dict[str, Any]:
    working_dir = Path(working_dir)
    data = load_shell(working_dir)
    known_docs, known_chunks = _known_ids(data, kb_id)
    if doc_ids:
        known_docs.update(doc_ids)
    if chunk_ids:
        known_chunks.update(chunk_ids)
    stores: dict[str, Any] = {}
    orphan_count = 0

    def add_store(name: str, leftovers: list[str], rows: list[dict[str, Any]]) -> None:
        nonlocal orphan_count
        orphan_count += sum(1 for row in rows if _orphan(row))
        stores[name] = {"count": len(leftovers), "samples": _samples(leftovers)}

    for name in ("doc_status", "full_docs", "text_chunks", "full_entities", "full_relations"):
        rows = _kv(working_dir, name)
        leftovers = [
            _record_id(key, row)
            for key, row in rows.items()
            if _hits(row, key, kb_id, known_docs, known_chunks)
        ]
        add_store(name, leftovers, list(rows.values()))

    for name in ("chunks_vdb", "entities_vdb", "relationships_vdb"):
        rows = _vdb_rows(working_dir / _STORE_FILES[name])
        leftovers = [
            _record_id(str(row.get("__id__") or ""), row)
            for row in rows
            if _hits(row, str(row.get("__id__") or ""), kb_id, known_docs, known_chunks)
        ]
        add_store(name, leftovers, rows)

    graph, _path = _graph(working_dir)
    node_rows = []
    node_left = []
    for node_id, attrs in graph.nodes(data=True):
        row = dict(attrs)
        row["id"] = str(node_id)
        node_rows.append(row)
        if _hits(row, str(node_id), kb_id, known_docs, known_chunks):
            node_left.append(str(node_id))
    add_store("graph_nodes", node_left, node_rows)
    edge_rows = []
    edge_left = []
    for source, target, attrs in graph.edges(data=True):
        row = dict(attrs)
        row["id"] = f"{source}->{target}"
        edge_rows.append(row)
        if _hits(row, row["id"], kb_id, known_docs, known_chunks):
            edge_left.append(row["id"])
    add_store("graph_edges", edge_left, edge_rows)

    cache = _load_cache(working_dir)
    cache_left = []
    for index, entry in enumerate(cache):
        chunk_hit = set(str(item) for item in entry.get("chunk_ids") or []) & known_chunks
        if entry.get("kb_id") == kb_id or chunk_hit:
            cache_left.append(str(entry.get("id") or index))
    add_store("semantic_cache", cache_left, cache)

    qa_left = [str(item.get("id") or "") for item in data.get("qa_pairs") or [] if item.get("kb_id") == kb_id]
    add_store("qa", qa_left, [])
    comparison_left = [
        str(item.get("task_id") or item.get("id") or "")
        for item in data.get("comparisons") or []
        if item.get("kb_id") == kb_id
    ]
    add_store("comparison", comparison_left, [])
    audit_left = [
        key for key, value in (data.get("audits") or {}).items()
        if isinstance(value, dict) and value.get("kb_id") == kb_id
    ]
    add_store("audit", audit_left, [])
    session_left = [
        key for key, value in (data.get("sessions") or {}).items()
        if isinstance(value, dict) and value.get("kb_id") == kb_id
    ]
    add_store("sessions", session_left, [])
    binding_left = [
        key for key, value in (data.get("file_bindings") or {}).items() if value == kb_id
    ]
    add_store("file_bindings", binding_left, [])

    source_left: list[str] = []
    root = Path(input_dir) if input_dir else None
    if root and root.exists():
        for row in (data.get("doc_index") or {}).values():
            if not isinstance(row, dict) or row.get("kb_id") != kb_id:
                continue
            key = str(row.get("storage_key") or "")
            if not key:
                continue
            path = root / key
            if path.is_file() and _inside(root, path):
                source_left.append(key)
    add_store("source_files", source_left, [])

    temp_left: list[str] = []
    known_names = set(_STORE_FILES.values()) | {_GRAPH_FILE, "product_shell.json", "semantic_cache.json"}
    if known_docs or known_chunks:
        for path in working_dir.rglob("*"):
            if not path.is_file() or path.name in known_names or path.name.endswith(".tmp"):
                continue
            if "document_ir" in path.parts:
                continue
            text = path.name
            if any(doc_id in text for doc_id in known_docs) or any(chunk_id in text for chunk_id in known_chunks):
                temp_left.append(path.name)
    add_store("temp_files", temp_left, [])
    ir_left: list[str] = []
    ir_root = working_dir / "document_ir"
    if ir_root.is_dir() and known_docs:
        for path in ir_root.rglob("*"):
            if not path.is_file():
                continue
            payload = _read_json(path) if path.suffix == ".json" or path.name.endswith(".json.bak") else None
            document_id = str(payload.get("document_id") or "") if isinstance(payload, dict) else ""
            if document_id in known_docs or any(doc_id in path.name for doc_id in known_docs):
                ir_left.append(str(path.relative_to(working_dir)))
    add_store("document_ir", ir_left, [])

    passed = all(item["count"] == 0 for item in stores.values())
    return {
        "checker_version": CHECKER_VERSION,
        "checked_at": _now(),
        "checked_kb_id": kb_id,
        "checked_doc_ids": sorted(known_docs),
        "checked_chunk_ids": sorted(known_chunks),
        "stores": stores,
        "orphan_count": orphan_count,
        "passed": passed,
    }


def document_consistency(
    working_dir: Path,
    kb_id: str,
    doc_id: str,
    chunk_ids: list[str],
    *,
    expect: str = "present",
) -> dict[str, Any]:
    """检查单个文档的切块、向量和图边是否恰好一份。

    expect=present 用于上传完成：期望的编号各出现一次。
    expect=absent 用于删除和失败清理：这些编号必须为 0。
    """
    working_dir = Path(working_dir)
    expected = [str(item) for item in chunk_ids if item]
    chunks = _kv(working_dir, "text_chunks")
    owned_chunks: list[str] = []
    for key, row in chunks.items():
        identity = _record_id(key, row)
        owner = str(row.get("doc_id") or row.get("full_doc_id") or "")
        if owner == doc_id or identity in expected:
            owned_chunks.append(identity)
    chunk_hits = {item: owned_chunks.count(item) for item in expected}
    for identity in owned_chunks:
        chunk_hits.setdefault(identity, owned_chunks.count(identity))
    vectors = _vdb_rows(working_dir / _STORE_FILES["chunks_vdb"])
    owned_vectors: list[str] = []
    for row in vectors:
        identity = _record_id(str(row.get("__id__") or ""), row)
        owner = str(row.get("doc_id") or "")
        if owner == doc_id or identity in expected:
            owned_vectors.append(identity)
    vector_hits = {item: owned_vectors.count(item) for item in expected}
    for identity in owned_vectors:
        vector_hits.setdefault(identity, owned_vectors.count(identity))
    graph, _path = _graph(working_dir)
    edge_ids: list[str] = []
    for source, target, attrs in graph.edges(data=True):
        sources = split_values(attrs.get("source_id"))
        owners = split_values(attrs.get("doc_id"))
        if doc_id and (doc_id in owners or doc_id in sources):
            edge_ids.append(f"{source}->{target}")
        elif set(sources) & set(expected):
            edge_ids.append(f"{source}->{target}")
    duplicate_edges = len(edge_ids) != len(set(edge_ids))
    docs = _kv(working_dir, "full_docs")
    doc_count = sum(1 for key, row in docs.items() if _record_id(key, row) == doc_id or str(row.get("doc_id") or "") == doc_id)
    ir_dir = working_dir / "document_ir"
    ir_exists = any((ir_dir / name).exists() for name in (f"{doc_id}.json", f"{doc_id}.json.bak", f"{doc_id}.json.tmp"))
    if expect == "absent":
        passed = doc_count == 0 and not owned_chunks and not owned_vectors and not edge_ids and not ir_exists
    else:
        passed = (
            bool(expected)
            and doc_count == 1
            and sorted(owned_chunks) == sorted(expected)
            and sorted(owned_vectors) == sorted(expected)
            and all(chunk_hits.get(item) == 1 for item in expected)
            and all(vector_hits.get(item) == 1 for item in expected)
            and not duplicate_edges
        )
    return {
        "checker_version": CHECKER_VERSION,
        "checked_at": _now(),
        "checked_kb_id": kb_id,
        "checked_doc_ids": [doc_id],
        "checked_chunk_ids": expected,
        "expect": expect,
        "doc_count": doc_count,
        "chunk_hits": chunk_hits,
        "vector_hits": vector_hits,
        "edge_ids": edge_ids,
        "duplicate_edges": duplicate_edges,
        "document_ir": ir_exists,
        "orphan_count": 0,
        "passed": passed,
    }


def _aligned(parts: list[str], original_sources: list[str], kept: set[str]) -> list[str]:
    if len(parts) == len(original_sources) and parts:
        return [part for part, source in zip(parts, original_sources) if source in kept]
    return parts


def _rewrite_graph_attrs(attrs: dict[str, Any], removed_chunks: set[str], chunk_kb: dict[str, set[str]], kb_id: str) -> str:
    sources = split_values(attrs.get("source_id"))
    kept = [source for source in sources if source not in removed_chunks]
    if not sources:
        if kb_id not in split_values(attrs.get("kb_id")):
            return "keep"
        rest = [item for item in split_values(attrs.get("kb_id")) if item != kb_id]
        if not rest:
            return "drop"
        attrs["kb_id"] = GRAPH_FIELD_SEP.join(rest)
        return "keep"
    if not (set(sources) & removed_chunks) and kb_id not in split_values(attrs.get("kb_id")):
        return "keep"
    if not kept:
        return "drop"
    attrs["source_id"] = GRAPH_FIELD_SEP.join(kept)
    docs = split_values(attrs.get("doc_id"))
    if len(docs) == len(sources):
        attrs["doc_id"] = GRAPH_FIELD_SEP.join(_aligned(docs, sources, set(kept)))
    attrs["description"] = GRAPH_FIELD_SEP.join(_aligned(split_values(attrs.get("description")), sources, set(kept)))
    attrs["file_path"] = GRAPH_FIELD_SEP.join(_aligned(split_values(attrs.get("file_path")), sources, set(kept)))
    kb_ids: list[str] = []
    for source in kept:
        for item in chunk_kb.get(source, set()):
            if item not in kb_ids:
                kb_ids.append(item)
    if not kb_ids:
        kb_ids = [item for item in split_values(attrs.get("kb_id")) if item != kb_id]
    attrs["kb_id"] = GRAPH_FIELD_SEP.join(kb_ids)
    return "keep"


def _open_vdb(path: Path):
    payload = _read_json(path)
    if not isinstance(payload, dict) or not payload.get("embedding_dim"):
        return None
    from nano_vectordb import NanoVectorDB

    return NanoVectorDB(embedding_dim=int(payload["embedding_dim"]), storage_file=str(path))


def _edit_vdb(path: Path, editor: Callable[[dict[str, Any]], str]) -> None:
    if not path.exists():
        return
    payload = _read_json(path) or {}
    dim = int(payload.get("embedding_dim") or 0)
    if dim <= 0:
        return

    def mutate(db) -> None:
        storage = getattr(db, "_NanoVectorDB__storage")
        remove: list[str] = []
        for row in list(storage["data"]):
            action = editor(row)
            if action == "drop":
                remove.append(str(row.get("__id__")))
        if remove:
            db.delete(remove)

    update_vector_store(path, dim, mutate)


def _maybe_fail(step: str) -> None:
    if _FAIL_STEP["name"] == step:
        _FAIL_STEP["name"] = ""
        raise RuntimeError("存储删除失败")


def _collect_chunks(working_dir: Path, data: dict[str, Any], kb_id: str, docs: set[str]) -> tuple[set[str], dict[str, set[str]]]:
    removed: set[str] = set()
    chunk_kb: dict[str, set[str]] = {}
    for key, row in _kv(working_dir, "text_chunks").items():
        chunk_id = _record_id(key, row)
        owners = set(split_values(row.get("kb_id")))
        if owners:
            chunk_kb.setdefault(chunk_id, set()).update(owners)
        doc_id = str(row.get("doc_id") or row.get("full_doc_id") or "")
        if doc_id in docs:
            removed.add(chunk_id)
    for row in (data.get("doc_index") or {}).values():
        if isinstance(row, dict) and str(row.get("doc_id") or "") in docs:
            removed.update(str(item) for item in row.get("chunk_ids") or [] if item)
    return removed, chunk_kb


def remove_documents(
    working_dir: Path,
    input_dir: Path,
    data: dict[str, Any],
    kb_id: str,
    doc_ids: list[str],
    *,
    crash_after: str | None = None,
    purge_kb_metadata: bool = False,
    known_chunks: set[str] | None = None,
    preserve_index: bool = False,
) -> dict[str, Any]:
    working_dir = Path(working_dir)
    input_dir = Path(input_dir)
    docs = {str(item) for item in doc_ids if item}
    removed_chunks, chunk_kb = _collect_chunks(working_dir, data, kb_id, docs)
    if known_chunks:
        removed_chunks.update(known_chunks)

    for name in ("doc_status", "full_docs", "text_chunks", "parent_chunks"):
        if name == "parent_chunks" and not (working_dir / _STORE_FILES[name]).is_file():
            continue
        rows = _kv(working_dir, name)
        kept = {}
        for key, row in rows.items():
            identity = _record_id(key, row)
            doc_id = str(row.get("doc_id") or row.get("full_doc_id") or row.get("document_id") or "")
            if name in {"text_chunks", "parent_chunks"} and (identity in removed_chunks or doc_id in docs):
                continue
            if name not in {"text_chunks", "parent_chunks"} and (identity in docs or doc_id in docs):
                continue
            kept[key] = row
        _save_kv(working_dir, name, kept)
    for name in ("full_entities", "full_relations"):
        rows = _kv(working_dir, name)
        kept = {}
        for key, row in rows.items():
            sources = split_values(row.get("source_id"))
            if sources and not [item for item in sources if item not in removed_chunks]:
                continue
            if set(sources) & removed_chunks:
                row["source_id"] = GRAPH_FIELD_SEP.join(item for item in sources if item not in removed_chunks)
                row["kb_id"] = GRAPH_FIELD_SEP.join(item for item in split_values(row.get("kb_id")) if item != kb_id)
            kept[key] = row
        _save_kv(working_dir, name, kept)
    if crash_after == "docs":
        raise ProcessCrash("进程中断")
    _maybe_fail("vectors")

    def edit_chunk(row: dict[str, Any]) -> str:
        identity = str(row.get("chunk_id") or row.get("__id__") or "")
        doc_id = str(row.get("doc_id") or row.get("full_doc_id") or "")
        if identity in removed_chunks or doc_id in docs:
            return "drop"
        return "keep"

    _edit_vdb(working_dir / _STORE_FILES["chunks_vdb"], edit_chunk)

    graph_path = working_dir / _GRAPH_FILE
    drop_nodes: list[str] = []

    def edit_graph(graph) -> None:
        for node_id, attrs in list(graph.nodes(data=True)):
            action = _rewrite_graph_attrs(attrs, removed_chunks, chunk_kb, kb_id)
            if action == "drop":
                drop_nodes.append(node_id)
            else:
                graph.nodes[node_id].update(attrs)
        drop_edges = []
        for source, target, attrs in list(graph.edges(data=True)):
            action = _rewrite_graph_attrs(attrs, removed_chunks, chunk_kb, kb_id)
            if action == "drop" or source in drop_nodes or target in drop_nodes:
                drop_edges.append((source, target))
            else:
                graph.edges[source, target].update(attrs)
        graph.remove_edges_from(drop_edges)
        graph.remove_nodes_from(drop_nodes)

    if graph_path.exists() or drop_nodes:
        rewrite_graph(graph_path, edit_graph)

    removed_names = {str(node_id) for node_id in drop_nodes}

    def edit_entity(row: dict[str, Any]) -> str:
        name = str(row.get("entity_name") or "")
        if name in removed_names:
            return "drop"
        sources = split_values(row.get("source_id"))
        kept = [item for item in sources if item not in removed_chunks]
        if sources and not kept:
            return "drop"
        if set(sources) & removed_chunks:
            row["source_id"] = GRAPH_FIELD_SEP.join(kept)
            docs = split_values(row.get("doc_id"))
            if len(docs) == len(sources):
                row["doc_id"] = GRAPH_FIELD_SEP.join(_aligned(docs, sources, set(kept)))
            row["kb_id"] = GRAPH_FIELD_SEP.join(item for item in split_values(row.get("kb_id")) if item != kb_id)
            row["content"] = GRAPH_FIELD_SEP.join(_aligned(split_values(row.get("content")), sources, set(kept)))
        return "keep"

    _edit_vdb(working_dir / _STORE_FILES["entities_vdb"], edit_entity)

    def edit_relation(row: dict[str, Any]) -> str:
        sources = split_values(row.get("source_id"))
        kept = [item for item in sources if item not in removed_chunks]
        src = str(row.get("src_id") or "")
        tgt = str(row.get("tgt_id") or "")
        if not sources:
            rest = [item for item in split_values(row.get("kb_id")) if item != kb_id]
            if kb_id not in split_values(row.get("kb_id")):
                return "keep"
            if not rest:
                return "drop"
            row["kb_id"] = GRAPH_FIELD_SEP.join(rest)
            return "keep"
        if src in removed_names or tgt in removed_names:
            return "drop"
        if sources and not kept:
            return "drop"
        if set(sources) & removed_chunks:
            row["source_id"] = GRAPH_FIELD_SEP.join(kept)
            docs = split_values(row.get("doc_id"))
            if len(docs) == len(sources):
                row["doc_id"] = GRAPH_FIELD_SEP.join(_aligned(docs, sources, set(kept)))
            row["kb_id"] = GRAPH_FIELD_SEP.join(item for item in split_values(row.get("kb_id")) if item != kb_id)
        elif kb_id in split_values(row.get("kb_id")) and not kept:
            return "drop"
        return "keep"

    _edit_vdb(working_dir / _STORE_FILES["relationships_vdb"], edit_relation)

    cache = []
    for entry in _load_cache(working_dir):
        chunks = {str(item) for item in entry.get("chunk_ids") or []}
        if chunks & removed_chunks:
            continue
        if purge_kb_metadata and entry.get("kb_id") == kb_id:
            continue
        cache.append(entry)
    _save_cache(working_dir, cache)

    for row in (data.get("doc_index") or {}).values():
        if not isinstance(row, dict) or str(row.get("doc_id") or "") not in docs:
            continue
        if preserve_index:
            row["chunk_ids"] = []
            continue
        owned = [item for item in row.get("chunk_ids") or [] if item in removed_chunks]
        row["deleted_at"] = row.get("deleted_at") or _now()
        row["chunk_ids"] = owned or sorted(removed_chunks)
        row["status"] = "deleted"
        key = str(row.get("storage_key") or "")
        if not key:
            continue
        path = input_dir / key
        if path.is_file() and _inside(input_dir, path):
            path.unlink()
        parent = path.parent
        if parent.is_dir() and _inside(input_dir, parent) and not any(parent.iterdir()):
            parent.rmdir()
    if purge_kb_metadata:
        data["file_bindings"] = {
            key: value for key, value in (data.get("file_bindings") or {}).items() if value != kb_id
        }
        data["qa_pairs"] = [item for item in data.get("qa_pairs") or [] if item.get("kb_id") != kb_id]
        data["comparisons"] = [item for item in data.get("comparisons") or [] if item.get("kb_id") != kb_id]
        audits = data.get("audits") or {}
        if isinstance(audits, dict):
            data["audits"] = {
                key: value for key, value in audits.items()
                if not isinstance(value, dict) or value.get("kb_id") != kb_id
            }
        sessions = data.get("sessions") or {}
        if isinstance(sessions, dict):
            data["sessions"] = {
                key: value for key, value in sessions.items()
                if not isinstance(value, dict) or value.get("kb_id") != kb_id
            }
    for path in list(working_dir.glob("*")):
        if path.is_file() and any(doc_id in path.name for doc_id in docs):
            if path.name not in set(_STORE_FILES.values()) | {_GRAPH_FILE, "product_shell.json", "semantic_cache.json"}:
                path.unlink()
    _drop_document_ir(working_dir, docs)
    return {"doc_ids": sorted(docs), "chunk_ids": sorted(removed_chunks)}


def _drop_document_ir(working_dir: Path, docs: set[str]) -> None:
    root = working_dir / "document_ir"
    if not root.exists() or not docs:
        return
    for path in list(root.rglob("*")):
        if not path.is_file():
            continue
        payload: dict[str, Any] = {}
        if path.suffix == ".json" or path.name.endswith(".json.bak"):
            loaded = _read_json(path)
            if isinstance(loaded, dict):
                payload = loaded
        document_id = str(payload.get("document_id") or "")
        if document_id in docs or any(doc_id in path.name for doc_id in docs):
            path.unlink()
    pending = root / "pending"
    if pending.is_dir() and not any(pending.iterdir()):
        pending.rmdir()


def execute_purge(
    working_dir: Path,
    input_dir: Path,
    job_id: str,
    *,
    crash_after: str | None = None,
) -> dict[str, Any]:
    working_dir = Path(working_dir)
    input_dir = Path(input_dir)
    data = load_shell(working_dir)
    job = next((item for item in data.get("purge_jobs") or [] if item.get("job_id") == job_id), None)
    if job is None:
        raise KeyError(job_id)
    kb_id = str(job.get("kb_id") or "")
    job["status"] = "running"
    job["step"] = "documents"
    doc_ids = [
        str(row.get("doc_id"))
        for row in (data.get("doc_index") or {}).values()
        if isinstance(row, dict) and row.get("kb_id") == kb_id and row.get("doc_id") and not row.get("deleted_at")
    ]
    for key, row in _kv(working_dir, "doc_status").items():
        if _hits(row, key, kb_id, set(doc_ids), set()):
            doc_ids.append(_record_id(key, row))
    doc_ids = sorted(set(doc_ids))
    discovered, _owners = _collect_chunks(working_dir, data, kb_id, set(doc_ids))
    known_chunks = set(job.get("checked_chunk_ids") or []) | discovered
    job["checked_doc_ids"] = doc_ids
    job["checked_chunk_ids"] = sorted(known_chunks)
    save_shell(working_dir, data)
    from lightrag.product_appdb import open_appdb

    open_appdb(working_dir).purge_kb(kb_id)
    try:
        removed = remove_documents(
            working_dir,
            input_dir,
            data,
            kb_id,
            doc_ids,
            crash_after=crash_after,
            purge_kb_metadata=True,
            known_chunks=known_chunks,
        )
        job["checked_chunk_ids"] = sorted(set(removed["chunk_ids"]) | known_chunks)
        for row in (data.get("doc_index") or {}).values():
            if isinstance(row, dict) and row.get("kb_id") == kb_id:
                row["chunk_ids"] = removed["chunk_ids"]
        save_shell(working_dir, data)
        report = deep_check(
            working_dir,
            kb_id,
            input_dir,
            doc_ids=doc_ids,
            chunk_ids=removed["chunk_ids"],
        )
        job["attempts"] = int(job.get("attempts") or 0) + 1
        job["checked_at"] = report["checked_at"]
        if report["passed"]:
            job["status"] = "succeeded"
            job["step"] = "checked"
            job["error"] = ""
        else:
            job["status"] = "consistency_failed"
            job["step"] = "checked"
            job["error"] = "删除后一致性检查未通过"
        save_shell(working_dir, data)
        report["job_status"] = job["status"]
        return report
    except ProcessCrash:
        data = load_shell(working_dir)
        current = next((item for item in data.get("purge_jobs") or [] if item.get("job_id") == job_id), job)
        current["status"] = "running"
        current["step"] = "documents"
        save_shell(working_dir, data)
        raise
    except Exception as exc:
        message = str(exc).strip() or "删除失败"
        if any(word in message.lower() for word in ("token", "password", "api_key", "cookie", "sk-")):
            message = "删除失败"
        job["attempts"] = int(job.get("attempts") or 0) + 1
        job["status"] = "failed"
        job["step"] = job.get("step") or "documents"
        job["error"] = message[:300]
        save_shell(working_dir, data)
        return {"passed": False, "job_status": "failed", "error": job["error"]}
