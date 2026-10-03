"""激活索引的模型清单。

查询前对照运行配置。模型、维度、instruction 或切块版本不一致时，
不允许继续搜索旧向量。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCHEMA_VERSION = 1
CHUNKING_VERSION = "retrieval-180-350-overlap-45"
PARENT_CHUNK_VERSION = "parent-800-1500-section"
NORMALIZATION = "l2"
MANIFEST_NAME = "index_manifest.json"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _env(name: str) -> str:
    return os.getenv(name, "").strip()


def corpus_revision(chunk_ids: list[str]) -> str:
    payload = "\n".join(sorted(str(item) for item in chunk_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_manifest(dimension: int, chunk_ids: list[str]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "embedding_provider": _env("EMBEDDING_BINDING"),
        "embedding_model": _env("EMBEDDING_MODEL"),
        "embedding_dimension": int(dimension),
        "embedding_instruction": _env("EMBEDDING_INSTRUCTION"),
        "normalization": NORMALIZATION,
        "chunking_version": CHUNKING_VERSION,
        "parent_chunk_version": PARENT_CHUNK_VERSION,
        "corpus_revision": corpus_revision(chunk_ids),
        "created_at": _now(),
    }


def manifest_path(working: Path) -> Path:
    return Path(working) / MANIFEST_NAME


def load_manifest(working: Path) -> dict[str, Any] | None:
    path = manifest_path(working)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def save_manifest(
    working: Path, dimension: int, chunk_ids: list[str]
) -> dict[str, Any]:
    payload = build_manifest(dimension, chunk_ids)
    path = manifest_path(working)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)
    return payload


def manifest_mismatches(manifest: dict[str, Any] | None) -> list[str]:
    if not manifest:
        return ["索引没有模型清单"]
    reasons: list[str] = []
    if int(manifest.get("schema_version") or 0) != SCHEMA_VERSION:
        reasons.append("清单结构版本不一致")
    expected_model = _env("EMBEDDING_MODEL")
    actual_model = str(manifest.get("embedding_model") or "")
    if expected_model and actual_model != expected_model:
        reasons.append("embedding 模型不一致")
    expected_provider = _env("EMBEDDING_BINDING")
    actual_provider = str(manifest.get("embedding_provider") or "")
    if expected_provider and actual_provider and expected_provider != actual_provider:
        reasons.append("embedding 供应商不一致")
    expected_dim = _env("EMBEDDING_DIM")
    if expected_dim:
        try:
            if int(expected_dim) != int(manifest.get("embedding_dimension") or 0):
                reasons.append("embedding 维度不一致")
        except ValueError:
            reasons.append("embedding 维度不一致")
    expected_instruction = _env("EMBEDDING_INSTRUCTION")
    actual_instruction = str(manifest.get("embedding_instruction") or "")
    if expected_instruction != actual_instruction and (
        expected_instruction or actual_instruction
    ):
        if _env("EMBEDDING_MODEL"):
            reasons.append("embedding instruction 不一致")
    if str(manifest.get("normalization") or "") != NORMALIZATION:
        reasons.append("向量归一化方式不一致")
    if str(manifest.get("chunking_version") or "") != CHUNKING_VERSION:
        reasons.append("切块版本不一致")
    if str(manifest.get("parent_chunk_version") or "") != PARENT_CHUNK_VERSION:
        reasons.append("父块版本不一致")
    return reasons


def compatibility_error(working: Path) -> dict[str, str] | None:
    """有向量文件但清单对不上时，返回 index_rebuild_required。不搜索旧向量。"""
    root = Path(working)
    vector_file = root / "vdb_chunks.json"
    if not vector_file.is_file():
        return None
    reasons = manifest_mismatches(load_manifest(root))
    if not reasons:
        return None
    return {
        "status": "index_rebuild_required",
        "reason": "；".join(reasons) + "。已停止查询旧向量，需要重建索引",
    }


def _read_vector_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def rebuild_index(working: Path, embed: Callable[[list[str]], Any]) -> dict[str, Any]:
    """在临时索引里重嵌，校验后再原子切换。失败时保留旧索引。"""
    import asyncio

    import numpy as np

    from lightrag.product_index import write_chunk_vectors
    from lightrag.product_storage import _kv, _vdb_rows

    root = Path(working)
    rows = _kv(root, "text_chunks")
    chunks = [
        dict(item)
        for item in rows.values()
        if isinstance(item, dict) and item.get("chunk_id")
    ]
    if not chunks:
        raise RuntimeError("没有可重建的切块")
    texts = [
        str(item.get("index_text") or item.get("content") or "") for item in chunks
    ]
    if any(not text.strip() for text in texts):
        raise RuntimeError("存在空切块，已保留旧索引")

    async def _embed() -> np.ndarray:
        raw = embed(texts)
        if asyncio.iscoroutine(raw):
            raw = await raw
        matrix = np.asarray(raw, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        return matrix

    try:
        matrix = asyncio.run(_embed())
    except RuntimeError as exc:
        if "asyncio.run() cannot be called" in str(exc):
            matrix = np.asarray(
                asyncio.get_event_loop().run_until_complete(_embed()), dtype=np.float32
            )
        else:
            raise
    if matrix.shape[0] != len(chunks):
        raise RuntimeError("重嵌数量与切块数量不一致，已保留旧索引")
    dimension = int(matrix.shape[1])
    if dimension < 2:
        raise RuntimeError("重嵌维度无效，已保留旧索引")
    sample = np.asarray(matrix[0], dtype=np.float32)
    again = embed([texts[0]])
    if asyncio.iscoroutine(again):
        try:
            again = asyncio.run(again)
        except RuntimeError:
            again = asyncio.get_event_loop().run_until_complete(again)
    again_vec = np.asarray(again, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(sample) * np.linalg.norm(again_vec)) or 1.0
    cosine = float(np.dot(sample, again_vec) / denom)
    if cosine < 0.98:
        raise RuntimeError("抽样检索未通过，已保留旧索引")

    staging = root / ".rebuild_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        grouped: dict[tuple[str, str, str], list[tuple[dict[str, Any], int]]] = {}
        for index, item in enumerate(chunks):
            key = (
                str(item.get("doc_id") or ""),
                str(item.get("kb_id") or ""),
                str(item.get("owner_id") or item.get("user_id") or ""),
            )
            grouped.setdefault(key, []).append((item, index))
        for (doc_id, kb_id, owner_id), pairs in grouped.items():
            if not doc_id or not kb_id or not owner_id:
                raise RuntimeError("切块缺少归属，已保留旧索引")
            part = np.vstack([matrix[index] for _item, index in pairs])
            write_chunk_vectors(
                staging, doc_id, kb_id, owner_id, [item for item, _index in pairs], part
            )
        written = _vdb_rows(staging / "vdb_chunks.json")
        if len(written) != len(chunks):
            raise RuntimeError("临时索引记录数量不一致，已保留旧索引")
        payload = _read_vector_payload(staging / "vdb_chunks.json")
        if int(payload.get("embedding_dim") or 0) != dimension:
            raise RuntimeError("临时索引维度不一致，已保留旧索引")
        for row in written:
            if not row.get("owner_id") or not row.get("kb_id") or not row.get("doc_id"):
                raise RuntimeError("临时索引缺少归属，已保留旧索引")
        target = root / "vdb_chunks.json"
        os.replace(staging / "vdb_chunks.json", target)
        manifest = save_manifest(root, dimension, [item["chunk_id"] for item in chunks])
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    return {
        "status": "rebuilt",
        "records": len(chunks),
        "dimension": dimension,
        "manifest": manifest,
    }


def run_index_rebuild(store: Any, job: dict[str, Any], working: Path) -> dict[str, Any]:
    """持久任务入口。失败保留旧索引，并把任务标成可重试。"""
    from lightrag.product_index import embed_texts

    store.transition(job["job_id"], "running", stage="reembedding", progress=20)
    try:
        report = rebuild_index(Path(working), embed_texts)
    except Exception as exc:
        message = str(exc).replace(os.getenv("EMBEDDING_BINDING_API_KEY", " "), "***")[
            :180
        ]
        return store.transition(
            job["job_id"],
            "failed",
            stage="failed",
            progress=0,
            error_code="rebuild",
            error_message=message or "索引重建失败，旧索引已保留",
        )
    return store.transition(
        job["job_id"],
        "succeeded",
        stage="ready",
        progress=100,
        error_code="",
        error_message="",
        output_summary=report,
    )
