"""
产品壳适配层。

前端沿用 AgentOS 的信息架构（知识库 / 文档 / 检索测试 / 问答 / 图谱 / 设置 / 比对 / 审核），
数据全部落到已经打通的 LightRAG 实例上。
"""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import Response, StreamingResponse
from starlette.websockets import WebSocketState
from pydantic import BaseModel, Field

from lightrag.base import QueryParam
from lightrag.constants import DEFAULT_ENTITY_TYPES
from lightrag.utils import logger

from lightrag.answer_pipeline import (
    PROMPT_VERSION,
    redact_text,
    should_summarize,
    stream_answer,
    summarize_prompt,
    window_and_older,
)
from lightrag.product_accounts import (
    LOCAL_OWNER_ID,
    AccountError,
    actor_id,
    create_auth_router,
    local_owner_view,
    migrate_shell_data,
    product_auth_enabled,
    public_user,
    upload_limits,
    visible_kbs,
    CURRENT_USER,
    access_is_current,
    decode_access_payload,
    inspect_upload,
)
from lightrag.product_scope import scope_visible
from lightrag.product_deletion import (
    begin_purge,
    public_job,
    queue_retry,
)
from lightrag.search_runtime import run_search_test

from .document_routes import (
    sanitize_filename,
)

router = APIRouter(prefix="/api/v1", tags=["product-shell"])

DEFAULT_KB_ID = "a3-default"
_STORE_LOCK = asyncio.Lock()
_BACKGROUND_TASKS: set[asyncio.Task] = set()
_UPLOAD_LOCKS: dict[str, asyncio.Lock] = {}

DEFAULT_RELATION_TYPES = [
    "发生于",
    "导致",
    "根本原因为",
    "属于",
    "采取措施",
    "验证于",
    "记录于",
    "负责",
    "改进效果为",
    "影响",
    "连接",
    "加速",
]

# 产品壳检索名 -> LightRAG QueryParam.mode
_MODE_MAP = {
    "hybrid": "hybrid",
    "mix": "mix",
    "graph": "local",
    "vector": "naive",
    "keyword": "global",
    "tree": "mix",
}

_STATUS_MAP = {
    "pending": "pending",
    "processing": "graphing",
    "processed": "ready",
    "failed": "failed",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _voice_local_intent(text: str) -> tuple[str, str] | None:
    """Handle short conversational controls before knowledge retrieval."""
    normalized = re.sub(r"[\s,，。.!！?？、]+", "", str(text or "")).lower()
    finish_phrases = {
        "结束",
        "结束吧",
        "结束对话",
        "结束通话",
        "停止",
        "停止吧",
        "退出",
        "再见",
        "拜拜",
        "不聊了",
        "先这样",
        "就这样",
        "就到这里",
        "今天先到这里",
        "ok结束吧",
        "okay结束吧",
        "好的结束吧",
        "好了结束吧",
    }
    if normalized in finish_phrases:
        return "finish", "好的，那我们先聊到这里。需要的时候再找我。"
    if normalized in {"谢谢", "谢谢你", "多谢"}:
        return "ack", "不客气，你可以继续问我。"
    if normalized in {"ok", "okay", "好", "好的", "明白了", "知道了", "可以了"}:
        return "ack", "好，你可以继续说。"
    return None


def _voice_thinking_ack(text: str) -> str:
    options = ("好，我查一下。", "明白，我看看相关资料。", "好的，稍等一下。")
    return options[sum(ord(char) for char in str(text or "")) % len(options)]


def _default_settings() -> dict[str, Any]:
    return {
        "retrieval_mode": "mix",
        "top_k": 10,
        "similarity_ratio": 0.5,
        "similarity_threshold": 0.2,
        "rerank_enabled": True,
        "graph_enabled": True,
        "context_expand": False,
        "ignore_whitespace": True,
        "only_changes": True,
        "mark_number_changes": True,
        "audit_auto_pass": False,
        "audit_reject_comment": True,
        "answer_detail": "standard",
        "show_citations": True,
    }


def _default_graph_config() -> dict[str, Any]:
    return {
        "entity_types": list(DEFAULT_ENTITY_TYPES),
        "relation_types": list(DEFAULT_RELATION_TYPES),
        "extraction_mode": "open",
        "domain": "general",
        "gleaning_rounds": 1,
        "custom_rules": "",
        "few_shot_examples": "",
    }


def _clean_text(value: Any, limit: int = 400) -> str:
    text = str(value or "")
    text = text.replace("<SEP>", "；")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        return text[:limit] + "…"
    return text


def _status_value(status: Any) -> str:
    if hasattr(status, "value"):
        return str(status.value)
    return str(status or "")


def _filename_of(file_path: str | None) -> str:
    if not file_path:
        return ""
    return Path(str(file_path)).name


class ShellStore:
    def __init__(self, working_dir: str):
        self.path = Path(working_dir) / "product_shell.json"

    def load(self) -> dict[str, Any]:
        from lightrag.product_storage import load_shell

        data = load_shell(self.path.parent)
        data.setdefault("kbs", [])
        data.setdefault("file_bindings", {})
        data.setdefault("comparisons", [])
        data.setdefault("audits", {})
        data.setdefault("qa_pairs", [])
        data.setdefault("sessions", {})
        data.setdefault("doc_index", {})
        changed = migrate_shell_data(data)
        if not any(kb.get("id") == DEFAULT_KB_ID for kb in data["kbs"]):
            data["kbs"].insert(
                0,
                {
                    "id": DEFAULT_KB_ID,
                    "name": "默认知识库",
                    "description": "当前工作区里已经入库的文档与图谱",
                    "created_at": _now(),
                    "owner_id": LOCAL_OWNER_ID,
                    "settings": _default_settings(),
                    "graph_config": _default_graph_config(),
                },
            )
            changed = True
        for kb in data["kbs"]:
            if (
                kb.get("id") == DEFAULT_KB_ID
                and kb.get("name") == "发动机故障 A3 知识库"
            ):
                kb["name"] = "默认知识库"
                kb["description"] = "当前工作区里已经入库的文档与图谱"
                changed = True
        if changed and self.path.exists():
            self.save(data)
        return data

    def save(self, data: dict[str, Any]) -> None:
        from lightrag.product_storage import save_shell

        save_shell(self.path.parent, data)

    def update(self, editor: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        from lightrag.product_storage import mutate_shell

        return mutate_shell(self.path.parent, editor)


def _kb_by_id(data: dict[str, Any], kb_id: str) -> dict[str, Any]:
    for kb in data["kbs"]:
        if kb.get("id") == kb_id:
            return kb
    raise HTTPException(status_code=404, detail="知识库不存在")


def _find_kb(data: dict[str, Any], kb_id: str) -> dict[str, Any]:
    kb = _kb_by_id(data, kb_id)
    if kb.get("deleted_at"):
        raise HTTPException(status_code=404, detail="知识库不存在")
    owner = kb.get("owner_id") or LOCAL_OWNER_ID
    if owner != actor_id():
        raise HTTPException(status_code=404, detail="知识库不存在")
    settings = kb.setdefault("settings", {})
    for key, value in _default_settings().items():
        settings.setdefault(key, value)
    kb.setdefault("graph_config", _default_graph_config())
    return kb


def _norm_lines(text: str, ignore_ws: bool) -> list[str]:
    lines = []
    for line in (text or "").splitlines():
        line = line.strip()
        if ignore_ws:
            line = re.sub(r"\s+", " ", line)
        if line:
            lines.append(line)
    return lines


def _numbers_changed(left: str, right: str) -> bool:
    pattern = re.compile(r"\d+(?:\.\d+)?")
    return set(pattern.findall(left or "")) != set(pattern.findall(right or ""))


def _text_diffs(
    old: str, new: str, ignore_ws: bool, mark_numbers: bool, limit: int = 40
) -> list[dict[str, str]]:
    old_lines = _norm_lines(old, ignore_ws)
    new_lines = _norm_lines(new, ignore_ws)
    diffs: list[dict[str, str]] = []
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines)
    labels = {"replace": "修改", "delete": "删除", "insert": "新增"}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        old_text = "\n".join(old_lines[i1:i2])[:500]
        new_text = "\n".join(new_lines[j1:j2])[:500]
        severity = (
            "高" if mark_numbers and _numbers_changed(old_text, new_text) else "中"
        )
        diffs.append(
            {
                "type": labels.get(tag, tag),
                "old": old_text,
                "new": new_text,
                "severity": severity,
            }
        )
        if len(diffs) >= limit:
            break
    return diffs


def _belongs(filename: str, kb_id: str, bindings: dict[str, str]) -> bool:
    return bindings.get(filename) == kb_id


def _chunk_mentions(source_id: Any) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for part in str(source_id or "").split("<SEP>"):
        part = part.strip()
        if not part or part in seen:
            continue
        seen.add(part)
        found.append(part)
    return found


def _kb_filenames(file_path: Any, kb_id: str, bindings: dict[str, str]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for part in str(file_path or "").split("<SEP>"):
        name = Path(part.replace("\\", "/")).name.strip()
        if not name or name in seen:
            continue
        if _belongs(name, kb_id, bindings):
            seen.add(name)
            found.append(name)
    return found


def _record_in_kb(file_path: Any, kb_id: str, bindings: dict[str, str]) -> bool:
    raw = str(file_path or "").strip()
    if not raw:
        return False
    return bool(_kb_filenames(raw, kb_id, bindings))


def _doc_in_kb(public: dict[str, Any], kb_id: str, data: dict[str, Any]) -> bool:
    index = data.get("doc_index") or {}
    doc_id = str(public.get("id") or "")
    row = index.get(doc_id)
    if isinstance(row, dict):
        return row.get("kb_id") == kb_id and not row.get("deleted_at")
    path = str(public.get("file_path") or "").replace("\\", "/")
    for item in index.values():
        if (
            not isinstance(item, dict)
            or item.get("deleted_at")
            or item.get("kb_id") != kb_id
        ):
            continue
        key = str(item.get("storage_key") or "")
        if key and key in path:
            return True
    return _belongs(
        str(public.get("name") or ""), kb_id, data.get("file_bindings") or {}
    )


def _node_in_scope(
    record: dict[str, Any], kb_id: str, bindings: dict[str, str]
) -> bool:
    return scope_visible(record, kb_id, bindings)


def scoped_graph_records(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    kb_id: str,
    bindings: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept_nodes = [node for node in nodes if _node_in_scope(node, kb_id, bindings)]
    names = {str(node.get("id") or "") for node in kept_nodes}
    kept_edges = []
    for edge in edges:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source not in names or target not in names:
            continue
        if not scope_visible(edge, kb_id, bindings):
            continue
        kept_edges.append(edge)
    return kept_nodes, kept_edges


def _relation_labels(raw: Any) -> list[str]:
    text = str(raw or "").replace("<SEP>", "；")
    found: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[；;，,]", text):
        label = re.sub(r"\s+", " ", part).strip()
        if not label or label in seen:
            continue
        seen.add(label)
        found.append(label)
    return found


def _map_mode(retrieval_mode: str | None, graph_enabled: bool = True) -> str:
    if not graph_enabled:
        return "naive"
    return _MODE_MAP.get((retrieval_mode or "mix").lower(), "mix")


def _public_error(exc: Exception) -> HTTPException:
    text = str(exc)
    if "balance is insufficient" in text or "30001" in text:
        return HTTPException(
            status_code=402,
            detail="模型账户余额不足，检索和问答暂时无法调用大模型。知识列表和图谱仍可查看。",
        )
    logger.exception("产品壳请求失败")
    return HTTPException(status_code=500, detail=text[:500] or "请求失败")


async def _embedding_for(rag: Any, text: str) -> list[float]:
    try:
        values = await rag.embedding_func([text])
        row = values[0] if len(values) else []
        if hasattr(row, "tolist"):
            row = row.tolist()
        return [float(value) for value in row]
    except Exception as exc:
        logger.warning("记忆向量生成失败，回退关键词召回: %s", exc)
        return []


def _citations_from_answer(answer: str) -> list[dict[str, Any]]:
    cites: list[dict[str, Any]] = []
    for raw in answer.splitlines():
        line = raw.strip().lstrip("-*").strip()
        kind = None
        if line.startswith("[KG]"):
            kind = "KG"
            name = line[4:].strip()
        elif line.startswith("[DC]"):
            kind = "DC"
            name = line[4:].strip()
        else:
            continue
        if name:
            cites.append({"doc_name": name, "chunk_id": kind, "page_num": None})
    return cites[:8]


async def _list_doc_records(rag) -> list[dict[str, Any]]:
    from lightrag.base import DocStatus

    statuses = (
        DocStatus.PENDING,
        DocStatus.PROCESSING,
        DocStatus.PROCESSED,
        DocStatus.FAILED,
    )
    grouped = await asyncio.gather(
        *[rag.get_docs_by_status(status) for status in statuses]
    )
    records: list[dict[str, Any]] = []
    for bucket in grouped:
        for doc_id, doc in bucket.items():
            records.append({"id": doc_id, "doc": doc})
    records.sort(
        key=lambda row: str(getattr(row["doc"], "updated_at", "") or ""),
        reverse=True,
    )
    return records


def _doc_field(doc: Any, name: str, default: Any = None) -> Any:
    if isinstance(doc, dict):
        return doc.get(name, default)
    return getattr(doc, name, default)


def _file_size(file_path: str, name: str, input_dir: Path) -> int:
    candidates = []
    if file_path:
        candidates.append(Path(file_path))
    if name:
        candidates.append(input_dir / name)
        candidates.append(input_dir / "__enqueued__" / name)
    for path in candidates:
        try:
            if path.is_file():
                return path.stat().st_size
        except OSError:
            continue
    return 0


def _public_doc(row: dict[str, Any], input_dir: Path) -> dict[str, Any]:
    from lightrag.ingest_progress import clear, get

    doc = row["doc"]
    file_path = _doc_field(doc, "file_path", "") or ""
    name = _filename_of(file_path) or file_path or row["id"]
    raw_status = _status_value(_doc_field(doc, "status", ""))
    status = _STATUS_MAP.get(raw_status.lower(), raw_status or "pending")
    size = _file_size(file_path, name, input_dir)
    job = get(name)
    if job and job.get("file_size") and not size:
        size = int(job["file_size"])
    if status == "ready":
        clear(name)
        progress = 100
    elif status == "failed":
        clear(name)
        progress = 0
    elif job:
        status = job.get("stage") or status
        progress = int(job.get("progress") or 0)
    elif status == "pending":
        status, progress = "parsing", 18
    elif status == "graphing":
        progress = 60
    else:
        progress = 0
    updated = _doc_field(doc, "updated_at") or _doc_field(doc, "created_at") or _now()
    return {
        "id": row["id"],
        "name": name,
        "chunk_count": _doc_field(doc, "chunks_count", 0) or 0,
        "char_count": _doc_field(doc, "content_length", 0) or 0,
        "file_size": size,
        "updated_at": str(updated),
        "created_at": str(_doc_field(doc, "created_at", "") or updated),
        "status": status,
        "progress": progress,
        "error_msg": _doc_field(doc, "error_msg"),
        "file_path": file_path,
    }


def _entity_display_name(node_id: str, props: dict[str, Any]) -> str:
    name = _clean_text(props.get("name") or props.get("entity_name"), 120).strip()
    if name:
        return name
    # Legacy graphs used the entity name as the node key. New graphs use an
    # opaque ent-* key and keep the user-facing name in node properties.
    if node_id and not re.match(r"^ent-[0-9a-f]+$", node_id, re.IGNORECASE):
        return _clean_text(node_id, 120)
    return "未命名实体"


def _graph_document_refs(
    props: dict[str, Any],
    data: dict[str, Any],
    kb_id: str,
    bindings: dict[str, str],
) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    seen: set[str] = set()
    index = data.get("doc_index") or {}
    for doc_id in str(props.get("doc_id") or "").split("<SEP>"):
        doc_id = doc_id.strip()
        row = index.get(doc_id)
        if not doc_id or doc_id in seen or not isinstance(row, dict):
            continue
        if row.get("kb_id") != kb_id or row.get("deleted_at"):
            continue
        seen.add(doc_id)
        refs.append(
            {"document_id": doc_id, "name": str(row.get("display_name") or "资料")}
        )
    for name in _kb_filenames(props.get("file_path"), kb_id, bindings):
        key = f"file:{name}"
        if key in seen:
            continue
        seen.add(key)
        refs.append({"document_id": "", "name": name})
    return refs


def _node_view(
    node_id: str,
    props: dict[str, Any],
    degree: int = 1,
    documents: Optional[list[dict[str, str]]] = None,
) -> dict[str, Any]:
    entity_type = props.get("entity_type") or "未知"
    if isinstance(entity_type, list):
        entity_type = entity_type[0] if entity_type else "未知"
    return {
        "id": node_id,
        "label": _entity_display_name(node_id, props),
        "type": str(entity_type),
        "desc": _clean_text(props.get("description"), 160),
        "frequency": max(int(degree or 1), 1),
        "documents": documents or [],
    }


def _edge_view(source: str, target: str, props: dict[str, Any]) -> dict[str, Any]:
    label = props.get("keywords") or props.get("description") or "关联"
    weight = props.get("weight") or 1
    try:
        weight = float(weight)
    except (TypeError, ValueError):
        weight = 1.0
    return {
        "from": source,
        "to": target,
        "label": _clean_text(label, 24),
        "desc": _clean_text(props.get("description"), 80),
        "weight": weight,
    }


class KBCreate(BaseModel):
    name: str
    description: Optional[str] = ""


class KBUpdate(BaseModel):
    name: str
    description: Optional[str] = None


class ChatBody(BaseModel):
    query: str
    kb_id: str
    retrieval_mode: Optional[str] = "mix"
    top_k: Optional[int] = None
    similarity_ratio: Optional[float] = None
    enable_rerank: Optional[bool] = None
    graph_enabled: Optional[bool] = None
    similarity_threshold: Optional[float] = None
    context_expand: Optional[bool] = None
    history: list[dict[str, Any]] = Field(default_factory=list)
    stream: Optional[bool] = False
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    answer_detail: Optional[str] = None
    voice_stream: bool = False


class ConversationCreate(BaseModel):
    kb_id: str
    title: Optional[str] = None


class ConversationPatch(BaseModel):
    kb_id: str
    title: Optional[str] = None
    summary: Optional[str] = None
    is_archived: Optional[bool] = None


class FeedbackBody(BaseModel):
    kb_id: str
    rating: str
    reason: Optional[str] = None
    comment: Optional[str] = None


class MemoryCandidateBody(BaseModel):
    kb_id: str
    content: str
    category: Optional[str] = "other"
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None


class MemoryPatch(BaseModel):
    content: Optional[str] = None
    category: Optional[str] = None
    enabled: Optional[bool] = None
    scope: Optional[str] = None
    kb_id: Optional[str] = None
    expires_at: Optional[str] = None


class VoiceSessionCreate(BaseModel):
    kb_id: str
    goal: str = "free"


class VoiceTurnBody(BaseModel):
    transcript: str
    kb_id: Optional[str] = None


class VoiceFinishBody(BaseModel):
    summary: Optional[str] = None
    kb_id: Optional[str] = None


class VoiceSpeechBody(BaseModel):
    turn_id: str
    voice: Optional[str] = None
    kb_id: Optional[str] = None


class SearchBody(BaseModel):
    query: str
    kb_id: str
    retrieval_mode: Optional[str] = "mix"
    top_k: int = 10
    similarity_ratio: float = 0.5
    enable_rerank: bool = False
    similarity_threshold: Optional[float] = None
    history: Optional[list[str]] = None


class ComparisonBody(BaseModel):
    kb_id: str
    doc_id_old: str
    doc_id_new: str
    reason: Optional[str] = ""
    compare_type: Optional[str] = "file"


class QABody(BaseModel):
    kb_id: str
    question: str
    answer: str
    doc_id: Optional[str] = ""
    doc_name: Optional[str] = ""


class GraphConfigBody(BaseModel):
    entity_types: list[str] = Field(default_factory=list)
    relation_types: list[str] = Field(default_factory=list)
    extraction_mode: str = "preset"
    domain: str = "fault"
    gleaning_rounds: int = 1
    custom_rules: str = ""
    few_shot_examples: str = ""


def create_product_shell_routes(rag, doc_manager, api_key: Optional[str] = None):
    router.routes.clear()
    router.dependencies.clear()
    router._product_guard = False
    store = ShellStore(rag.working_dir)
    input_dir = Path(doc_manager.input_dir)
    saved: dict[str, Any] = {}
    try:
        saved = store.load()
        kbs = saved.get("kbs") or []
        mode = "open"
        if kbs:
            mode = (kbs[0].get("graph_config") or {}).get("extraction_mode") or "open"
        addon = getattr(rag, "addon_params", None)
        if isinstance(addon, dict):
            addon["kg_schema"] = "open" if mode == "open" else "fault"
    except Exception:
        pass

    async def _read() -> dict[str, Any]:
        async with _STORE_LOCK:
            return store.load()

    async def _write(data: dict[str, Any]) -> None:
        async with _STORE_LOCK:
            store.save(data)

    def _app():
        from lightrag.product_appdb import open_appdb

        return open_appdb(Path(rag.working_dir))

    def _conversation_title(query: str) -> str:
        text = " ".join((query or "").split())
        if not text:
            return "新会话"
        return text[:32] + ("…" if len(text) > 32 else "")

    async def _bind_user(
        request: Request = None, authorization: str | None = Header(default=None)
    ):
        # WebSocket routes do not receive an HTTP Request object. Their own
        # handler performs token validation from headers/query/subprotocol.
        if request is None:
            yield
            return
        path = request.url.path.rstrip("/")
        if path in {
            "/api/v1/auth/register",
            "/api/v1/auth/login",
            "/api/v1/auth/refresh",
            "/api/v1/auth/status",
        }:
            yield
            return
        if not product_auth_enabled():
            token = CURRENT_USER.set(local_owner_view())
            try:
                yield
            finally:
                CURRENT_USER.reset(token)
            return
        raw = ""
        if authorization and authorization.lower().startswith("bearer "):
            raw = authorization[7:].strip()
        payload = {}
        if raw:
            try:
                payload = decode_access_payload(raw)
            except Exception:
                payload = {}
        if not payload:
            raise HTTPException(status_code=401, detail="未登录")
        data = await _read()
        user = access_is_current(data, payload)
        if user is None:
            raise HTTPException(status_code=401, detail="未登录")
        token = CURRENT_USER.set(public_user(user))
        try:
            yield
        finally:
            CURRENT_USER.reset(token)

    if not getattr(router, "_product_guard", False):
        router.dependencies.append(Depends(_bind_user))
        router.include_router(create_auth_router(_read, _write))
        router._product_guard = True

    @router.get("/kb")
    async def list_kb(tenant_id: str = "default"):
        data = await _read()
        return {"items": visible_kbs(data, actor_id()), "tenant_id": tenant_id}

    @router.post("/kb")
    async def create_kb(body: KBCreate):
        name = body.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="请输入知识库名称")
        data = await _read()
        kb = {
            "id": uuid4().hex,
            "name": name,
            "description": (body.description or "").strip(),
            "created_at": _now(),
            "owner_id": actor_id(),
            "settings": _default_settings(),
            "graph_config": _default_graph_config(),
        }
        data["kbs"].append(kb)
        await _write(data)
        return kb

    async def _doc_chunks(doc_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        status = await rag.doc_status.get_by_id(doc_id)
        if status is None:
            raise HTTPException(status_code=404, detail="文档不存在")
        public = _public_doc({"id": doc_id, "doc": status}, input_dir)
        chunk_ids = _doc_field(status, "chunks_list") or []
        items = []
        for index, chunk_id in enumerate(chunk_ids):
            row = await rag.text_chunks.get_by_id(chunk_id)
            content = ""
            parent_id = ""
            parent_content = ""
            if isinstance(row, dict):
                content = row.get("content") or ""
                parent_id = row.get("parent_id") or ""
                parent_content = row.get("parent_content") or ""
            item = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "doc_name": public["name"],
                "index": index + 1,
                "content": content,
                "parent_id": parent_id,
                "parent_content": parent_content,
                "excerpt": _clean_text(content, 180),
                "char_count": len(content),
                "updated_at": public["updated_at"],
            }
            if isinstance(row, dict):
                from lightrag.product_document_ir import apply_location

                apply_location(item, row)
            items.append(item)
        return public, items

    async def _kb_chunks(kb_id: str, data: dict[str, Any]) -> list[dict[str, Any]]:
        records = await _list_doc_records(rag)
        items: list[dict[str, Any]] = []
        for row in records:
            public = _public_doc(row, input_dir)
            if not _doc_in_kb(public, kb_id, data):
                continue
            _, chunks = await _doc_chunks(public["id"])
            items.extend(chunks)
        return items

    def _build_comparison(
        body: ComparisonBody,
        settings: dict[str, Any],
        old_doc,
        old_chunks,
        new_doc,
        new_chunks,
        qa_pairs,
    ):
        ignore_ws = bool(settings.get("ignore_whitespace", True))
        mark_numbers = bool(settings.get("mark_number_changes", True))
        only_changes = bool(settings.get("only_changes", True))
        old_text = "\n".join(chunk["content"] for chunk in old_chunks)
        new_text = "\n".join(chunk["content"] for chunk in new_chunks)
        file_diffs = _text_diffs(old_text, new_text, ignore_ws, mark_numbers)
        chunk_diffs = []
        shared = min(len(old_chunks), len(new_chunks))
        for index in range(shared):
            left = old_chunks[index]["content"]
            right = new_chunks[index]["content"]
            same = left.strip() == right.strip() or (
                ignore_ws and re.sub(r"\s+", "", left) == re.sub(r"\s+", "", right)
            )
            if same:
                if only_changes:
                    continue
                chunk_diffs.append(
                    {
                        "type": "相同",
                        "index": index + 1,
                        "old": _clean_text(left, 240),
                        "new": _clean_text(right, 240),
                        "severity": "低",
                    }
                )
                continue
            chunk_diffs.append(
                {
                    "type": "修改",
                    "index": index + 1,
                    "old": _clean_text(left, 240),
                    "new": _clean_text(right, 240),
                    "severity": "高"
                    if mark_numbers and _numbers_changed(left, right)
                    else "中",
                }
            )
        for chunk in old_chunks[shared:]:
            chunk_diffs.append(
                {
                    "type": "删除",
                    "index": chunk["index"],
                    "old": chunk["excerpt"],
                    "new": "",
                    "severity": "中",
                }
            )
        for chunk in new_chunks[shared:]:
            chunk_diffs.append(
                {
                    "type": "新增",
                    "index": chunk["index"],
                    "old": "",
                    "new": chunk["excerpt"],
                    "severity": "中",
                }
            )
        chunk_diffs = chunk_diffs[:40]

        scoped_pairs = [item for item in qa_pairs if item.get("kb_id") == body.kb_id]
        old_qa = [
            item for item in scoped_pairs if item.get("doc_id") == body.doc_id_old
        ]
        new_qa = [
            item for item in scoped_pairs if item.get("doc_id") == body.doc_id_new
        ]
        new_by_q = {
            re.sub(r"\s+", "", item.get("question") or ""): item for item in new_qa
        }
        qa_diffs = []
        seen = set()
        for item in old_qa:
            key = re.sub(r"\s+", "", item.get("question") or "")
            seen.add(key)
            other = new_by_q.get(key)
            if other is None:
                qa_diffs.append(
                    {
                        "type": "仅旧文档",
                        "question": item.get("question"),
                        "old": _clean_text(item.get("answer"), 240),
                        "new": "",
                        "severity": "中",
                    }
                )
            elif _clean_text(item.get("answer"), 240) != _clean_text(
                other.get("answer"), 240
            ):
                qa_diffs.append(
                    {
                        "type": "答案不同",
                        "question": item.get("question"),
                        "old": _clean_text(item.get("answer"), 240),
                        "new": _clean_text(other.get("answer"), 240),
                        "severity": "高"
                        if mark_numbers
                        and _numbers_changed(
                            item.get("answer") or "", other.get("answer") or ""
                        )
                        else "中",
                    }
                )
        for item in new_qa:
            key = re.sub(r"\s+", "", item.get("question") or "")
            if key in seen:
                continue
            qa_diffs.append(
                {
                    "type": "仅新文档",
                    "question": item.get("question"),
                    "old": "",
                    "new": _clean_text(item.get("answer"), 240),
                    "severity": "中",
                }
            )
        old_questions = "\n".join(re.findall(r"[^\n。！]{6,80}[？?]", old_text)[:12])
        new_questions = "\n".join(re.findall(r"[^\n。！]{6,80}[？?]", new_text)[:12])
        for diff in _text_diffs(
            old_questions, new_questions, ignore_ws, mark_numbers, 20
        ):
            qa_diffs.append(
                {
                    "type": "文档问句" + diff["type"],
                    "question": diff["new"] or diff["old"],
                    "old": diff["old"],
                    "new": diff["new"],
                    "severity": diff["severity"],
                }
            )
        if not qa_diffs:
            for diff in file_diffs[:8]:
                qa_diffs.append(
                    {
                        "type": "要点" + diff["type"],
                        "question": (diff["new"] or diff["old"])[:80],
                        "old": diff["old"],
                        "new": diff["new"],
                        "severity": diff["severity"],
                    }
                )
        return {
            "task_id": uuid4().hex,
            "kb_id": body.kb_id,
            "compare_type": body.compare_type or "file",
            "doc_id_old": body.doc_id_old,
            "doc_id_new": body.doc_id_new,
            "doc_name_old": old_doc["name"],
            "doc_name_new": new_doc["name"],
            "reason": body.reason or "版本差异",
            "status": "done",
            "created_at": _now(),
            "summary": {
                "file_changes": len(file_diffs),
                "chunk_changes": len(chunk_diffs),
                "qa_changes": len(qa_diffs),
            },
            "file_diffs": file_diffs,
            "chunk_diffs": chunk_diffs,
            "qa_diffs": qa_diffs[:40],
        }

    @router.get("/kb/comparison")
    async def list_comparison(kb_id: str, compare_type: str = ""):
        data = await _read()
        _find_kb(data, kb_id)
        items = [row for row in data["comparisons"] if row.get("kb_id") == kb_id]
        if compare_type:
            items = [
                row for row in items if row.get("compare_type", "file") == compare_type
            ]
        return {"items": items}

    @router.post("/kb/comparison")
    async def create_comparison(body: ComparisonBody):
        if body.doc_id_old == body.doc_id_new:
            raise HTTPException(status_code=400, detail="请选择两份不同的文档")
        data = await _read()
        kb = _find_kb(data, body.kb_id)
        old_doc, old_chunks = await _doc_chunks(body.doc_id_old)
        new_doc, new_chunks = await _doc_chunks(body.doc_id_new)
        task = _build_comparison(
            body,
            kb["settings"],
            old_doc,
            old_chunks,
            new_doc,
            new_chunks,
            data.get("qa_pairs") or [],
        )
        data["comparisons"].insert(0, task)
        await _write(data)
        return task

    @router.get("/kb/comparison/{task_id}")
    async def get_comparison(task_id: str):
        data = await _read()
        for row in data["comparisons"]:
            if row.get("task_id") == task_id:
                _find_kb(data, str(row.get("kb_id") or ""))
                return row
        raise HTTPException(status_code=404, detail="比对任务不存在")

    def _audit_item(
        data: dict[str, Any],
        item_id: str,
        name: str,
        action_type: str,
        chunk_info: str,
        updated_at: str,
        default_status: str,
    ) -> dict[str, Any]:
        saved = data["audits"].get(item_id, {})
        return {
            "id": item_id,
            "name": name,
            "action_type": action_type,
            "chunk_info": chunk_info,
            "updated_at": saved.get("updated_at") or updated_at,
            "status": saved.get("status", default_status),
            "comment": saved.get("comment", ""),
        }

    @router.get("/kb/audit")
    async def list_audit(kb_id: str, item_type: str = "document", status: str = ""):
        data = await _read()
        kb = _find_kb(data, kb_id)
        default_status = "已通过" if kb["settings"].get("audit_auto_pass") else "待审核"
        items = []
        if item_type == "qa":
            for qa in data.get("qa_pairs") or []:
                if qa.get("kb_id") != kb_id:
                    continue
                items.append(
                    _audit_item(
                        data,
                        qa["id"],
                        qa.get("question") or "问答",
                        "问答",
                        _clean_text(qa.get("answer"), 80),
                        qa.get("updated_at") or "",
                        default_status,
                    )
                )
        elif item_type == "chunk":
            for chunk in await _kb_chunks(kb_id, data):
                items.append(
                    _audit_item(
                        data,
                        chunk["chunk_id"],
                        f"{chunk['doc_name']} · 第 {chunk['index']} 块",
                        "文本块",
                        chunk["excerpt"],
                        chunk["updated_at"],
                        default_status,
                    )
                )
        else:
            records = await _list_doc_records(rag)
            for row in records:
                public = _public_doc(row, input_dir)
                if not _doc_in_kb(public, kb_id, data):
                    continue
                if public["status"] != "ready":
                    continue
                items.append(
                    _audit_item(
                        data,
                        public["id"],
                        public["name"],
                        "入库抽取",
                        f"{public['chunk_count']} 段 · {public['char_count']} 字",
                        public["updated_at"],
                        default_status,
                    )
                )
        if status:
            items = [item for item in items if item["status"] == status]
        return {"items": items, "total": len(items), "item_type": item_type}

    @router.put("/kb/audit/{item_id}")
    async def review_audit(item_id: str, kb_id: str, action: str, comment: str = ""):
        if action not in {"approved", "rejected"}:
            raise HTTPException(
                status_code=400, detail="action 只能是 approved 或 rejected"
            )
        data = await _read()
        kb = _find_kb(data, kb_id)
        if (
            action == "rejected"
            and kb["settings"].get("audit_reject_comment")
            and not (comment or "").strip()
        ):
            raise HTTPException(status_code=400, detail="拒绝时需要填写原因")
        data["audits"][item_id] = {
            "kb_id": kb_id,
            "status": "已通过" if action == "approved" else "已拒绝",
            "comment": comment,
            "updated_at": _now(),
        }
        await _write(data)
        return {"id": item_id, "status": data["audits"][item_id]["status"]}

    @router.get("/kb/{kb_id}")
    async def get_kb(kb_id: str):
        data = await _read()
        return _find_kb(data, kb_id)

    @router.put("/kb/{kb_id}")
    async def update_kb(kb_id: str, body: KBUpdate):
        name = body.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="请输入知识库名称")
        data = await _read()
        kb = _find_kb(data, kb_id)
        kb["name"] = name
        if body.description is not None:
            kb["description"] = body.description.strip()
        await _write(data)
        return kb

    async def _run_purge(job_id: str) -> None:
        from lightrag.product_storage import execute_purge

        await asyncio.to_thread(execute_purge, Path(rag.working_dir), input_dir, job_id)

    def _schedule_purge(job_id: str) -> None:
        task = asyncio.create_task(_run_purge(job_id))
        _BACKGROUND_TASKS.add(task)
        task.add_done_callback(_BACKGROUND_TASKS.discard)

    def _task_store():
        from lightrag.product_db import open_store

        return open_store(Path(rag.working_dir) / "product_jobs.sqlite")

    def _public_task(
        job_id: str, fallback: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        from lightrag.product_db import public_job

        job = _task_store().get_job(job_id)
        if job is None:
            return fallback or {}
        return public_job(job)

    def _enqueue_purge(shell_job: dict[str, Any], kb_id: str) -> dict[str, Any]:
        from lightrag.product_db import public_job

        job = _task_store().create_job(
            {
                "job_id": shell_job["job_id"],
                "job_type": "purge",
                "user_id": actor_id(),
                "kb_id": kb_id,
                "doc_id": "",
                "idempotency_key": f"purge:{kb_id}:{shell_job['job_id']}",
                "input_snapshot": {"kb_id": kb_id},
            }
        )
        return public_job(job)

    def _enqueue_existing_purge(shell_job: dict[str, Any]) -> None:
        store = _task_store()
        existing = store.get_job(str(shell_job.get("job_id") or ""))
        if existing is None:
            _enqueue_purge(shell_job, str(shell_job.get("kb_id") or ""))
            return
        if existing.get("status") in {"failed", "timeout", "consistency_failed"}:
            store.request_retry(existing["job_id"], actor_id())

    def _owned_job(data: dict[str, Any], job_id: str) -> dict[str, Any]:
        job = next(
            (
                item
                for item in data.get("purge_jobs") or []
                if item.get("job_id") == job_id
            ),
            None,
        )
        if job is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        kb = _kb_by_id(data, str(job.get("kb_id") or ""))
        owner = kb.get("owner_id") or LOCAL_OWNER_ID
        if owner != actor_id():
            raise HTTPException(status_code=404, detail="任务不存在")
        return job

    def _owned_task(job_id: str) -> dict[str, Any]:
        job = _task_store().get_job(job_id)
        if job is None or job.get("user_id") != actor_id():
            raise HTTPException(status_code=404, detail="任务不存在")
        return job

    @router.get("/jobs/{job_id}")
    async def get_job(job_id: str):
        from lightrag.product_db import public_job

        return public_job(_owned_task(job_id))

    @router.get("/jobs")
    async def list_jobs(
        kb_id: str = "",
        doc_id: str = "",
        status: str = "",
        page: int = 1,
        page_size: int = 20,
    ):
        from lightrag.product_db import public_job

        found = _task_store().list_jobs(
            user_id=actor_id(),
            kb_id=kb_id,
            doc_id=doc_id,
            status=status,
            page=page,
            page_size=page_size,
        )
        found["items"] = [public_job(item) for item in found["items"]]
        return found

    @router.get("/documents/{doc_id}/jobs")
    async def list_document_jobs(doc_id: str, page: int = 1, page_size: int = 20):
        from lightrag.product_db import public_job

        found = _task_store().list_jobs(
            user_id=actor_id(), doc_id=doc_id, page=page, page_size=page_size
        )
        found["items"] = [public_job(item) for item in found["items"]]
        return found

    @router.get("/kb/{kb_id}/jobs")
    async def list_kb_jobs(
        kb_id: str, status: str = "", page: int = 1, page_size: int = 20
    ):
        data = await _read()
        _find_kb(data, kb_id)
        from lightrag.product_db import public_job

        found = _task_store().list_jobs(
            user_id=actor_id(),
            kb_id=kb_id,
            status=status,
            page=page,
            page_size=page_size,
        )
        found["items"] = [public_job(item) for item in found["items"]]
        return found

    @router.post("/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str):
        from lightrag.product_db import public_job

        try:
            job = _task_store().request_cancel(job_id, actor_id())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if job is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        return public_job(job)

    @router.post("/jobs/{job_id}/retry")
    async def retry_job(job_id: str):
        from lightrag.product_db import public_job

        try:
            job = _task_store().request_retry(job_id, actor_id())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if job is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        return public_job(job)

    @router.get("/kb/purge-jobs/{job_id}")
    async def get_purge_job(job_id: str):
        data = await _read()
        owned = _owned_job(data, job_id)
        return _public_task(job_id, public_job(owned))

    @router.post("/kb/purge-jobs/{job_id}/retry")
    async def retry_purge_job(job_id: str):
        data = await _read()
        job = _owned_job(data, job_id)
        try:
            queue_retry(job)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        await _write(data)
        _enqueue_existing_purge(job)
        return _public_task(job["job_id"], public_job(job))

    async def _check_target(job_id: str) -> tuple[str, dict[str, Any]]:
        data = await _read()
        try:
            job = _owned_job(data, job_id)
        except HTTPException:
            task = _owned_task(job_id)
            if str(task.get("job_type") or "") not in {"delete_document", "purge_kb"}:
                raise HTTPException(status_code=404, detail="任务不存在") from None
            job = task
        return str(job.get("kb_id") or ""), job

    @router.get("/kb/purge-jobs/{job_id}/check")
    async def check_purge_job(job_id: str):
        return await deep_check_job(job_id)

    @router.get("/kb/purge-jobs/{job_id}/deep-check")
    async def deep_check_job(job_id: str):
        from lightrag.product_storage import deep_check

        kb_id, job = await _check_target(job_id)
        report = await asyncio.to_thread(
            deep_check, Path(rag.working_dir), kb_id, input_dir
        )
        report["job"] = public_job(job)
        return report

    @router.delete("/kb/{kb_id}")
    async def delete_kb(kb_id: str):
        data = await _read()
        kb = _find_kb(data, kb_id)
        job = begin_purge(data, kb)
        await _write(data)
        from lightrag.product_appdb import open_appdb

        open_appdb(Path(rag.working_dir)).purge_kb(kb_id)
        created = _enqueue_purge(job, kb_id)
        return {"message": "已移入回收站，正在清理", **created}

    @router.get("/capabilities")
    async def capabilities():
        from lightrag.product_accounts import upload_capability

        return upload_capability()

    @router.get("/kb/{kb_id}/index-status")
    async def index_status(kb_id: str):
        data = await _read()
        _find_kb(data, kb_id)
        from lightrag.index_manifest import compatibility_error, load_manifest

        root = Path(rag.working_dir)
        blocked = compatibility_error(root)
        manifest = load_manifest(root) or {}
        return {
            "status": "index_rebuild_required" if blocked else "ready",
            "reason": (blocked or {}).get("reason") or "",
            "embedding_model": manifest.get("embedding_model")
            or os.getenv("EMBEDDING_MODEL")
            or "",
            "embedding_dimension": manifest.get("embedding_dimension"),
            "chunking_version": manifest.get("chunking_version") or "",
            "parent_chunk_version": manifest.get("parent_chunk_version") or "",
            "corpus_revision": manifest.get("corpus_revision") or "",
            "created_at": manifest.get("created_at") or "",
            "rerank_model": os.getenv("RERANK_MODEL") or "",
        }

    @router.post("/kb/{kb_id}/index-rebuild")
    async def start_index_rebuild(kb_id: str):
        data = await _read()
        _find_kb(data, kb_id)
        from lightrag.product_db import public_job

        job = _task_store().create_job(
            {
                "job_type": "index_rebuild",
                "user_id": actor_id(),
                "kb_id": kb_id,
                "doc_id": f"index-{kb_id}",
                "file_name": "索引重建",
                "idempotency_key": f"index-rebuild:{kb_id}:{uuid4().hex}",
                "max_attempts": 3,
            }
        )
        return public_job(job)

    @router.get("/kb/{kb_id}/settings")
    async def get_settings(kb_id: str):
        data = await _read()
        kb = _find_kb(data, kb_id)
        return kb["settings"]

    @router.put("/kb/{kb_id}/settings")
    async def update_settings(kb_id: str, body: dict[str, Any]):
        data = await _read()
        kb = _find_kb(data, kb_id)
        kb["settings"].update(body)
        await _write(data)
        return kb["settings"]

    @router.get("/documents")
    async def list_documents(
        kb_id: str,
        page: int = 1,
        page_size: int = 10,
        search: str = "",
    ):
        data = await _read()
        _find_kb(data, kb_id)
        records = await _list_doc_records(rag)
        items = []
        needle = search.strip().lower()
        for row in records:
            public = _public_doc(row, input_dir)
            if not _doc_in_kb(public, kb_id, data):
                continue
            if needle and needle not in public["name"].lower():
                continue
            items.append(public)
        from lightrag.product_db import public_job

        store = _task_store()
        known_ids = {item["id"] for item in items}
        for doc_id, row in (data.get("doc_index") or {}).items():
            if (
                not isinstance(row, dict)
                or row.get("kb_id") != kb_id
                or row.get("deleted_at")
            ):
                continue
            task = store.latest_for_doc(actor_id(), str(doc_id))
            view = public_job(task) if task else None
            job_status = str((view or {}).get("status") or "")
            stage = str((view or {}).get("stage") or "")
            if job_status == "succeeded" or stage == "ready":
                status = "ready"
            elif job_status in {"failed", "timeout", "cancelled", "consistency_failed"}:
                status = job_status
            elif stage:
                status = stage
            else:
                status = str(row.get("status") or "queued")
            matched = next((item for item in items if item["id"] == doc_id), None)
            payload = {
                "job_id": (view or {}).get("job_id") or "",
                "status": status if status != "checked" else "ready",
                "progress": 100
                if status in {"ready", "checked"}
                else int((view or {}).get("progress") or 0),
                "error_msg": (view or {}).get("error_message") or "",
            }
            if matched:
                matched.update(payload)
                if row.get("display_name"):
                    matched["name"] = row["display_name"]
            elif doc_id not in known_ids and status != "cancelled":
                items.append(
                    {
                        "id": doc_id,
                        "name": row.get("display_name") or doc_id,
                        "chunk_count": len(row.get("chunk_ids") or []),
                        "char_count": 0,
                        "file_size": _file_size(
                            "", row.get("display_name") or "", input_dir
                        ),
                        "updated_at": (view or {}).get("created_at") or _now(),
                        "created_at": (view or {}).get("created_at") or _now(),
                        "file_path": row.get("storage_key") or "",
                        **payload,
                    }
                )
        items.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
        from lightrag.index_manifest import compatibility_error
        from lightrag.product_parse import load_document
        from lightrag.product_source import mime_of

        blocked = compatibility_error(Path(rag.working_dir))
        index_status = "index_rebuild_required" if blocked else "ready"
        visible = []
        for item in items:
            document = load_document(rag.working_dir, item["id"])
            view = {key: value for key, value in item.items() if key != "file_path"}
            view["mime_type"] = mime_of(
                item.get("name") or "", document.source_type if document else ""
            )
            view["version_id"] = document.version_id if document else ""
            view["version"] = view["version_id"] or item.get("version") or ""
            view["unit_count"] = len(document.walk()) if document else 0
            view["index_status"] = index_status
            visible.append(view)
        total = len(visible)
        start = max(page - 1, 0) * page_size
        return {"items": visible[start : start + page_size], "total": total}

    def _owned_source(
        data: dict[str, Any], doc_id: str, kb_id: str, version_id: str = ""
    ):
        from lightrag.product_parse import load_document
        from lightrag.product_source import resolve_owned_file, source_kind

        _find_kb(data, kb_id)
        row = (data.get("doc_index") or {}).get(doc_id)
        if (
            not isinstance(row, dict)
            or row.get("deleted_at")
            or row.get("kb_id") != kb_id
        ):
            raise HTTPException(status_code=404, detail="来源不存在")
        owner = str(row.get("owner_id") or "")
        if owner and owner != actor_id():
            raise HTTPException(status_code=404, detail="来源不存在")
        document = load_document(rag.working_dir, doc_id)
        if version_id and (document is None or document.version_id != version_id):
            raise HTTPException(status_code=404, detail="来源不存在")
        storage = str(row.get("storage_key") or "")
        path = resolve_owned_file(input_dir, storage) if storage else None
        if storage and path is None:
            raise HTTPException(status_code=404, detail="来源不存在")
        name = str(
            row.get("display_name")
            or (document.source_name if document else "")
            or doc_id
        )
        kind = source_kind(name, document.source_type if document else "")
        return row, document, path, name, kind

    @router.get("/documents/{doc_id}/source")
    async def document_source(
        doc_id: str,
        kb_id: str,
        version_id: str = "",
        chunk_id: str = "",
        unit_id: str = "",
    ):
        from lightrag.product_source import build_source_view

        data = await _read()
        _row, document, _path, name, _kind = _owned_source(
            data, doc_id, kb_id, version_id
        )
        payload = build_source_view(
            document, doc_name=name, chunk_id=chunk_id, unit_id=unit_id
        )
        payload["document_id"] = doc_id
        if document:
            payload["version_id"] = document.version_id
        return payload

    @router.get("/documents/{doc_id}/content")
    async def document_content(
        doc_id: str,
        kb_id: str,
        version_id: str = "",
        chunk_id: str = "",
        unit_id: str = "",
    ):
        from lightrag.product_source import build_reading, build_source_view

        data = await _read()
        _row, document, path, name, kind = _owned_source(
            data, doc_id, kb_id, version_id
        )
        file_text = ""
        if path is not None and kind in {"text", "markdown"}:
            file_text = path.read_text(encoding="utf-8", errors="replace")
        payload = build_reading(document, kind=kind, file_text=file_text)
        located = build_source_view(
            document, doc_name=name, chunk_id=chunk_id, unit_id=unit_id
        )
        payload.update(
            {
                "document_id": doc_id,
                "version_id": document.version_id if document else "",
                "doc_name": name,
                "unit": located["unit"],
                "matched_chunk": located["matched_chunk"],
            }
        )
        return payload

    def _file_response(path: Path, name: str, inline: bool):
        from fastapi.responses import FileResponse
        from lightrag.product_source import content_disposition, mime_of

        return FileResponse(
            path,
            media_type=mime_of(name),
            headers={"Content-Disposition": content_disposition(name, inline=inline)},
        )

    @router.get("/documents/{doc_id}/download")
    async def document_download(doc_id: str, kb_id: str, version_id: str = ""):
        data = await _read()
        _row, _document, path, name, _kind = _owned_source(
            data, doc_id, kb_id, version_id
        )
        if path is None:
            raise HTTPException(status_code=404, detail="来源不存在")
        return _file_response(path, name, inline=False)

    @router.get("/documents/{doc_id}/preview")
    async def document_preview(doc_id: str, kb_id: str, version_id: str = ""):
        data = await _read()
        _row, _document, path, name, kind = _owned_source(
            data, doc_id, kb_id, version_id
        )
        if path is None:
            raise HTTPException(status_code=404, detail="来源不存在")
        if kind not in {"pdf", "text", "markdown"}:
            raise HTTPException(status_code=415, detail="该格式暂不提供原件预览")
        return _file_response(path, name, inline=True)

    @router.post("/documents")
    async def upload_document(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        kb_id: str = Form(...),
        chunk_strategy: str = Form("smart"),
        chunk_size: int = Form(512),
        chunk_overlap: int = Form(64),
        chunk_delimiter: str = Form("\\n\\n"),
    ):
        data = await _read()
        _find_kb(data, kb_id)
        safe_name = sanitize_filename(file.filename or "upload.bin", input_dir)
        if not doc_manager.is_supported_file(safe_name):
            raise HTTPException(
                status_code=400, detail=f"不支持的文件类型：{safe_name}"
            )
        lock = _UPLOAD_LOCKS.setdefault(f"{kb_id}:{safe_name}", asyncio.Lock())
        async with lock:
            return await _store_upload(
                background_tasks,
                file,
                kb_id,
                safe_name,
                data,
                chunk_strategy,
                chunk_size,
                chunk_overlap,
                chunk_delimiter,
            )

    async def _store_upload(
        background_tasks,
        file,
        kb_id,
        safe_name,
        data,
        chunk_strategy,
        chunk_size,
        chunk_overlap,
        chunk_delimiter,
    ):
        from lightrag.product_db import public_job
        from lightrag.product_ingest import content_key
        from lightrag.product_uploads import allocate_upload

        fresh = await _read()
        _find_kb(fresh, kb_id)
        data = fresh
        _max_bytes, max_files = upload_limits()
        owned_files = sum(
            1
            for item in (data.get("doc_index") or {}).values()
            if isinstance(item, dict)
            and item.get("kb_id") == kb_id
            and not item.get("deleted_at")
        )
        owned_files += sum(
            1 for bound in data["file_bindings"].values() if bound == kb_id
        )
        if owned_files >= max_files:
            raise HTTPException(status_code=400, detail="该知识库文件数量已达上限")
        payload = await file.read()
        try:
            inspect_upload(safe_name, payload)
        except AccountError as exc:
            status = 413 if exc.code == "too_large" else 400
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        idempotency = content_key(actor_id(), kb_id, payload)
        store = _task_store()
        existing = store.find_idempotent(idempotency)
        if existing:
            view = public_job(existing)
            view["message"] = "相同文件已有任务"
            return view
        try:
            allocated = allocate_upload(data, actor_id(), kb_id, safe_name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        allocated["status"] = "queued"
        dest = input_dir / allocated["storage_key"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        await _write(data)
        allowed = {
            "smart",
            "delimiter",
            "fixed",
            "recursive",
            "ledger",
            "qa",
            "clause",
            "one",
        }
        strategy = chunk_strategy if chunk_strategy in allowed else "smart"
        job = store.create_job(
            {
                "job_type": "ingestion",
                "user_id": actor_id(),
                "kb_id": kb_id,
                "doc_id": allocated["doc_id"],
                "file_name": safe_name,
                "file_path": allocated["storage_key"],
                "idempotency_key": idempotency,
                "input_snapshot": {
                    "owner_id": allocated["owner_id"],
                    "storage_key": allocated["storage_key"],
                    "content_hash": idempotency.rsplit(":", 1)[-1],
                    "strategy": strategy,
                    "chunk_size": max(1, int(chunk_size or 512)),
                    "chunk_overlap": max(0, int(chunk_overlap or 0)),
                    "version": allocated["version"],
                },
            }
        )
        if job.get("doc_id") != allocated["doc_id"]:
            fresh = await _read()
            fresh.get("doc_index", {}).pop(allocated["doc_id"], None)
            await _write(fresh)
            if dest.is_file():
                dest.unlink()
        view = public_job(job)
        view["message"] = "文档已提交，正在排队处理"
        logger.info(
            "产品壳上传 %s -> kb=%s job=%s", safe_name, kb_id, view.get("job_id")
        )
        return view

    @router.get("/documents/{doc_id}/status")
    async def document_status(doc_id: str):
        status = await rag.doc_status.get_by_id(doc_id)
        if status is None:
            raise HTTPException(status_code=404, detail="文档不存在")
        public = _public_doc({"id": doc_id, "doc": status}, input_dir)
        data = await _read()
        bound = (data.get("file_bindings") or {}).get(public["name"])
        _find_kb(data, bound or DEFAULT_KB_ID)
        return public

    @router.post("/documents/cancel")
    async def cancel_upload(kb_id: str, name: str = "", doc_id: str = ""):
        from lightrag.ingest_progress import cancel
        from lightrag.product_uploads import abort_upload

        data = await _read()
        _find_kb(data, kb_id)
        target = None
        for item in (data.get("doc_index") or {}).values():
            if (
                not isinstance(item, dict)
                or item.get("kb_id") != kb_id
                or item.get("deleted_at")
            ):
                continue
            if doc_id and item.get("doc_id") == doc_id:
                target = item
            elif name and item.get("display_name") == Path(name).name:
                target = item
        if target is None:
            raise HTTPException(status_code=404, detail="文档不存在")
        task = _task_store().latest_for_doc(actor_id(), str(target["doc_id"]))
        if task:
            _task_store().request_cancel(task["job_id"], actor_id())
        cancel(str(target.get("doc_id") or ""))
        abort_upload(
            data,
            Path(rag.working_dir),
            input_dir,
            str(target["doc_id"]),
            "cancel",
            cancelled=True,
        )
        await _write(data)
        view = (
            _public_task(task["job_id"])
            if task
            else {"status": "cancelled", "doc_id": target["doc_id"]}
        )
        view["name"] = target.get("display_name") or name
        return view

    @router.delete("/documents/{doc_id}")
    async def delete_document(
        doc_id: str, kb_id: str, background_tasks: BackgroundTasks
    ):
        data = await _read()
        _find_kb(data, kb_id)
        index = data.setdefault("doc_index", {})
        row = index.get(doc_id)
        records = await _list_doc_records(rag)
        target = next((item for item in records if item["id"] == doc_id), None)
        display_name = ""
        if target is not None:
            public = _public_doc(target, input_dir)
            if not _doc_in_kb(public, kb_id, data) and not (
                isinstance(row, dict) and row.get("kb_id") == kb_id
            ):
                raise HTTPException(status_code=404, detail="文档不存在")
            display_name = public["name"]
            data["file_bindings"].pop(public["name"], None)
        elif not isinstance(row, dict) or row.get("kb_id") != kb_id:
            raise HTTPException(status_code=404, detail="文档不存在")
        else:
            display_name = str(row.get("display_name") or doc_id)
        if not isinstance(row, dict):
            row = {
                "doc_id": doc_id,
                "kb_id": kb_id,
                "owner_id": actor_id(),
                "display_name": display_name,
            }
            index[doc_id] = row
        row["status"] = "deleting"
        await _write(data)
        created = _task_store().create_job(
            {
                "job_type": "delete_document",
                "user_id": actor_id(),
                "kb_id": kb_id,
                "doc_id": doc_id,
                "file_name": display_name,
                "file_path": str(row.get("storage_key") or ""),
                "idempotency_key": f"delete-document:{kb_id}:{doc_id}",
            }
        )
        from lightrag.product_db import public_job

        return public_job(created)

    @router.get("/chunks")
    async def list_chunks(
        kb_id: str, page: int = 1, page_size: int = 10, search: str = ""
    ):
        data = await _read()
        _find_kb(data, kb_id)
        items = await _kb_chunks(kb_id, data)
        needle = search.strip().lower()
        if needle:
            items = [
                item
                for item in items
                if needle in item["doc_name"].lower()
                or needle in (item["content"] or "").lower()
            ]
        total = len(items)
        start = max(page - 1, 0) * page_size
        page_items = []
        for item in items[start : start + page_size]:
            page_items.append({**item, "content": _clean_text(item["content"], 280)})
        return {"items": page_items, "total": total}

    @router.get("/qa")
    async def list_qa(kb_id: str, page: int = 1, page_size: int = 10, search: str = ""):
        data = await _read()
        _find_kb(data, kb_id)
        items = [
            item for item in data.get("qa_pairs") or [] if item.get("kb_id") == kb_id
        ]
        needle = search.strip().lower()
        if needle:
            items = [
                item
                for item in items
                if needle in (item.get("question") or "").lower()
                or needle in (item.get("answer") or "").lower()
            ]
        total = len(items)
        start = max(page - 1, 0) * page_size
        return {"items": items[start : start + page_size], "total": total}

    @router.post("/qa")
    async def create_qa(body: QABody):
        question = body.question.strip()
        answer = body.answer.strip()
        if not question or not answer:
            raise HTTPException(status_code=400, detail="问题和答案都不能为空")
        data = await _read()
        _find_kb(data, body.kb_id)
        item = {
            "id": uuid4().hex,
            "kb_id": body.kb_id,
            "question": question,
            "answer": answer,
            "doc_id": body.doc_id or "",
            "doc_name": body.doc_name or "",
            "source": "manual",
            "updated_at": _now(),
        }
        data.setdefault("qa_pairs", []).insert(0, item)
        await _write(data)
        return item

    @router.delete("/qa/{qa_id}")
    async def delete_qa(qa_id: str, kb_id: str):
        data = await _read()
        before = len(data.get("qa_pairs") or [])
        data["qa_pairs"] = [
            item
            for item in data.get("qa_pairs") or []
            if not (item.get("id") == qa_id and item.get("kb_id") == kb_id)
        ]
        if len(data["qa_pairs"]) == before:
            raise HTTPException(status_code=404, detail="问答不存在")
        await _write(data)
        return {"message": "已删除"}

    async def _retrieve(
        body_query: str,
        kb: dict[str, Any],
        mode_name: str,
        top_k: int,
        rerank: bool,
        history: list | None,
    ):
        mode = _map_mode(mode_name, bool(kb["settings"].get("graph_enabled", True)))
        history_msgs = [
            {"role": item.get("role"), "content": item.get("content", "")}
            for item in (history or [])
            if item.get("role") in {"user", "assistant"} and item.get("content")
        ]
        param = QueryParam(
            mode=mode,
            top_k=top_k,
            chunk_top_k=top_k,
            enable_rerank=rerank,
            conversation_history=history_msgs,
            history_turns=min(3, len(history_msgs) // 2) if history_msgs else 0,
        )
        return mode, await rag.aquery_data(body_query, param=param)

    @router.post("/chat/search-test")
    async def search_test(body: SearchBody):
        data = await _read()
        kb = _find_kb(data, body.kb_id)
        settings = kb["settings"]
        top_k = body.top_k or settings.get("top_k") or 10
        ratio = body.similarity_ratio
        if ratio is None:
            ratio = settings.get("similarity_ratio", 0.5)
        rerank = (
            body.enable_rerank
            if body.enable_rerank is not None
            else bool(settings.get("rerank_enabled"))
        )
        mode_name = body.retrieval_mode or settings.get("retrieval_mode") or "mix"
        if not bool(settings.get("graph_enabled", True)) and mode_name in {
            "mix",
            "graph",
        }:
            mode_name = "vector"
        threshold = body.similarity_threshold
        if threshold is None:
            threshold = settings.get("similarity_threshold", 0.2)
        bindings = data.get("file_bindings") or {}

        def file_in_kb(file_path: str, record: dict | None = None) -> bool:
            payload = dict(record or {})
            if file_path and not payload.get("file_path"):
                payload["file_path"] = file_path
            return scope_visible(
                payload, body.kb_id, bindings, data.get("doc_index") or {}
            )

        try:
            kb_chunks = await _kb_chunks(body.kb_id, data)
            return await run_search_test(
                rag,
                query=body.query.strip(),
                mode=mode_name,
                top_k=int(top_k),
                ratio=float(ratio),
                enable_rerank=bool(rerank),
                kb_chunks=kb_chunks,
                file_in_kb=file_in_kb,
                score_threshold=float(threshold or 0),
                kb_id=body.kb_id,
                owner_id=str(kb.get("owner_id") or ""),
                working_dir=Path(rag.working_dir),
                staged=True,
                history=list(body.history or []),
            )
        except Exception as exc:
            raise _public_error(exc) from exc

    def _cache_path() -> Path:
        return Path(rag.working_dir) / "semantic_cache.json"

    def _load_cache() -> list[dict[str, Any]]:
        path = _cache_path()
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return data if isinstance(data, list) else []

    def _save_cache(entries: list[dict[str, Any]]) -> None:
        _cache_path().write_text(
            json.dumps(entries[-100:], ensure_ascii=False),
            encoding="utf-8",
        )

    async def _kb_version(kb_id: str, data: dict[str, Any]) -> str:
        records = await _list_doc_records(rag)
        parts = []
        for row in records:
            public = _public_doc(row, input_dir)
            if _doc_in_kb(public, kb_id, data):
                parts.append(
                    f"{public['id']}:{public.get('updated_at')}:{public.get('chunk_count')}"
                )
        return "|".join(sorted(parts))

    @router.get("/conversations")
    async def list_conversations(kb_id: str, q: str = "", archived: bool = False):
        data = await _read()
        _find_kb(data, kb_id)
        items = _app().list_conversations(actor_id(), kb_id, q=q, archived=archived)
        return {"items": items}

    @router.post("/conversations")
    async def create_conversation(body: ConversationCreate):
        data = await _read()
        _find_kb(data, body.kb_id)
        return _app().create_conversation(
            actor_id(), body.kb_id, body.title or "新会话"
        )

    @router.get("/conversations/{conversation_id}")
    async def get_conversation(conversation_id: str, kb_id: str):
        data = await _read()
        _find_kb(data, kb_id)
        item = _app().get_conversation(conversation_id, actor_id(), kb_id)
        if item is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        return item

    @router.patch("/conversations/{conversation_id}")
    async def patch_conversation(conversation_id: str, body: ConversationPatch):
        data = await _read()
        _find_kb(data, body.kb_id)
        item = _app().update_conversation(
            conversation_id,
            actor_id(),
            body.kb_id,
            title=body.title,
            summary=body.summary,
            is_archived=body.is_archived,
        )
        if item is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        return item

    @router.delete("/conversations/{conversation_id}")
    async def delete_conversation(conversation_id: str, kb_id: str):
        data = await _read()
        _find_kb(data, kb_id)
        if not _app().delete_conversation(conversation_id, actor_id(), kb_id):
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"ok": True}

    @router.post("/conversations/{conversation_id}/pin")
    async def pin_conversation(conversation_id: str, kb_id: str):
        data = await _read()
        _find_kb(data, kb_id)
        store = _app()
        current = store.get_conversation(conversation_id, actor_id(), kb_id)
        if current is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        item = store.update_conversation(
            conversation_id, actor_id(), kb_id, is_pinned=not current["is_pinned"]
        )
        return item

    @router.post("/conversations/{conversation_id}/archive")
    async def archive_conversation(conversation_id: str, kb_id: str):
        data = await _read()
        _find_kb(data, kb_id)
        item = _app().update_conversation(
            conversation_id, actor_id(), kb_id, is_archived=True
        )
        if item is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        return item

    @router.get("/conversations/{conversation_id}/messages")
    async def list_conversation_messages(conversation_id: str, kb_id: str):
        data = await _read()
        _find_kb(data, kb_id)
        store = _app()
        if store.get_conversation(conversation_id, actor_id(), kb_id) is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"items": store.list_messages(conversation_id, actor_id(), kb_id)}

    @router.post("/messages/{message_id}/feedback")
    async def message_feedback(message_id: str, body: FeedbackBody):
        data = await _read()
        _find_kb(data, body.kb_id)
        rating = (body.rating or "").strip()
        if rating not in {"positive", "negative"}:
            raise HTTPException(status_code=400, detail="反馈类型无效")
        reason = (body.reason or "").strip()
        allowed = {
            "answer_wrong",
            "citation_wrong",
            "not_found",
            "outdated",
            "unclear",
            "other",
        }
        if rating == "negative" and reason not in allowed:
            raise HTTPException(status_code=400, detail="请选择反馈原因")
        store = _app()
        message_row = store.get_message(message_id, actor_id(), body.kb_id)
        if message_row is None or message_row["role"] != "assistant":
            raise HTTPException(status_code=404, detail="消息不存在")
        question = ""
        for item in store.list_messages(
            message_row["conversation_id"], actor_id(), body.kb_id
        ):
            if item["id"] == message_id:
                break
            if item["role"] == "user":
                question = item["content"]
        return store.add_feedback(
            message_id=message_id,
            owner_id=actor_id(),
            kb_id=body.kb_id,
            rating=rating,
            reason=reason if rating == "negative" else "",
            comment=(body.comment or "")[:500],
            question=question,
            answer=message_row["content"],
            citations=list(message_row.get("citations") or []),
            index_version=str(message_row.get("index_version") or ""),
            embedding_model=str(message_row.get("embedding_model") or ""),
            rerank_model=str(message_row.get("rerank_model") or ""),
            llm_model=str(message_row.get("llm_model") or ""),
        )

    @router.get("/memory-candidates")
    async def list_memory_candidates(kb_id: str, status: str = "pending"):
        data = await _read()
        _find_kb(data, kb_id)
        items = _app().list_memory_candidates(
            actor_id(), kb_id, status=status or "pending"
        )
        return {"items": items}

    @router.post("/memory-candidates")
    async def create_memory_candidate(body: MemoryCandidateBody):
        data = await _read()
        _find_kb(data, body.kb_id)
        content = (body.content or "").strip()
        if not content:
            raise HTTPException(status_code=400, detail="记忆内容不能为空")
        return _app().add_memory_candidate(
            owner_id=actor_id(),
            kb_id=body.kb_id,
            content=content[:800],
            category=(body.category or "other")[:32],
            conversation_id=body.conversation_id or "",
            message_id=body.message_id or "",
            status="pending",
        )

    @router.post("/memory-candidates/{candidate_id}/accept")
    async def accept_memory_candidate(candidate_id: str, kb_id: str):
        data = await _read()
        _find_kb(data, kb_id)
        candidate = _app().get_memory_candidate(candidate_id, actor_id(), kb_id)
        embedding = (
            await _embedding_for(rag, str((candidate or {}).get("content") or ""))
            if candidate
            else []
        )
        item = _app().accept_memory_candidate(
            candidate_id,
            actor_id(),
            kb_id,
            embedding=embedding,
            embedding_model=os.getenv("EMBEDDING_MODEL") or "",
        )
        if item is None:
            raise HTTPException(status_code=404, detail="记忆候选不存在或已处理")
        return item

    @router.post("/memory-candidates/{candidate_id}/reject")
    async def reject_memory_candidate(candidate_id: str, kb_id: str):
        data = await _read()
        _find_kb(data, kb_id)
        item = _app().set_candidate_status(candidate_id, actor_id(), "rejected", kb_id)
        if item is None:
            raise HTTPException(status_code=404, detail="记忆候选不存在")
        return item

    @router.get("/memories")
    async def list_memories(kb_id: str | None = None):
        if kb_id:
            data = await _read()
            _find_kb(data, kb_id)
        return {"items": _app().list_memories(actor_id(), kb_id)}

    @router.patch("/memories/{memory_id}")
    async def patch_memory(memory_id: str, body: MemoryPatch):
        if body.scope is not None and body.scope not in {"global", "kb"}:
            raise HTTPException(status_code=400, detail="记忆作用域只能是 global 或 kb")
        if body.scope == "kb":
            if not body.kb_id:
                raise HTTPException(status_code=400, detail="知识库记忆必须提供 kb_id")
            data = await _read()
            _find_kb(data, body.kb_id)
        item = _app().update_memory(
            memory_id,
            actor_id(),
            content=body.content,
            category=body.category,
            enabled=body.enabled,
            scope=body.scope,
            kb_id=body.kb_id,
            expires_at=body.expires_at,
        )
        if item is None:
            raise HTTPException(status_code=404, detail="记忆不存在")
        return item

    @router.delete("/memories/{memory_id}")
    async def delete_memory(memory_id: str):
        if not _app().delete_memory(memory_id, actor_id()):
            raise HTTPException(status_code=404, detail="记忆不存在")
        return {"ok": True}

    def _voice_http_error(exc: Exception) -> HTTPException:
        from lightrag.product_voice import VoiceProviderError

        if isinstance(exc, VoiceProviderError):
            return HTTPException(status_code=exc.status_code, detail=str(exc))
        return HTTPException(status_code=503, detail="语音服务暂时不可用")

    @router.post("/voice/practice/sessions")
    async def create_voice_practice_session(body: VoiceSessionCreate):
        data = await _read()
        _find_kb(data, body.kb_id)
        user_id = actor_id()
        app = _app()
        conversation = app.create_conversation(user_id, body.kb_id, "语音练习")
        return app.create_voice_practice_session(
            user_id,
            body.kb_id,
            goal=body.goal,
            conversation_id=conversation["id"],
        )

    @router.get("/voice/practice/sessions")
    async def list_voice_practice_sessions(kb_id: str, limit: int = 30):
        data = await _read()
        _find_kb(data, kb_id)
        return {
            "items": _app().list_voice_practice_sessions(actor_id(), kb_id, limit=limit)
        }

    @router.get("/voice/practice/sessions/{session_id}")
    async def get_voice_practice_session(session_id: str, kb_id: str):
        data = await _read()
        _find_kb(data, kb_id)
        app = _app()
        session = app.get_voice_practice_session(session_id, actor_id(), kb_id)
        if session is None:
            raise HTTPException(status_code=404, detail="练习不存在")
        session["turns"] = app.list_voice_practice_turns(session_id, actor_id(), kb_id)
        return session

    @router.post("/voice/transcribe")
    async def transcribe_voice(file: UploadFile = File(...)):
        from lightrag.product_voice import transcribe_audio

        try:
            audio = await file.read()
            text = await transcribe_audio(
                audio, file.filename or "recording.webm", file.content_type or ""
            )
            return {"text": text}
        except Exception as exc:
            raise _voice_http_error(exc) from exc

    @router.post("/voice/practice/{session_id}/turn")
    async def voice_practice_turn(session_id: str, body: VoiceTurnBody):
        transcript = (body.transcript or "").strip()
        if not transcript:
            raise HTTPException(status_code=400, detail="请先确认转写文本")
        data = await _read()
        session = _app().get_voice_practice_session(session_id, actor_id(), None)
        if session is None:
            raise HTTPException(status_code=404, detail="练习不存在")
        if body.kb_id and body.kb_id != session["kb_id"]:
            raise HTTPException(status_code=404, detail="练习不存在")
        _find_kb(data, session["kb_id"])
        if session["status"] != "active":
            raise HTTPException(status_code=409, detail="练习已经结束")
        body_for_chat = ChatBody(
            query=transcript,
            kb_id=session["kb_id"],
            conversation_id=session.get("conversation_id") or None,
            stream=False,
        )
        answer = ""
        citations: list[dict[str, Any]] = []
        memories: list[dict[str, Any]] = []
        try:
            async for event in _answer_events(body_for_chat):
                if event.get("type") == "meta":
                    citations = list(event.get("citations") or [])
                elif event.get("type") == "done":
                    answer = str(event.get("answer") or answer)
                    citations = list(event.get("citations") or citations)
                    memories = list(event.get("memories") or [])
                elif event.get("type") == "error":
                    raise HTTPException(
                        status_code=503,
                        detail=str(event.get("message") or "问答服务暂时不可用"),
                    )
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("语音练习回合失败")
            _app().add_voice_practice_turn(
                session_id=session_id,
                owner_id=actor_id(),
                kb_id=session["kb_id"],
                transcript=transcript,
                status="failed",
            )
            raise HTTPException(status_code=503, detail="本轮回答失败，请重试") from exc
        app = _app()
        turn = app.add_voice_practice_turn(
            session_id=session_id,
            owner_id=actor_id(),
            kb_id=session["kb_id"],
            transcript=transcript,
            answer=answer,
            citations=citations,
            memory_refs=memories,
            status="completed" if answer else "failed",
        )
        return {
            "turn_id": turn["id"],
            "session_id": session_id,
            "conversation_id": session.get("conversation_id") or "",
            "transcript": transcript,
            "answer": answer,
            "citations": citations,
            "memories": memories,
            "audio_status": turn["audio_status"],
        }

    @router.post("/voice/speech")
    async def voice_speech(body: VoiceSpeechBody):
        from lightrag.product_voice import synthesize_speech

        app = _app()
        found = app.get_voice_practice_turn(body.turn_id, actor_id())
        if (
            found is None
            or found.get("status") != "completed"
            or (body.kb_id and found.get("kb_id") != body.kb_id)
        ):
            raise HTTPException(status_code=404, detail="练习回答不存在")
        app.update_voice_practice_turn_audio(body.turn_id, actor_id(), "generating")
        try:
            audio, media_type = await synthesize_speech(
                found.get("answer") or "", body.voice
            )
        except Exception as exc:
            app.update_voice_practice_turn_audio(body.turn_id, actor_id(), "failed")
            raise _voice_http_error(exc) from exc
        app.update_voice_practice_turn_audio(body.turn_id, actor_id(), "ready")
        return Response(
            content=audio, media_type=media_type, headers={"Cache-Control": "no-store"}
        )

    @router.websocket("/voice/realtime")
    async def realtime_voice(websocket: WebSocket):
        """Browser voice session: PCM frames in, ASR/answer/audio events out."""
        from lightrag.product_realtime_voice import (
            AliyunNlsConfig,
            RealtimeVoiceError,
            close_aliyun_socket,
            open_aliyun_transcriber,
            parse_transcriber_event,
            stop_aliyun_transcriber,
        )
        from lightrag.product_voice import stream_speech

        await websocket.accept()
        user_token = None
        provider = None
        provider_task: asyncio.Task | None = None
        provider_state: dict[str, Any] = {}
        answer_task: asyncio.Task | None = None
        tts_task: asyncio.Task | None = None
        session_id = ""
        kb_id = ""
        session: dict[str, Any] = {}

        def websocket_user(data: dict[str, Any]) -> dict[str, Any]:
            if not product_auth_enabled():
                return local_owner_view()
            raw = ""
            authorization = websocket.headers.get("authorization") or ""
            if authorization.lower().startswith("bearer "):
                raw = authorization[7:].strip()
            if not raw:
                raw = websocket.query_params.get("access_token") or ""
            if not raw:
                for protocol in websocket.scope.get("subprotocols") or []:
                    if protocol.startswith("kb-access."):
                        raw = protocol[len("kb-access.") :]
                        break
            if not raw:
                raise HTTPException(status_code=401, detail="未登录")
            try:
                payload = decode_access_payload(raw)
            except Exception as exc:
                raise HTTPException(status_code=401, detail="未登录") from exc
            user = access_is_current(data, payload)
            if user is None:
                raise HTTPException(status_code=401, detail="未登录")
            return public_user(user)

        async def send(event: dict[str, Any]) -> bool:
            if (
                websocket.client_state != WebSocketState.CONNECTED
                or websocket.application_state != WebSocketState.CONNECTED
            ):
                return False
            try:
                await websocket.send_text(json.dumps(event, ensure_ascii=False))
                return True
            except (WebSocketDisconnect, RuntimeError):
                return False

        async def stop_provider() -> dict[str, Any]:
            nonlocal provider, provider_task, provider_state
            current = provider
            task = provider_task
            provider = None
            provider_task = None
            if current is not None:
                await stop_aliyun_transcriber(current[0], current[1], current[2])
            if task is not None:
                with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=2.5)
            if current is not None:
                await close_aliyun_socket(current[0])
            state = dict(provider_state)
            provider_state = {}
            return state

        async def provider_reader(socket: Any, state: dict[str, Any]) -> None:
            try:
                async for raw in socket:
                    event = parse_transcriber_event(raw)
                    if not event:
                        continue
                    if event.get("type") == "transcript":
                        state["text"] = str(event.get("text") or "")
                        if event.get("final"):
                            state["final"] = state["text"]
                        await send(event)
                    elif event.get("type") == "provider_ready":
                        await send({"type": "asr_ready"})
                    elif event.get("type") == "provider_error":
                        state["error"] = event.get("message") or "阿里云实时识别失败"
                        state["provider_code"] = event.get("provider_code") or ""
                        await send(
                            {
                                "type": "error",
                                "stage": "asr",
                                "message": state["error"],
                                "provider_code": state["provider_code"],
                            }
                        )
                        break
                    elif event.get("type") == "provider_done":
                        state["completed"] = True
                        break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                state["error"] = "阿里云实时识别连接已断开"
                logger.warning("实时 ASR 连接断开: %s", exc)
                with contextlib.suppress(Exception):
                    await send(
                        {"type": "error", "stage": "asr", "message": state["error"]}
                    )

        async def start_provider() -> None:
            nonlocal provider, provider_task, provider_state
            config = AliyunNlsConfig.from_env()
            provider = await open_aliyun_transcriber(config)
            provider_state = {"text": "", "final": "", "ready": True}
            provider_task = asyncio.create_task(
                provider_reader(provider[0], provider_state)
            )
            await send({"type": "asr_ready"})

        async def cancel_answer(reason: str = "cancelled") -> None:
            nonlocal answer_task, tts_task
            for task in (answer_task, tts_task):
                if task and not task.done():
                    task.cancel()
            for task in (answer_task, tts_task):
                if task:
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task
            answer_task = None
            tts_task = None
            with contextlib.suppress(Exception):
                await send({"type": "answer_cancelled", "reason": reason})

        async def speak_local_reply(text: str) -> None:
            await send(
                {
                    "type": "answer_done",
                    "answer": text,
                    "citations": [],
                    "memories": [],
                    "answerable": True,
                }
            )
            try:
                await send({"type": "tts_start", "purpose": "local_reply"})
                async for chunk in stream_speech(text):
                    await websocket.send_bytes(chunk)
                await send({"type": "tts_end", "purpose": "local_reply"})
            except Exception as exc:
                await send({"type": "error", "stage": "tts", "message": str(exc)[:240]})

        async def answer_turn(transcript: str) -> None:
            nonlocal tts_task
            body_for_chat = ChatBody(
                query=transcript,
                kb_id=kb_id,
                conversation_id=session.get("conversation_id") or None,
                stream=True,
                # Generate the structured answer first so citations are validated
                # before any sentence is spoken. The validated answer is still
                # emitted and synthesized incrementally below.
                voice_stream=False,
            )
            tts_queue: asyncio.Queue[str | None] = asyncio.Queue()
            tts_error_sent = False
            tts_started = False

            async def tts_worker() -> None:
                nonlocal tts_error_sent, tts_started
                while True:
                    text = await tts_queue.get()
                    try:
                        if text is None:
                            if tts_started:
                                await send({"type": "tts_end"})
                            return
                        if not tts_started:
                            tts_started = True
                            await send({"type": "tts_start"})
                        async for chunk in stream_speech(text):
                            await websocket.send_bytes(chunk)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        if not tts_error_sent:
                            tts_error_sent = True
                            await send(
                                {
                                    "type": "error",
                                    "stage": "tts",
                                    "message": str(exc)[:240],
                                }
                            )
                    finally:
                        tts_queue.task_done()

            tts_task = asyncio.create_task(tts_worker())
            await send(
                {
                    "type": "answer_status",
                    "phase": "retrieving",
                    "message": "正在查找相关资料",
                }
            )
            await tts_queue.put(_voice_thinking_ack(transcript))
            sentence_buffer = ""
            answer_text = ""
            citations: list[dict[str, Any]] = []
            memories: list[dict[str, Any]] = []
            try:
                async for event in _answer_events(body_for_chat):
                    event_type = event.get("type")
                    if event_type == "meta":
                        citations = list(event.get("citations") or [])
                        await send(
                            {
                                "type": "answer_status",
                                "phase": "composing",
                                "message": "已找到相关资料，正在组织回答",
                            }
                        )
                    elif event_type == "token":
                        text = str(event.get("text") or "")
                        answer_text += text
                        await send(
                            {
                                "type": "answer_token",
                                "text": text,
                                "draft": bool(event.get("draft")),
                            }
                        )
                        sentence_buffer += text
                        while True:
                            match = re.search(r"[。！？!?；;]\s*", sentence_buffer)
                            if not match:
                                break
                            segment = sentence_buffer[: match.end()].strip()
                            sentence_buffer = sentence_buffer[match.end() :]
                            if segment:
                                await tts_queue.put(segment)
                    elif event_type == "done":
                        answer_text = str(event.get("answer") or answer_text)
                        citations = list(event.get("citations") or citations)
                        memories = list(event.get("memories") or [])
                        if sentence_buffer.strip():
                            await tts_queue.put(sentence_buffer.strip())
                            sentence_buffer = ""
                        await send(
                            {
                                "type": "answer_done",
                                "answer": answer_text,
                                "citations": citations,
                                "memories": memories,
                                "answerable": event.get("answerable"),
                            }
                        )
                        break
                    elif event_type == "error":
                        await send(
                            {
                                "type": "error",
                                "stage": "answer",
                                "message": str(
                                    event.get("message") or "问答服务暂时不可用"
                                ),
                            }
                        )
                        break
                await tts_queue.join()
                await tts_queue.put(None)
                await tts_task
                app = _app()
                turn = app.add_voice_practice_turn(
                    session_id=session_id,
                    owner_id=actor_id(),
                    kb_id=kb_id,
                    transcript=transcript,
                    answer=answer_text,
                    citations=citations,
                    memory_refs=memories,
                    status="completed" if answer_text else "failed",
                )
                await send(
                    {
                        "type": "turn_done",
                        "transcript": transcript,
                        "turn": turn,
                        "conversation_id": session.get("conversation_id") or "",
                    }
                )
            except asyncio.CancelledError:
                with contextlib.suppress(Exception):
                    _app().add_voice_practice_turn(
                        session_id=session_id,
                        owner_id=actor_id(),
                        kb_id=kb_id,
                        transcript=transcript,
                        status="cancelled",
                    )
                raise
            finally:
                if tts_task and not tts_task.done():
                    tts_task.cancel()

        try:
            data = await _read()
            user = websocket_user(data)
            user_token = CURRENT_USER.set(user)
            first = await websocket.receive_json()
            if (
                first.get("type") != "start"
                or not str(first.get("kb_id") or "").strip()
            ):
                await send({"type": "error", "message": "实时会话缺少知识库"})
                return
            kb_id = str(first["kb_id"])
            _find_kb(data, kb_id)
            session_id = str(first.get("session_id") or "")
            app = _app()
            if session_id:
                session = (
                    app.get_voice_practice_session(session_id, actor_id(), kb_id) or {}
                )
            if not session:
                conversation = app.create_conversation(
                    actor_id(), kb_id, "实时语音练习"
                )
                session = app.create_voice_practice_session(
                    actor_id(),
                    kb_id,
                    goal=str(first.get("goal") or "free"),
                    conversation_id=conversation["id"],
                )
                session_id = str(session.get("id") or "")
            if session.get("status") != "active":
                await send({"type": "error", "message": "练习会话已经结束"})
                return
            try:
                await start_provider()
            except RealtimeVoiceError as exc:
                await send(
                    {
                        "type": "error",
                        "stage": "asr",
                        "code": exc.code,
                        "message": str(exc),
                        "retryable": exc.code
                        not in {
                            "aliyun_not_configured",
                            "aliyun_permission_denied",
                            "aliyun_access_key_invalid",
                            "realtime_dependency_missing",
                        },
                    }
                )
                return
            await send(
                {
                    "type": "ready",
                    "session_id": session_id,
                    "kb_id": kb_id,
                    "conversation_id": session.get("conversation_id") or "",
                    "transport": "websocket",
                    "continuous": True,
                }
            )

            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    if provider is None:
                        continue
                    try:
                        await provider[0].send(message["bytes"])
                    except Exception:
                        await send(
                            {
                                "type": "error",
                                "stage": "asr",
                                "message": "实时识别连接已断开，请重新连接",
                            }
                        )
                    continue
                raw_text = message.get("text") or ""
                try:
                    command = json.loads(raw_text)
                except ValueError:
                    continue
                command_type = command.get("type")
                if command_type in {"barge_in", "cancel"}:
                    await cancel_answer(command_type)
                    continue
                if command_type == "pause":
                    await stop_provider()
                    await send({"type": "paused"})
                    continue
                if command_type == "resume":
                    await start_provider()
                    await send({"type": "resumed"})
                    continue
                if command_type == "end_turn":
                    state = await stop_provider()
                    transcript = str(
                        state.get("final")
                        or state.get("text")
                        or command.get("text")
                        or ""
                    ).strip()
                    if not transcript:
                        await send(
                            {
                                "type": "error",
                                "stage": "asr",
                                "message": "没有识别到可用语音内容",
                            }
                        )
                        await start_provider()
                        continue
                    await send({"type": "transcript_final", "text": transcript})
                    local_intent = _voice_local_intent(transcript)
                    if local_intent is not None:
                        intent, reply = local_intent
                        if answer_task and not answer_task.done():
                            await cancel_answer("voice_command")
                        await send(
                            {
                                "type": "answer_status",
                                "phase": "ending"
                                if intent == "finish"
                                else "responding",
                                "message": "正在结束本次对话"
                                if intent == "finish"
                                else "正在回应",
                            }
                        )
                        await speak_local_reply(reply)
                        turn = _app().add_voice_practice_turn(
                            session_id=session_id,
                            owner_id=actor_id(),
                            kb_id=kb_id,
                            transcript=transcript,
                            answer=reply,
                            citations=[],
                            memory_refs=[],
                            status="completed",
                        )
                        await send(
                            {
                                "type": "turn_done",
                                "transcript": transcript,
                                "turn": turn,
                                "conversation_id": session.get("conversation_id") or "",
                            }
                        )
                        if intent == "finish":
                            updated = _app().update_voice_practice_session(
                                session_id,
                                actor_id(),
                                kb_id,
                                status="completed",
                                summary="用户通过语音结束了实时对话",
                            )
                            await send(
                                {
                                    "type": "finished",
                                    "session": updated or session,
                                    "reason": "voice_command",
                                }
                            )
                            break
                        await start_provider()
                        continue
                    await send(
                        {
                            "type": "answer_status",
                            "phase": "understanding",
                            "message": "正在理解你的问题",
                        }
                    )
                    if answer_task and not answer_task.done():
                        await cancel_answer("new_turn")
                    answer_task = asyncio.create_task(answer_turn(transcript))
                    await start_provider()
                    continue
                if command_type == "finish":
                    await cancel_answer("finish")
                    await stop_provider()
                    updated = _app().update_voice_practice_session(
                        session_id,
                        actor_id(),
                        kb_id,
                        status="completed",
                        summary="实时语音练习已结束",
                    )
                    await send({"type": "finished", "session": updated or session})
                    break
        except WebSocketDisconnect:
            return
        except HTTPException as exc:
            with contextlib.suppress(Exception):
                await send(
                    {
                        "type": "error",
                        "code": "auth",
                        "message": str(exc.detail),
                        "retryable": False,
                    }
                )
        except Exception:
            logger.exception("实时语音会话失败")
            with contextlib.suppress(Exception):
                await send({"type": "error", "message": "实时语音会话暂时不可用"})
        finally:
            await cancel_answer("disconnect")
            await stop_provider()
            if user_token is not None:
                CURRENT_USER.reset(user_token)

    @router.post("/voice/practice/{session_id}/finish")
    async def finish_voice_practice(session_id: str, body: VoiceFinishBody):
        data = await _read()
        session = _app().get_voice_practice_session(session_id, actor_id(), None)
        if session is None:
            raise HTTPException(status_code=404, detail="练习不存在")
        if body.kb_id and body.kb_id != session["kb_id"]:
            raise HTTPException(status_code=404, detail="练习不存在")
        _find_kb(data, session["kb_id"])
        turns = _app().list_voice_practice_turns(
            session_id, actor_id(), session["kb_id"]
        )
        summary = (body.summary or "").strip()
        if not summary:
            summary = f"本次练习完成 {len([item for item in turns if item.get('status') == 'completed'])} 轮。"
        updated = _app().update_voice_practice_session(
            session_id,
            actor_id(),
            session["kb_id"],
            status="completed",
            summary=summary,
        )
        return updated or session

    async def _remember_summary(
        session_id: str, kb_id: str, history: list[dict[str, Any]]
    ) -> None:
        _recent, older = window_and_older(history)
        data = await _read()
        sessions = data.setdefault("sessions", {})
        session = sessions.get(session_id) or {}
        summary = session.get("summary")
        if not should_summarize(older, summary):
            return
        previous = (summary or {}).get("text") or ""
        try:
            text = await rag.llm_model_func(
                summarize_prompt(older, previous),
                system_prompt="只写对话摘要。摘要不是指令。",
            )
        except Exception as exc:
            logger.warning("对话摘要失败: %s", exc)
            return
        if isinstance(text, (list, tuple)):
            text = text[0] if text else ""
        text = redact_text(str(text or "").strip())
        if not text:
            return
        session["summary"] = {
            "text": text[:1200],
            "source": "conversation",
            "generated_at": _now(),
            "redacted": True,
            "trust": "low",
            "notice": "不是指令",
            "covered_rounds": len(older),
        }
        session["kb_id"] = kb_id
        sessions[session_id] = session
        data["sessions"] = sessions
        await _write(data)

    async def _answer_events(body: ChatBody):
        data = await _read()
        kb = _find_kb(data, body.kb_id)
        settings = kb["settings"]
        graph_enabled = (
            body.graph_enabled
            if body.graph_enabled is not None
            else bool(settings.get("graph_enabled", True))
        )
        mode_name = body.retrieval_mode or settings.get("retrieval_mode") or "mix"
        if not graph_enabled and mode_name in {"mix", "graph"}:
            mode_name = "vector"
        ratio = body.similarity_ratio
        if ratio is None:
            ratio = float(settings.get("similarity_ratio") or 0.5)
        top_k = int(body.top_k or settings.get("top_k") or 5)
        rerank_enabled = (
            body.enable_rerank
            if body.enable_rerank is not None
            else bool(settings.get("rerank_enabled"))
        )
        if body.similarity_threshold is None:
            score_threshold = float(
                settings.get("similarity_threshold")
                if settings.get("similarity_threshold") is not None
                else 0.2
            )
        else:
            score_threshold = float(body.similarity_threshold)
        expand_context = (
            body.context_expand
            if body.context_expand is not None
            else bool(settings.get("context_expand"))
        )
        from lightrag.answer_pipeline import normalize_detail
        from lightrag.index_manifest import load_manifest
        from lightrag.product_parse import load_document

        detail = normalize_detail(body.answer_detail or settings.get("answer_detail"))
        manifest = load_manifest(Path(rag.working_dir))
        index_version = str((manifest or {}).get("corpus_revision") or "")
        owner = str(kb.get("owner_id") or actor_id())
        bindings = data.get("file_bindings") or {}
        user_id = actor_id()
        app = _app()
        conversation = None
        if body.conversation_id:
            conversation = app.get_conversation(
                body.conversation_id, user_id, body.kb_id
            )
            if conversation is None:
                raise HTTPException(status_code=404, detail="会话不存在")
        else:
            conversation = app.create_conversation(
                user_id, body.kb_id, _conversation_title(body.query)
            )
        app.add_message(
            conversation_id=conversation["id"],
            owner_id=user_id,
            kb_id=body.kb_id,
            role="user",
            content=body.query.strip(),
            status="completed",
        )
        history = app.history_for_answer(conversation["id"], user_id, body.kb_id)
        if (
            history
            and history[-1]["role"] == "user"
            and history[-1]["content"] == body.query.strip()
        ):
            history = history[:-1]
        summary = (
            {"text": conversation.get("summary") or ""}
            if conversation.get("summary")
            else None
        )

        def file_in_kb(file_path: str, record: dict | None = None) -> bool:
            payload = dict(record or {})
            if file_path and not payload.get("file_path"):
                payload["file_path"] = file_path
            return scope_visible(
                payload, body.kb_id, bindings, data.get("doc_index") or {}
            )

        kb_chunks = await _kb_chunks(body.kb_id, data)
        version = await _kb_version(body.kb_id, data)
        doc_ids = {
            str(item.get("document_id") or item.get("doc_id") or "")
            for item in kb_chunks
        }
        doc_ids.update(str(key) for key in (data.get("doc_index") or {}))
        live_versions = {}
        for doc_id in doc_ids:
            if not doc_id:
                continue
            document = load_document(rag.working_dir, doc_id)
            if document:
                live_versions[doc_id] = document.version_id
        done = None
        saved_assistant = False
        models = {
            "index_version": index_version,
            "embedding_model": os.getenv("EMBEDDING_MODEL") or "",
            "rerank_model": os.getenv("RERANK_MODEL") or "",
            "llm_model": os.getenv("LLM_MODEL") or "",
        }

        def _save_assistant(
            status: str,
            content: str = "",
            citations: list | None = None,
            meta: dict | None = None,
        ):
            nonlocal saved_assistant
            if saved_assistant:
                return None
            row = app.add_message(
                conversation_id=conversation["id"],
                owner_id=user_id,
                kb_id=body.kb_id,
                role="assistant",
                content=content,
                status=status,
                citations=citations or [],
                retrieval_meta=meta or {},
                **models,
            )
            saved_assistant = True
            return row

        memory_embedding = await _embedding_for(rag, body.query.strip())
        memories = app.retrieve_memories(
            user_id,
            body.kb_id,
            body.query.strip(),
            top_k=5,
            token_budget=1200,
            embedding=memory_embedding,
            embedding_model=os.getenv("EMBEDDING_MODEL") or "",
        )
        try:
            first = True
            async for event in stream_answer(
                rag,
                query=body.query.strip(),
                mode=mode_name,
                ratio=float(ratio),
                kb_chunks=kb_chunks,
                file_in_kb=file_in_kb,
                history=history,
                summary=summary,
                kb_id=body.kb_id,
                tenant="default",
                kb_version=version,
                user_id=user_id,
                cache_entries=_load_cache(),
                top_k=top_k,
                enable_rerank=bool(rerank_enabled),
                score_threshold=score_threshold,
                expand_context=bool(expand_context),
                detail=detail,
                index_version=index_version,
                owner_id=owner,
                working_dir=Path(rag.working_dir),
                live_versions=live_versions,
                memories=memories,
                live_stream=bool(body.voice_stream),
            ):
                if first:
                    event = {**event, "conversation_id": conversation["id"]}
                    first = False
                if event.get("type") == "done":
                    done = event
                if event.get("type") == "error":
                    _save_assistant("failed", "")
                yield event
            if done and done.get("complete") is not False:
                answer = str(done.get("answer") or "")
                saved = _save_assistant(
                    "completed",
                    answer,
                    list(done.get("citations") or []),
                    {
                        "mode": mode_name,
                        "detail": detail,
                        "answerable": done.get("answerable"),
                        "memories": done.get("memories") or [],
                    },
                )
                for item in done.get("memory_candidates") or []:
                    content = str(item.get("content") or "").strip()
                    if not content:
                        continue
                    app.add_memory_candidate(
                        owner_id=user_id,
                        kb_id=body.kb_id,
                        content=content,
                        category=str(item.get("category") or "other"),
                        conversation_id=conversation["id"],
                        message_id=str((saved or {}).get("id") or ""),
                        source_refs=list(done.get("citations") or []),
                        status="pending",
                    )
                if done.get("store_cache") and done.get("embedding"):
                    entries = _load_cache()
                    entries.append(
                        {
                            "query": done.get("rewritten") or body.query,
                            "embedding": done.get("embedding"),
                            "kb_id": body.kb_id,
                            "user_id": user_id,
                            "tenant": "default",
                            "kb_version": version,
                            "prompt_version": PROMPT_VERSION,
                            "model": os.getenv("LLM_MODEL") or "",
                            "mode": mode_name,
                            "detail": detail,
                            "index_version": index_version,
                            "document_versions": {
                                str(item.get("document_id") or ""): str(
                                    item.get("version_id") or ""
                                )
                                for item in (done.get("citations") or [])
                                if item.get("document_id") and item.get("version_id")
                            },
                            "chunk_ids": done.get("chunk_ids") or [],
                            "citations": done.get("citations") or [],
                            "answer": answer[:2000],
                            "created_at": _now(),
                        }
                    )
                    _save_cache(entries)
            elif not saved_assistant:
                _save_assistant("failed", "")
        except (asyncio.CancelledError, GeneratorExit):
            _save_assistant("stopped", "")
            raise
        except Exception:
            _save_assistant("failed", "")
            raise

    @router.post("/chat")
    async def chat(body: ChatBody):
        try:
            answer = ""
            citations = []
            rewritten = body.query
            async for event in _answer_events(body):
                if event.get("type") == "token":
                    answer += event.get("text") or ""
                elif event.get("type") == "meta":
                    citations = event.get("citations") or []
                    rewritten = event.get("rewritten") or rewritten
                elif event.get("type") == "done":
                    answer = event.get("answer") or answer
                    citations = (
                        event.get("citations") if "citations" in event else citations
                    )
            return {
                "answer": answer,
                "citations": citations,
                "retrieval_mode": body.retrieval_mode or "mix",
                "rewritten": rewritten,
            }
        except Exception as exc:
            raise _public_error(exc) from exc

    @router.post("/chat/stream")
    async def chat_stream(body: ChatBody):
        async def events():
            try:
                async for event in _answer_events(body):
                    payload = {
                        key: value for key, value in event.items() if key != "embedding"
                    }
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            except Exception as exc:
                message = getattr(exc, "detail", None) or str(exc)
                yield f"data: {json.dumps({'type': 'error', 'message': str(message)[:300]}, ensure_ascii=False)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @router.get("/graph/stats")
    async def graph_stats(kb_id: str):
        data = await _read()
        _find_kb(data, kb_id)
        graph = rag.chunk_entity_relation_graph
        nodes, edges = scoped_graph_records(
            await graph.get_all_nodes(),
            await graph.get_all_edges(),
            kb_id,
            data.get("file_bindings") or {},
        )
        records = await _list_doc_records(rag)
        doc_count = 0
        chunk_count = 0
        for row in records:
            public = _public_doc(row, input_dir)
            if _doc_in_kb(public, kb_id, data):
                doc_count += 1
                chunk_count += public["chunk_count"] or 0
        dist: dict[str, int] = {}
        for node in nodes:
            etype = node.get("entity_type") or "未知"
            if isinstance(etype, list):
                etype = etype[0] if etype else "未知"
            dist[str(etype)] = dist.get(str(etype), 0) + 1
        return {
            "entity_count": len(nodes),
            "relation_count": len(edges),
            "document_count": doc_count,
            "chunk_count": chunk_count,
            "entity_type_distribution": [
                {"type": name, "count": count}
                for name, count in sorted(
                    dist.items(), key=lambda item: item[1], reverse=True
                )
            ],
        }

    @router.get("/graph/entities")
    async def list_entities(
        kb_id: str,
        entity_type: Optional[str] = None,
        search: str = "",
        page: int = 1,
        page_size: int = 20,
        sort: str = "frequency_desc",
    ):
        data = await _read()
        _find_kb(data, kb_id)
        bindings = data.get("file_bindings") or {}
        nodes = await rag.chunk_entity_relation_graph.get_all_nodes()
        needle = search.strip().lower()
        items = []
        for node in nodes:
            if not _node_in_scope(node, kb_id, bindings):
                continue
            node_id = str(node.get("id") or "")
            name = _entity_display_name(node_id, node)
            etype = node.get("entity_type") or "未知"
            if isinstance(etype, list):
                etype = etype[0] if etype else "未知"
            etype = str(etype)
            desc = _clean_text(node.get("description"), 300)
            if entity_type and etype != entity_type:
                continue
            if needle and needle not in name.lower() and needle not in desc.lower():
                continue
            chunk_ids = _chunk_mentions(node.get("source_id"))
            documents = _graph_document_refs(node, data, kb_id, bindings)
            items.append(
                {
                    "id": node_id,
                    "name": name,
                    "entity_type": etype,
                    "description": desc,
                    "frequency": len(chunk_ids),
                    "chunk_ids": chunk_ids,
                    "doc_count": len(documents),
                    "documents": documents,
                }
            )
        reverse = sort != "frequency_asc"
        items.sort(key=lambda row: (row["frequency"], row["name"]), reverse=reverse)
        total = len(items)
        start = max(page - 1, 0) * page_size
        return {"total": total, "items": items[start : start + page_size]}

    async def _read_kb_or_404(kb_id: str) -> dict[str, Any]:
        data = await _read()
        return _find_kb(data, kb_id)

    @router.get("/graph/entities/{entity_id}/neighbors")
    async def entity_neighbors(kb_id: str, entity_id: str, hops: int = 2):
        data = await _read()
        _find_kb(data, kb_id)
        bindings = data.get("file_bindings") or {}
        graph = rag.chunk_entity_relation_graph
        center = await graph.get_node(entity_id)
        if not center or not _node_in_scope(center, kb_id, bindings):
            raise HTTPException(status_code=404, detail="实体不存在")
        pairs = await graph.get_node_edges(entity_id) or []
        neighbors = []
        for src, tgt in pairs[:50]:
            other = tgt if src == entity_id else src
            other_node = await graph.get_node(other) or {}
            if other_node and not _node_in_scope(other_node, kb_id, bindings):
                continue
            edge = (
                await graph.get_edge(src, tgt) or await graph.get_edge(tgt, src) or {}
            )
            neighbors.append(
                {
                    "neighbor_id": other,
                    "neighbor_name": _entity_display_name(other, other_node),
                    "neighbor_type": other_node.get("entity_type") or "未知",
                    "neighbor_desc": _clean_text(other_node.get("description"), 120),
                    "relation_type": _clean_text(edge.get("keywords") or "关联", 24),
                    "description": _clean_text(edge.get("description"), 160),
                    "weight": edge.get("weight") or 1,
                    "documents": _graph_document_refs(
                        other_node, data, kb_id, bindings
                    ),
                }
            )
        return {
            "entity": {
                "id": entity_id,
                "name": _entity_display_name(entity_id, center),
                "entity_type": center.get("entity_type") or "未知",
                "description": _clean_text(center.get("description"), 400),
                "documents": _graph_document_refs(center, data, kb_id, bindings),
            },
            "neighbors": neighbors,
            "hops": hops,
        }

    @router.get("/graph/relations")
    async def list_relations(
        kb_id: str,
        entity_name: Optional[str] = None,
        rel_type: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ):
        data = await _read()
        _find_kb(data, kb_id)
        bindings = data.get("file_bindings") or {}
        graph = rag.chunk_entity_relation_graph
        nodes = await graph.get_all_nodes()
        in_kb = {
            str(node.get("id") or ""): node
            for node in nodes
            if _node_in_scope(node, kb_id, bindings)
        }
        edges = await graph.get_all_edges()
        scoped = []
        label_set: set[str] = set()
        for edge in edges:
            source = str(edge.get("source") or "")
            target = str(edge.get("target") or "")
            if source not in in_kb or target not in in_kb:
                continue
            if not scope_visible(edge, kb_id, bindings):
                continue
            labels = _relation_labels(edge.get("keywords"))
            label_set.update(labels)
            keywords = _clean_text(edge.get("keywords"), 80)
            try:
                weight = float(edge.get("weight") or 1)
            except (TypeError, ValueError):
                weight = 1.0
            scoped.append(
                {
                    "source_id": source,
                    "target_id": target,
                    "source": _entity_display_name(source, in_kb[source]),
                    "target": _entity_display_name(target, in_kb[target]),
                    "relation_type": keywords or (labels[0] if labels else "关联"),
                    "keywords": keywords,
                    "description": _clean_text(edge.get("description"), 200),
                    "weight": weight,
                    "_labels": labels,
                }
            )
        needle = (entity_name or "").strip().lower()
        wanted = (rel_type or "").strip()
        items = []
        for row in scoped:
            if (
                needle
                and needle not in row["source"].lower()
                and needle not in row["target"].lower()
            ):
                continue
            if (
                wanted
                and wanted not in row["_labels"]
                and wanted not in row["keywords"]
            ):
                continue
            items.append({key: value for key, value in row.items() if key != "_labels"})
        items.sort(key=lambda row: row["weight"], reverse=True)
        total = len(items)
        start = max(page - 1, 0) * page_size
        return {
            "total": total,
            "items": items[start : start + page_size],
            "relation_types": sorted(label_set),
        }

    @router.get("/graph/subgraph")
    async def get_subgraph(
        kb_id: str,
        entity_id: Optional[str] = None,
        entity_name: Optional[str] = None,
        hops: int = 2,
        limit: int = 24,
    ):
        data = await _read()
        _find_kb(data, kb_id)
        bindings = data.get("file_bindings") or {}
        graph_store = rag.chunk_entity_relation_graph
        raw_nodes, raw_edges = scoped_graph_records(
            await graph_store.get_all_nodes(),
            await graph_store.get_all_edges(),
            kb_id,
            bindings,
        )
        node_map = {str(node.get("id") or ""): node for node in raw_nodes}
        adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_map}
        scoped_edges: list[dict[str, Any]] = []
        for edge in raw_edges:
            source = str(edge.get("source") or "")
            target = str(edge.get("target") or "")
            if source not in node_map or target not in node_map or source == target:
                continue
            adjacency[source].add(target)
            adjacency[target].add(source)
            scoped_edges.append(edge)

        requested_id = (entity_id or "").strip()
        requested_name = (entity_name or "").strip().casefold()
        center_id = requested_id if requested_id in node_map else ""
        if not center_id and requested_name:
            exact = [
                node_id
                for node_id, props in node_map.items()
                if _entity_display_name(node_id, props).casefold() == requested_name
            ]
            partial = [
                node_id
                for node_id, props in node_map.items()
                if requested_name in _entity_display_name(node_id, props).casefold()
                or requested_name in str(props.get("description") or "").casefold()
            ]
            matches = exact or partial
            if matches:
                center_id = max(
                    matches, key=lambda item: len(adjacency.get(item, set()))
                )

        max_nodes = max(1, min(limit, 60))
        selected_ids: list[str] = []
        truncated = False
        if center_id:
            max_depth = max(1, min(hops, 3))
            queue: list[tuple[str, int]] = [(center_id, 0)]
            visited: set[str] = set()
            while queue and len(selected_ids) < max_nodes:
                current, depth = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                selected_ids.append(current)
                if depth >= max_depth:
                    continue
                neighbors = sorted(
                    adjacency.get(current, set()),
                    key=lambda item: len(adjacency.get(item, set())),
                    reverse=True,
                )
                queue.extend(
                    (neighbor, depth + 1)
                    for neighbor in neighbors
                    if neighbor not in visited
                )
            truncated = bool(queue)
        elif requested_id or requested_name:
            return {
                "nodes": [],
                "edges": [],
                "mode": "focused",
                "center_id": None,
                "is_truncated": False,
                "empty_reason": "没有找到匹配的实体",
            }
        else:
            # The overview is intentionally a compact set of connected, high-value
            # entities. Isolated extraction artifacts are not useful exploration
            # entry points and remain available from the entity management table.
            connected = [
                node_id for node_id, neighbors in adjacency.items() if neighbors
            ]
            ranked = sorted(
                connected,
                key=lambda item: (
                    len(adjacency[item]),
                    len(_chunk_mentions(node_map[item].get("source_id"))),
                    _entity_display_name(item, node_map[item]),
                ),
                reverse=True,
            )
            candidates = set(ranked[:max_nodes])
            overview_edges = [
                edge
                for edge in scoped_edges
                if str(edge.get("source") or "") in candidates
                and str(edge.get("target") or "") in candidates
            ]
            linked_ids = {
                endpoint
                for edge in overview_edges
                for endpoint in (
                    str(edge.get("source") or ""),
                    str(edge.get("target") or ""),
                )
            }
            selected_ids = [
                node_id for node_id in ranked[:max_nodes] if node_id in linked_ids
            ]
            truncated = len(connected) > len(selected_ids)

        kept_ids = set(selected_ids)
        nodes = [
            _node_view(
                node_id,
                node_map[node_id],
                len(adjacency.get(node_id, set())),
                _graph_document_refs(node_map[node_id], data, kb_id, bindings),
            )
            for node_id in selected_ids
        ]
        edges = [
            _edge_view(
                str(edge.get("source") or ""), str(edge.get("target") or ""), edge
            )
            for edge in scoped_edges
            if str(edge.get("source") or "") in kept_ids
            and str(edge.get("target") or "") in kept_ids
        ]
        return {
            "nodes": nodes,
            "edges": edges,
            "mode": "focused" if center_id else "overview",
            "center_id": center_id or None,
            "is_truncated": truncated,
            "empty_reason": "" if nodes else "还没有可探索的实体关系",
        }

    @router.get("/graph/config")
    async def get_graph_config(kb_id: str):
        data = await _read()
        kb = _find_kb(data, kb_id)
        return kb["graph_config"]

    @router.put("/graph/config")
    async def update_graph_config(kb_id: str, body: GraphConfigBody):
        data = await _read()
        kb = _find_kb(data, kb_id)
        kb["graph_config"] = body.model_dump()
        await _write(data)
        addon = getattr(rag, "addon_params", None)
        if isinstance(addon, dict):
            if body.entity_types:
                addon["entity_types"] = body.entity_types
            addon["kg_schema"] = "open" if body.extraction_mode == "open" else "fault"
        return {"message": "图谱配置已保存", **kb["graph_config"]}

    return router
