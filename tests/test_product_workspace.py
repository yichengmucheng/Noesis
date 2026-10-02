# -*- coding: utf-8 -*-
import asyncio
import json
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightrag.product_appdb import open_appdb, reset_appdb_cache
from lightrag.product_storage import load_shell, mutate_shell


def _client(tmp_path, monkeypatch):
    working = tmp_path / "rag"
    inputs = tmp_path / "inputs"
    working.mkdir()
    inputs.mkdir()
    monkeypatch.setenv("PRODUCT_AUTH", "1")
    monkeypatch.setenv("TOKEN_SECRET", "unit-test-secret-for-workspace-32b!")
    monkeypatch.setenv("AUTH_RATE_LIMIT", "1000")
    monkeypatch.setenv("APP_ENV", "development")
    reset_appdb_cache()

    class Rag:
        def __init__(self):
            self.working_dir = str(working)
            self.addon_params = {}

        async def get_docs_by_status(self, _status):
            return {}

    class Docs:
        def __init__(self):
            self.input_dir = str(inputs)

        def is_supported_file(self, name: str) -> bool:
            return True

    sys.argv = [sys.argv[0]]
    from lightrag.api.routers.product_shell import create_product_shell_routes

    app = FastAPI()
    app.include_router(create_product_shell_routes(Rag(), Docs()))
    client = TestClient(app)
    csrf = {"X-KB-Request": "1"}
    return client, working, csrf


def test_conversation_crud_isolation_and_kb_purge(tmp_path, monkeypatch):
    client, working, csrf = _client(tmp_path, monkeypatch)
    owner_token = client.post("/api/v1/auth/register", json={"email": "conv-owner@example.com", "password": "correct-horse"}, headers=csrf)
    other_token = client.post("/api/v1/auth/register", json={"email": "conv-other@example.com", "password": "correct-horse"}, headers=csrf)
    owner = {"Authorization": f"Bearer {owner_token.json()['access_token']}", **csrf}
    stranger = {"Authorization": f"Bearer {other_token.json()['access_token']}", **csrf}
    kb_id = client.post("/api/v1/kb", json={"name": "资料库"}, headers=owner).json()["id"]
    other_kb = client.post("/api/v1/kb", json={"name": "另一库"}, headers=owner).json()["id"]
    created = client.post("/api/v1/conversations", json={"kb_id": kb_id, "title": "冷却系统"}, headers=owner)
    assert created.status_code == 200, created.text
    conv_id = created.json()["id"]
    listed = client.get("/api/v1/conversations", params={"kb_id": kb_id}, headers=owner).json()["items"]
    assert listed[0]["id"] == conv_id
    assert listed[0]["is_pinned"] is False
    renamed = client.patch(f"/api/v1/conversations/{conv_id}", json={"kb_id": kb_id, "title": "节温器"}, headers=owner)
    assert renamed.json()["title"] == "节温器"
    pinned = client.post(f"/api/v1/conversations/{conv_id}/pin", params={"kb_id": kb_id}, headers=owner)
    assert pinned.json()["is_pinned"] is True
    archived = client.post(f"/api/v1/conversations/{conv_id}/archive", params={"kb_id": kb_id}, headers=owner)
    assert archived.json()["is_archived"] is True
    assert client.get("/api/v1/conversations", params={"kb_id": kb_id}, headers=owner).json()["items"] == []
    restored = client.patch(f"/api/v1/conversations/{conv_id}", json={"kb_id": kb_id, "is_archived": False}, headers=owner)
    assert restored.json()["is_archived"] is False
    assert client.get(f"/api/v1/conversations/{conv_id}", params={"kb_id": kb_id}, headers=stranger).status_code == 404
    assert client.get(f"/api/v1/conversations/{conv_id}", params={"kb_id": other_kb}, headers=owner).status_code == 404
    searched = client.get("/api/v1/conversations", params={"kb_id": kb_id, "q": "节温"}, headers=owner).json()["items"]
    assert searched[0]["id"] == conv_id

    store = open_appdb(working)
    assert store.schema_versions() == ["app-001", "app-002", "app-003"]
    store.migrate()
    assert store.schema_versions() == ["app-001", "app-002", "app-003"]

    doc_before = {"doc-keep": {"kb_id": kb_id, "display_name": "note.txt"}}

    def keep(data):
        data["doc_index"] = dict(doc_before)

    mutate_shell(working, keep)
    assert client.delete(f"/api/v1/conversations/{conv_id}", params={"kb_id": kb_id}, headers=owner).status_code == 200
    assert load_shell(working)["doc_index"]["doc-keep"]["display_name"] == "note.txt"

    again = client.post("/api/v1/conversations", json={"kb_id": kb_id, "title": "待清理"}, headers=owner).json()
    client.delete(f"/api/v1/kb/{kb_id}", headers=owner)
    assert open_appdb(working).get_conversation(again["id"], again["owner_id"], kb_id) is None


def test_stream_persists_completed_and_not_partial(tmp_path, monkeypatch):
    client, working, csrf = _client(tmp_path, monkeypatch)
    token = client.post("/api/v1/auth/register", json={"email": "conv-stream@example.com", "password": "correct-horse"}, headers=csrf).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}", **csrf}
    kb_id = client.post("/api/v1/kb", json={"name": "资料库"}, headers=headers).json()["id"]
    conv_id = client.post("/api/v1/conversations", json={"kb_id": kb_id, "title": "问答"}, headers=headers).json()["id"]

    async def complete(*_args, **_kwargs):
        yield {"type": "meta", "answerable": True, "citations": []}
        yield {"type": "token", "text": "完整回答"}
        yield {"type": "done", "answer": "完整回答", "complete": True, "citations": [], "store_cache": False}

    async def hanging(*_args, **_kwargs):
        yield {"type": "meta", "answerable": True, "citations": []}
        yield {"type": "token", "text": "半截答案"}
        await asyncio.Event().wait()
        yield {"type": "done", "answer": "半截答案", "complete": True}

    async def failing(*_args, **_kwargs):
        yield {"type": "error", "message": "问答服务暂时不可用"}

    sys.argv = [sys.argv[0]]
    from lightrag.api.routers import product_shell as shell_module

    monkeypatch.setattr(shell_module, "stream_answer", complete)
    streamed = client.post("/api/v1/chat/stream", json={"query": "怎么办", "kb_id": kb_id, "conversation_id": conv_id}, headers=headers)
    assert streamed.status_code == 200
    items = client.get(f"/api/v1/conversations/{conv_id}/messages", params={"kb_id": kb_id}, headers=headers).json()["items"]
    assert [item["role"] for item in items] == ["user", "assistant"]
    assert items[0]["status"] == "completed"
    assert items[1]["status"] == "completed"
    assert items[1]["content"] == "完整回答"

    monkeypatch.setattr(shell_module, "stream_answer", failing)
    failed = client.post("/api/v1/chat/stream", json={"query": "失败问题", "kb_id": kb_id, "conversation_id": conv_id}, headers=headers)
    assert failed.status_code == 200
    statuses = [item["status"] for item in client.get(f"/api/v1/conversations/{conv_id}/messages", params={"kb_id": kb_id}, headers=headers).json()["items"]]
    assert "failed" in statuses
    failed_row = [item for item in client.get(f"/api/v1/conversations/{conv_id}/messages", params={"kb_id": kb_id}, headers=headers).json()["items"] if item["status"] == "failed"][0]
    assert failed_row["content"] == ""

    monkeypatch.setattr(shell_module, "stream_answer", hanging)
    app = client.app

    async def stop_early():
        payload = json.dumps({"query": "停一下", "kb_id": kb_id, "conversation_id": conv_id}).encode("utf-8")
        state = {"phase": "request", "seen": asyncio.Event(), "bodies": []}

        async def receive():
            if state["phase"] == "request":
                state["phase"] = "open"
                return {"type": "http.request", "body": payload, "more_body": False}
            await state["seen"].wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body":
                chunk = message.get("body") or b""
                state["bodies"].append(chunk)
                if "半截答案".encode("utf-8") in chunk:
                    state["seen"].set()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/v1/chat/stream",
            "raw_path": b"/api/v1/chat/stream",
            "query_string": b"",
            "headers": [
                [b"content-type", b"application/json"],
                [b"content-length", str(len(payload)).encode("ascii")],
                [b"authorization", f"Bearer {token}".encode("utf-8")],
            ],
            "client": ("127.0.0.1", 9),
            "server": ("test", 80),
        }
        await app(scope, receive, send)

    asyncio.run(asyncio.wait_for(stop_early(), timeout=20))
    owner_id = client.get("/api/v1/kb", headers=headers).json()["items"][0]["owner_id"]
    saved = open_appdb(working).list_messages(conv_id, owner_id, kb_id)
    stopped = [item for item in saved if item["status"] == "stopped"]
    assert stopped
    assert all(item["content"] != "半截答案" for item in stopped)
    assert "半截答案" not in json.dumps(load_shell(working).get("qa_pairs") or [], ensure_ascii=False)


def test_feedback_saves_snapshot_and_offline_eval_without_changing_thresholds(tmp_path, monkeypatch):
    client, working, csrf = _client(tmp_path, monkeypatch)
    owner_token = client.post("/api/v1/auth/register", json={"email": "fb-owner@example.com", "password": "correct-horse"}, headers=csrf)
    other_token = client.post("/api/v1/auth/register", json={"email": "fb-other@example.com", "password": "correct-horse"}, headers=csrf)
    owner = {"Authorization": f"Bearer {owner_token.json()['access_token']}", **csrf}
    stranger = {"Authorization": f"Bearer {other_token.json()['access_token']}", **csrf}
    kb_id = client.post("/api/v1/kb", json={"name": "资料库"}, headers=owner).json()["id"]
    conv_id = client.post("/api/v1/conversations", json={"kb_id": kb_id, "title": "问答"}, headers=owner).json()["id"]

    async def complete(*_args, **_kwargs):
        yield {"type": "done", "answer": "完整回答 [C1]", "complete": True, "citations": [{"citation_id": "C1"}], "store_cache": False}

    import sys
    sys.argv = [sys.argv[0]]
    from lightrag.api.routers import product_shell as shell_module
    from lightrag.retrieval_admission import _PROFILE_PATH

    before = _PROFILE_PATH.read_text(encoding="utf-8") if _PROFILE_PATH.is_file() else ""
    monkeypatch.setattr(shell_module, "stream_answer", complete)
    client.post("/api/v1/chat/stream", json={"query": "节温器怎么查", "kb_id": kb_id, "conversation_id": conv_id}, headers=owner)
    items = client.get(f"/api/v1/conversations/{conv_id}/messages", params={"kb_id": kb_id}, headers=owner).json()["items"]
    assistant = [item for item in items if item["role"] == "assistant"][0]
    liked = client.post(f"/api/v1/messages/{assistant['id']}/feedback", json={"kb_id": kb_id, "rating": "positive"}, headers=owner)
    assert liked.status_code == 200, liked.text
    assert liked.json()["rating"] == "positive"
    disliked = client.post(
        f"/api/v1/messages/{assistant['id']}/feedback",
        json={"kb_id": kb_id, "rating": "negative", "reason": "citation_wrong", "comment": "页码不对"},
        headers=owner,
    )
    assert disliked.status_code == 200, disliked.text
    body = disliked.json()
    assert body["question"] == "节温器怎么查"
    assert body["answer"] == "完整回答 [C1]"
    assert body["reason"] == "citation_wrong"
    assert client.post(
        f"/api/v1/messages/{assistant['id']}/feedback",
        json={"kb_id": kb_id, "rating": "negative", "reason": "citation_wrong"},
        headers=stranger,
    ).status_code == 404
    store = open_appdb(working)
    owner_id = client.get("/api/v1/kb", headers=owner).json()["items"][0]["owner_id"]
    offline = store.list_offline_eval(owner_id, kb_id)
    assert offline
    assert offline[0]["payload"]["reason"] == "citation_wrong"
    after = _PROFILE_PATH.read_text(encoding="utf-8") if _PROFILE_PATH.is_file() else ""
    assert after == before


def test_memory_confirm_isolation_and_kb_purge_keeps_user_memories(tmp_path, monkeypatch):
    client, working, csrf = _client(tmp_path, monkeypatch)
    owner_token = client.post("/api/v1/auth/register", json={"email": "mem-owner@example.com", "password": "correct-horse"}, headers=csrf)
    other_token = client.post("/api/v1/auth/register", json={"email": "mem-other@example.com", "password": "correct-horse"}, headers=csrf)
    owner = {"Authorization": f"Bearer {owner_token.json()['access_token']}", **csrf}
    stranger = {"Authorization": f"Bearer {other_token.json()['access_token']}", **csrf}
    kb_id = client.post("/api/v1/kb", json={"name": "资料库"}, headers=owner).json()["id"]
    conv_id = client.post("/api/v1/conversations", json={"kb_id": kb_id, "title": "问答"}, headers=owner).json()["id"]
    captured = {}

    async def complete(*_args, **kwargs):
        captured["memories"] = list(kwargs.get("memories") or [])
        yield {"type": "done", "answer": "完整回答", "complete": True, "citations": [], "store_cache": False, "memory_candidates": [{"content": "模型提出的偏好", "category": "preference"}]}

    from lightrag.api.routers import product_shell as shell_module
    monkeypatch.setattr(shell_module, "stream_answer", complete)
    client.post("/api/v1/chat/stream", json={"query": "记住我喜欢简洁", "kb_id": kb_id, "conversation_id": conv_id}, headers=owner)
    assert captured["memories"] == []
    pending = client.get("/api/v1/memory-candidates", params={"kb_id": kb_id}, headers=owner).json()["items"]
    assert pending and pending[0]["content"] == "模型提出的偏好"
    assert pending[0]["status"] == "pending"
    assert client.get("/api/v1/memory-candidates", params={"kb_id": kb_id}, headers=stranger).status_code == 404
    accepted = client.post(f"/api/v1/memory-candidates/{pending[0]['id']}/accept", params={"kb_id": kb_id}, headers=owner)
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["content"] == "模型提出的偏好"
    assert client.get("/api/v1/memory-candidates", params={"kb_id": kb_id}, headers=owner).json()["items"] == []

    client.post("/api/v1/chat/stream", json={"query": "再用一次", "kb_id": kb_id, "conversation_id": conv_id}, headers=owner)
    assert any(item["content"] == "模型提出的偏好" for item in captured["memories"])

    memory_id = client.get("/api/v1/memories", headers=owner).json()["items"][0]["id"]
    assert client.patch(f"/api/v1/memories/{memory_id}", json={"enabled": False}, headers=stranger).status_code == 404
    disabled = client.patch(f"/api/v1/memories/{memory_id}", json={"enabled": False}, headers=owner)
    assert disabled.json()["enabled"] is False
    client.post("/api/v1/chat/stream", json={"query": "关闭后", "kb_id": kb_id, "conversation_id": conv_id}, headers=owner)
    assert all(item["content"] != "模型提出的偏好" for item in captured["memories"])

    edited = client.patch(f"/api/v1/memories/{memory_id}", json={"content": "改成详细", "enabled": True}, headers=owner)
    assert edited.json()["content"] == "改成详细"
    client.post("/api/v1/chat/stream", json={"query": "编辑后", "kb_id": kb_id, "conversation_id": conv_id}, headers=owner)
    assert any(item["content"] == "改成详细" for item in captured["memories"])

    extra = client.post("/api/v1/memory-candidates", json={"kb_id": kb_id, "content": "待清理候选"}, headers=owner).json()
    client.post(f"/api/v1/memory-candidates/{extra['id']}/reject", params={"kb_id": kb_id}, headers=owner)
    leftover = client.post("/api/v1/memory-candidates", json={"kb_id": kb_id, "content": "知识库级候选"}, headers=owner).json()
    client.delete(f"/api/v1/kb/{kb_id}", headers=owner)
    store = open_appdb(working)
    owner_id = accepted.json()["owner_id"]
    assert store.get_memory_candidate(leftover["id"], owner_id, leftover["kb_id"]) is None
    kept = store.get_memory(memory_id, owner_id)
    assert kept is not None
    assert kept["content"] == "改成详细"
    assert client.delete(f"/api/v1/memories/{memory_id}", headers=owner).status_code == 200
    assert store.get_memory(memory_id, owner_id) is None
