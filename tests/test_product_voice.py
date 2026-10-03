import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient

import pytest

from lightrag.product_appdb import (
    backup_appdb,
    open_appdb,
    reset_appdb_cache,
    restore_appdb,
)
from lightrag.product_voice import (
    VoiceProviderError,
    synthesize_speech,
    transcribe_audio,
)


def _client(tmp_path, monkeypatch):
    working = tmp_path / "rag"
    inputs = tmp_path / "inputs"
    working.mkdir()
    inputs.mkdir()
    monkeypatch.setenv("PRODUCT_AUTH", "1")
    monkeypatch.setenv("TOKEN_SECRET", "voice-unit-test-secret-32-bytes-long!")
    monkeypatch.setenv("AUTH_RATE_LIMIT", "1000")
    monkeypatch.setenv("APP_ENV", "development")
    reset_appdb_cache()

    class DocStatus:
        async def get_docs_by_status(self, _status):
            return {}

    class Rag:
        def __init__(self):
            self.working_dir = str(working)
            self.addon_params = {}
            self.doc_status = DocStatus()

        async def get_docs_by_status(self, _status):
            return {}

    class Docs:
        input_dir = str(inputs)

        def is_supported_file(self, _name):
            return True

    sys.argv = [sys.argv[0]]
    from lightrag.api.routers.product_shell import create_product_shell_routes

    app = FastAPI()
    app.include_router(create_product_shell_routes(Rag(), Docs()))
    return TestClient(app), {"X-KB-Request": "1"}


def test_voice_schema_migrates_and_isolates(tmp_path):
    store = open_appdb(tmp_path)
    assert "app-005" in store.schema_versions()
    first = store.create_voice_practice_session("user-a", "kb-a", goal="interview")
    second = store.create_voice_practice_session("user-a", "kb-b")
    store.add_voice_practice_turn(
        session_id=first["id"],
        owner_id="user-a",
        kb_id="kb-a",
        transcript="如何复述这份资料？",
        answer="回答 [C1]",
        citations=[{"citation_id": "C1", "document_id": "doc-a"}],
        memory_refs=[{"marker": "M1", "content": "偏好简洁"}],
    )
    assert (
        store.get_voice_practice_session(first["id"], "user-a", "kb-a")["id"]
        == first["id"]
    )
    assert store.get_voice_practice_session(first["id"], "user-b", "kb-a") is None
    assert store.get_voice_practice_session(first["id"], "user-a", "kb-b") is None
    assert len(store.list_voice_practice_turns(first["id"], "user-a", "kb-a")) == 1
    assert store.update_voice_practice_turn_audio("missing", "user-a", "ready") is None
    assert (
        store.update_voice_practice_turn_audio(
            store.list_voice_practice_turns(first["id"], "user-a", "kb-a")[0]["id"],
            "user-a",
            "ready",
        )["audio_status"]
        == "ready"
    )
    listed = store.list_voice_practice_sessions("user-a", "kb-a")
    assert len(listed) == 1
    assert listed[0]["turn_count"] == 1
    assert listed[0]["completed_turn_count"] == 1
    assert listed[0]["cited_turn_count"] == 1
    assert listed[0]["first_transcript"] == "如何复述这份资料？"
    assert store.list_voice_practice_turns(second["id"], "user-a", "kb-b") == []
    store.purge_kb("kb-a", "user-a")
    assert store.get_voice_practice_session(first["id"], "user-a", "kb-a") is None
    assert store.get_voice_practice_session(second["id"], "user-a", "kb-b") is not None
    reset_appdb_cache()


def test_voice_backup_restore_keeps_session_and_turn(tmp_path):
    store = open_appdb(tmp_path)
    session = store.create_voice_practice_session("user-a", "kb-a")
    turn = store.add_voice_practice_turn(
        session_id=session["id"],
        owner_id="user-a",
        kb_id="kb-a",
        transcript="问题",
        answer="回答",
    )
    backup = backup_appdb(tmp_path, tmp_path / "backup.sqlite")
    reset_appdb_cache()
    restore_appdb(tmp_path, backup)
    restored = open_appdb(tmp_path)
    assert (
        restored.get_voice_practice_session(session["id"], "user-a", "kb-a")["id"]
        == session["id"]
    )
    assert restored.get_voice_practice_turn(turn["id"], "user-a")["answer"] == "回答"
    reset_appdb_cache()


@pytest.mark.asyncio
async def test_unconfigured_voice_provider_is_explicit(monkeypatch):
    for key in (
        "ASR_API_BASE",
        "ASR_API_KEY",
        "ASR_MODEL",
        "TTS_API_BASE",
        "TTS_API_KEY",
        "TTS_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(VoiceProviderError, match="尚未配置"):
        await transcribe_audio(b"audio", "recording.webm", "audio/webm")
    with pytest.raises(VoiceProviderError, match="尚未配置"):
        await synthesize_speech("回答")


@pytest.mark.asyncio
async def test_configured_voice_provider_success(monkeypatch):
    monkeypatch.setenv("ASR_API_BASE", "https://asr.example/v1")
    monkeypatch.setenv("ASR_API_KEY", "test-asr-key")
    monkeypatch.setenv("ASR_MODEL", "asr-test")
    monkeypatch.setenv("TTS_API_BASE", "https://tts.example/v1")
    monkeypatch.setenv("TTS_API_KEY", "test-tts-key")
    monkeypatch.setenv("TTS_MODEL", "tts-test")

    class Response:
        status_code = 200
        content = b"fake-mp3"
        headers = {"content-type": "audio/mpeg"}

        def json(self):
            return {"text": "转写结果"}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, url, **_kwargs):
            response = Response()
            if url.endswith("/audio/transcriptions"):
                response.json = lambda: {"text": "转写结果"}
            return response

    import lightrag.product_voice as voice

    monkeypatch.setattr(voice.httpx, "AsyncClient", lambda **_kwargs: Client())
    assert (
        await transcribe_audio(b"audio", "recording.webm", "audio/webm") == "转写结果"
    )
    audio, content_type = await synthesize_speech("回答")
    assert audio == b"fake-mp3"
    assert content_type == "audio/mpeg"


def test_voice_api_reuses_answer_and_enforces_scope(tmp_path, monkeypatch):
    client, csrf = _client(tmp_path, monkeypatch)
    token = client.post(
        "/api/v1/auth/register",
        json={"email": "voice-owner@example.com", "password": "correct-horse"},
        headers=csrf,
    ).json()["access_token"]
    other_token = client.post(
        "/api/v1/auth/register",
        json={"email": "voice-other@example.com", "password": "correct-horse"},
        headers=csrf,
    ).json()["access_token"]
    owner = {"Authorization": f"Bearer {token}", **csrf}
    other = {"Authorization": f"Bearer {other_token}", **csrf}
    kb = client.post("/api/v1/kb", json={"name": "练习库"}, headers=owner).json()["id"]
    created = client.post(
        "/api/v1/voice/practice/sessions",
        json={"kb_id": kb, "goal": "interview"},
        headers=owner,
    )
    assert created.status_code == 200, created.text
    session_id = created.json()["id"]
    assert (
        client.get(
            f"/api/v1/voice/practice/sessions/{session_id}",
            params={"kb_id": kb},
            headers=other,
        ).status_code
        == 404
    )

    async def fake_stream(*_args, **_kwargs):
        yield {
            "type": "meta",
            "citations": [{"citation_id": "C1", "doc_name": "notes.txt"}],
        }
        yield {
            "type": "done",
            "answer": "资料回答 [C1] [M1]",
            "citations": [{"citation_id": "C1", "doc_name": "notes.txt"}],
            "memories": [{"marker": "M1", "content": "偏好简洁"}],
            "complete": True,
            "store_cache": False,
        }

    import lightrag.api.routers.product_shell as shell_module

    monkeypatch.setattr(shell_module, "stream_answer", fake_stream)
    turn = client.post(
        f"/api/v1/voice/practice/{session_id}/turn",
        json={"kb_id": kb, "transcript": "请复述资料"},
        headers=owner,
    )
    assert turn.status_code == 200, turn.text
    assert turn.json()["citations"][0]["citation_id"] == "C1"
    assert turn.json()["memories"][0]["marker"] == "M1"
    turn_id = turn.json()["turn_id"]
    assert (
        client.post(
            "/api/v1/voice/speech",
            json={"turn_id": turn_id, "kb_id": kb},
            headers=other,
        ).status_code
        == 404
    )
    finished = client.post(
        f"/api/v1/voice/practice/{session_id}/finish", json={"kb_id": kb}, headers=owner
    )
    assert finished.status_code == 200
    assert finished.json()["status"] == "completed"
