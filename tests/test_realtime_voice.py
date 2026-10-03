import asyncio
import json
import sys
import time

import pytest
import numpy as np

from lightrag.product_realtime_voice import (
    AliyunNlsConfig,
    AliyunNlsTtsConfig,
    RealtimeVoiceError,
    _aliyun_signature,
    create_aliyun_token,
    open_aliyun_transcriber,
    parse_transcriber_event,
    stop_aliyun_transcriber,
    stream_aliyun_speech,
)


def test_voice_conversation_controls_do_not_enter_knowledge_retrieval(monkeypatch):
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])
    from lightrag.api.routers.product_shell import _voice_local_intent

    assert _voice_local_intent("OK，结束吧。") == ("finish", "好的，那我们先聊到这里。需要的时候再找我。")
    assert _voice_local_intent("谢谢你") == ("ack", "不客气，你可以继续问我。")
    assert _voice_local_intent("Maas 百事通是怎么做的") is None


@pytest.mark.asyncio
async def test_memory_embedding_accepts_numpy_matrix(monkeypatch):
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])
    from lightrag.api.routers.product_shell import _embedding_for

    class Rag:
        async def embedding_func(self, _texts):
            return np.asarray([[0.25, 0.75]], dtype=np.float32)

    assert await _embedding_for(Rag(), "问题") == [0.25, 0.75]


def _client(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from lightrag.product_appdb import reset_appdb_cache

    working = tmp_path / "rag"
    inputs = tmp_path / "inputs"
    working.mkdir()
    inputs.mkdir()
    monkeypatch.setenv("PRODUCT_AUTH", "1")
    monkeypatch.setenv("TOKEN_SECRET", "realtime-test-secret-32-bytes-long!")
    monkeypatch.setenv("AUTH_RATE_LIMIT", "1000")
    monkeypatch.setenv("APP_ENV", "development")
    reset_appdb_cache()
    sys.argv = [sys.argv[0]]
    from lightrag.api.routers.product_shell import create_product_shell_routes

    class DocStatus:
        async def get_docs_by_status(self, _status):
            return {}

    class Rag:
        working_dir = str(working)
        addon_params = {}
        doc_status = DocStatus()

    class Docs:
        input_dir = str(inputs)

        def is_supported_file(self, _name):
            return True

    app = FastAPI()
    app.include_router(create_product_shell_routes(Rag(), Docs()))
    return TestClient(app), {"X-KB-Request": "1"}


def test_aliyun_signature_is_deterministic_without_exposing_secret():
    params = {"Action": "CreateToken", "AccessKeyId": "access", "Format": "JSON"}
    signature = _aliyun_signature(params, "secret-value")
    assert signature
    assert "secret-value" not in signature
    assert signature == _aliyun_signature(params, "secret-value")
    assert signature != _aliyun_signature(params, "different-secret")


def test_transcriber_events_include_incremental_and_final_text():
    changed = parse_transcriber_event(json.dumps({"header": {"name": "TranscriptionResultChanged"}, "payload": {"result": "你好"}}))
    final = parse_transcriber_event(json.dumps({"header": {"name": "SentenceEnd"}, "payload": {"result": "你好。"}}))
    failed = parse_transcriber_event(json.dumps({
        "header": {"name": "TaskFailed", "status": 40000002, "status_text": "MESSAGE_INVALID"},
        "payload": {},
    }))
    assert changed == {"type": "transcript", "text": "你好", "final": False, "sentence_id": None, "event": "TranscriptionResultChanged"}
    assert final["final"] is True
    assert failed["type"] == "provider_error"
    assert failed["provider_code"] == "40000002"
    assert failed["message"] == "MESSAGE_INVALID"


@pytest.mark.asyncio
async def test_stop_transcription_keeps_app_key_and_empty_payload():
    class FakeSocket:
        sent = []

        async def send(self, value):
            self.sent.append(json.loads(value))

    socket = FakeSocket()
    await stop_aliyun_transcriber(socket, "task-id", "app-key")
    assert len(socket.sent) == 1
    message = socket.sent[0]
    assert message["header"]["name"] == "StopTranscription"
    assert message["header"]["namespace"] == "SpeechTranscriber"
    assert message["header"]["task_id"] == "task-id"
    assert message["header"]["appkey"] == "app-key"
    assert message["payload"] == {}


@pytest.mark.asyncio
async def test_open_transcriber_waits_for_provider_ready(monkeypatch):
    import lightrag.product_realtime_voice as voice

    class FakeSocket:
        def __init__(self):
            self.sent = []
            self.closed = False

        async def send(self, value):
            self.sent.append(value)

        async def recv(self):
            return json.dumps({"header": {"name": "SynthesisStarted"}})

        async def recv(self):
            return json.dumps({
                "header": {"name": "TranscriptionStarted", "task_id": "provider-task"},
                "payload": {},
            })

        async def close(self):
            self.closed = True

    socket = FakeSocket()

    async def connect(*_args, **_kwargs):
        return socket

    async def token(_config):
        return "short-lived-token"

    monkeypatch.setattr(voice.websockets, "connect", connect)
    monkeypatch.setattr(voice, "create_aliyun_token", token)
    result = await open_aliyun_transcriber(AliyunNlsConfig(app_key="app-key", token="token"))
    assert result[0] is socket
    assert result[2] == "app-key"
    assert len(socket.sent) == 1
    assert json.loads(socket.sent[0])["header"]["name"] == "StartTranscription"
    assert socket.closed is False


def test_aliyun_config_requires_app_key_and_credentials(monkeypatch):
    for key in ("ALIYUN_NLS_APP_KEY", "ALIYUN_NLS_TOKEN", "ALIYUN_ACCESS_KEY_ID", "ALIYUN_ACCESS_KEY_SECRET"):
        monkeypatch.delenv(key, raising=False)
    assert not AliyunNlsConfig.from_env().configured
    assert not AliyunNlsTtsConfig.from_env().configured


@pytest.mark.asyncio
async def test_aliyun_tts_is_explicitly_unconfigured(monkeypatch):
    monkeypatch.delenv("ALIYUN_TTS_APP_KEY", raising=False)
    monkeypatch.delenv("ALIYUN_TTS_TOKEN", raising=False)
    monkeypatch.delenv("ALIYUN_NLS_APP_KEY", raising=False)
    monkeypatch.delenv("ALIYUN_NLS_TOKEN", raising=False)
    with pytest.raises(RealtimeVoiceError, match="尚未配置"):
        chunks = []
        async for chunk in stream_aliyun_speech("你好"):
            chunks.append(chunk)


@pytest.mark.asyncio
async def test_aliyun_token_permission_error_is_actionable(monkeypatch):
    import lightrag.product_realtime_voice as voice

    class Response:
        status_code = 200

        def json(self):
            return {"ErrCode": 40020503, "ErrMsg": "No permission!"}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def get(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr(voice.httpx, "AsyncClient", lambda **_kwargs: Client())
    config = AliyunNlsConfig(
        access_key_id="access-key",
        access_key_secret="access-secret",
        app_key="app-key",
    )
    with pytest.raises(RealtimeVoiceError) as caught:
        await create_aliyun_token(config)
    assert caught.value.code == "aliyun_permission_denied"
    assert "NLS 权限" in str(caught.value)


@pytest.mark.asyncio
async def test_aliyun_token_is_reused_until_refresh_window(monkeypatch):
    import lightrag.product_realtime_voice as voice

    voice._ALIYUN_TOKEN_CACHE.clear()

    class Response:
        status_code = 200

        def json(self):
            return {"Token": {"Id": "cached-token", "ExpireTime": int(time.time()) + 3600}}

    class Client:
        calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def get(self, *_args, **_kwargs):
            Client.calls += 1
            return Response()

    monkeypatch.setattr(voice.httpx, "AsyncClient", lambda **_kwargs: Client())
    config = AliyunNlsConfig(
        access_key_id="cache-access-key",
        access_key_secret="cache-access-secret",
        app_key="app-key",
    )
    assert await create_aliyun_token(config) == "cached-token"
    assert await create_aliyun_token(config) == "cached-token"
    assert Client.calls == 1
    voice._ALIYUN_TOKEN_CACHE.clear()


@pytest.mark.asyncio
async def test_aliyun_tts_yields_binary_chunks(monkeypatch):
    import lightrag.product_realtime_voice as voice

    class FakeSocket:
        def __init__(self):
            self.sent = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def send(self, value):
            self.sent.append(value)

        def __aiter__(self):
            return self._events()

        async def _events(self):
            yield b"audio-1"
            yield b"audio-2"
            yield json.dumps({"header": {"name": "SynthesisCompleted"}})

    fake = FakeSocket()
    monkeypatch.setattr(voice, "create_aliyun_token", lambda _config: _token())
    monkeypatch.setattr(voice.websockets, "connect", lambda *_args, **_kwargs: fake)

    async def _token():
        return "token"

    config = AliyunNlsTtsConfig(app_key="tts-app", token="token")
    chunks = [chunk async for chunk in stream_aliyun_speech("你好", config)]
    assert chunks == [b"audio-1", b"audio-2"]
    assert '"StartSynthesis"' in fake.sent[0]
    assert '"text": "你好"' in fake.sent[0]
    assert len(fake.sent) == 1


@pytest.mark.asyncio
async def test_aliyun_tts_sends_text_in_start_directive(monkeypatch):
    import lightrag.product_realtime_voice as voice

    class FakeSocket:
        def __init__(self):
            self.sent = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def send(self, value):
            message = json.loads(value)
            self.sent.append(value)

        def __aiter__(self):
            return self._events()

        async def _events(self):
            yield b"audio"
            yield json.dumps({"header": {"name": "SynthesisCompleted"}})

    fake = FakeSocket()
    monkeypatch.setattr(voice.websockets, "connect", lambda *_args, **_kwargs: fake)
    chunks = [
        chunk
        async for chunk in stream_aliyun_speech(
            "你好", AliyunNlsTtsConfig(app_key="tts-app", token="token")
        )
    ]
    assert chunks == [b"audio"]
    request = json.loads(fake.sent[0])
    assert request["header"]["name"] == "StartSynthesis"
    assert request["payload"]["text"] == "你好"


def test_realtime_websocket_auth_and_kb_scope(tmp_path, monkeypatch):
    client, csrf = _client(tmp_path, monkeypatch)
    first = client.post(
        "/api/v1/auth/register",
        json={"email": "voice-a@example.com", "password": "correct-horse"},
        headers=csrf,
    ).json()
    second = client.post(
        "/api/v1/auth/register",
        json={"email": "voice-b@example.com", "password": "correct-horse"},
        headers=csrf,
    ).json()
    kb = client.post("/api/v1/kb", json={"name": "A"}, headers={**csrf, "Authorization": f"Bearer {first['access_token']}"}).json()["id"]

    with client.websocket_connect("/api/v1/voice/realtime") as socket:
        socket.send_json({"type": "start", "kb_id": kb})
        assert socket.receive_json()["code"] == "auth"

    with client.websocket_connect(
        "/api/v1/voice/realtime",
        subprotocols=[f"kb-access.{second['access_token']}"],
    ) as socket:
        socket.send_json({"type": "start", "kb_id": kb})
        event = socket.receive_json()
        assert event["type"] == "error"
        assert "知识库" in event["message"] or event.get("code") == "auth"
