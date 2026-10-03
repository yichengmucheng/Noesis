"""Realtime voice transport for Aliyun NLS.

The browser talks to the product WebSocket. This module only owns the
provider-side connection and token exchange; knowledge-base authorization and
conversation persistence stay in the product shell router.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator
from urllib.parse import quote

import httpx

try:
    import websockets
except ImportError:  # pragma: no cover - dependency is installed in the API image
    websockets = None  # type: ignore[assignment]


class RealtimeVoiceError(RuntimeError):
    def __init__(self, message: str, *, code: str = "realtime_voice_error"):
        self.code = code
        super().__init__(message)


_ALIYUN_TOKEN_CACHE: dict[tuple[str, str, str], tuple[str, float]] = {}
_ALIYUN_TOKEN_REFRESH_MARGIN_SECONDS = 300


@dataclass(frozen=True)
class AliyunNlsConfig:
    access_key_id: str = ""
    access_key_secret: str = ""
    app_key: str = ""
    region: str = "cn-shanghai"
    gateway: str = "nls-gateway-cn-shanghai.aliyuncs.com"
    token: str = ""
    sample_rate: int = 16000

    @classmethod
    def from_env(cls) -> "AliyunNlsConfig":
        region = os.getenv("ALIYUN_NLS_REGION", "cn-shanghai").strip() or "cn-shanghai"
        gateway = os.getenv(
            "ALIYUN_NLS_GATEWAY",
            f"nls-gateway-{region}.aliyuncs.com",
        ).strip()
        try:
            sample_rate = int(os.getenv("ALIYUN_NLS_SAMPLE_RATE", "16000") or 16000)
        except ValueError:
            sample_rate = 16000
        return cls(
            access_key_id=os.getenv("ALIYUN_ACCESS_KEY_ID", "").strip(),
            access_key_secret=os.getenv("ALIYUN_ACCESS_KEY_SECRET", "").strip(),
            app_key=os.getenv("ALIYUN_NLS_APP_KEY", "").strip(),
            region=region,
            gateway=gateway,
            token=os.getenv("ALIYUN_NLS_TOKEN", "").strip(),
            sample_rate=sample_rate if sample_rate in {8000, 16000} else 16000,
        )

    @property
    def configured(self) -> bool:
        return bool(
            self.app_key
            and (self.token or (self.access_key_id and self.access_key_secret))
        )


@dataclass(frozen=True)
class AliyunNlsTtsConfig:
    """NLS speech synthesis settings.

    NLS uses the same token and gateway as the realtime transcriber, while
    allowing a separate app key because Alibaba projects may enable ASR and
    TTS independently.
    """

    access_key_id: str = ""
    access_key_secret: str = ""
    app_key: str = ""
    region: str = "cn-shanghai"
    gateway: str = "nls-gateway-cn-shanghai.aliyuncs.com"
    token: str = ""
    voice: str = "siyue"
    audio_format: str = "mp3"
    sample_rate: int = 16000

    @classmethod
    def from_env(cls) -> "AliyunNlsTtsConfig":
        region = (
            os.getenv(
                "ALIYUN_TTS_REGION", os.getenv("ALIYUN_NLS_REGION", "cn-shanghai")
            ).strip()
            or "cn-shanghai"
        )
        gateway = os.getenv(
            "ALIYUN_TTS_GATEWAY",
            os.getenv("ALIYUN_NLS_GATEWAY", f"nls-gateway-{region}.aliyuncs.com"),
        ).strip()
        try:
            sample_rate = int(os.getenv("ALIYUN_TTS_SAMPLE_RATE", "16000") or 16000)
        except ValueError:
            sample_rate = 16000
        return cls(
            access_key_id=os.getenv("ALIYUN_ACCESS_KEY_ID", "").strip(),
            access_key_secret=os.getenv("ALIYUN_ACCESS_KEY_SECRET", "").strip(),
            app_key=(
                os.getenv("ALIYUN_TTS_APP_KEY", "").strip()
                or os.getenv("ALIYUN_NLS_APP_KEY", "").strip()
            ),
            region=region,
            gateway=gateway,
            token=(
                os.getenv("ALIYUN_TTS_TOKEN", "").strip()
                or os.getenv("ALIYUN_NLS_TOKEN", "").strip()
            ),
            voice=os.getenv("ALIYUN_TTS_VOICE", "siyue").strip() or "siyue",
            audio_format=os.getenv("ALIYUN_TTS_FORMAT", "mp3").strip().lower() or "mp3",
            sample_rate=sample_rate if sample_rate in {8000, 16000, 24000} else 16000,
        )

    @property
    def configured(self) -> bool:
        return bool(
            self.app_key
            and (self.token or (self.access_key_id and self.access_key_secret))
        )


def _percent(value: str) -> str:
    return quote(str(value), safe="-_.~")


def _aliyun_signature(params: dict[str, str], secret: str) -> str:
    canonical = "&".join(
        f"{_percent(key)}={_percent(params[key])}" for key in sorted(params)
    )
    string_to_sign = "GET&%2F&" + _percent(canonical)
    digest = hmac.new(
        (secret + "&").encode("utf-8"),
        string_to_sign.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    return base64.b64encode(digest).decode("ascii")


async def create_aliyun_token(config: AliyunNlsConfig) -> str:
    if config.token:
        return config.token
    if not config.access_key_id or not config.access_key_secret:
        raise RealtimeVoiceError(
            "阿里云实时语音服务尚未配置", code="aliyun_not_configured"
        )
    cache_key = (config.access_key_id, config.region, config.gateway)
    now = time.time()
    cached = _ALIYUN_TOKEN_CACHE.get(cache_key)
    if cached and cached[1] - _ALIYUN_TOKEN_REFRESH_MARGIN_SECONDS > now:
        return cached[0]
    params = {
        "AccessKeyId": config.access_key_id,
        "Action": "CreateToken",
        "Format": "JSON",
        "RegionId": config.region,
        "SignatureMethod": "HMAC-SHA1",
        "SignatureNonce": uuid.uuid4().hex,
        "SignatureVersion": "1.0",
        "Timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "Version": "2019-02-28",
    }
    params["Signature"] = _aliyun_signature(params, config.access_key_secret)
    endpoint = f"https://nls-meta.{config.region}.aliyuncs.com/"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(endpoint, params=params)
    except httpx.HTTPError as exc:
        if cached and cached[1] > now:
            return cached[0]
        raise RealtimeVoiceError(
            "阿里云实时语音 Token 获取失败", code="aliyun_token_failed"
        ) from exc
    if response.status_code >= 400:
        if response.status_code >= 500 and cached and cached[1] > now:
            return cached[0]
        raise RealtimeVoiceError(
            "阿里云实时语音 Token 获取失败", code="aliyun_token_failed"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise RealtimeVoiceError(
            "阿里云实时语音 Token 返回格式错误", code="aliyun_token_invalid"
        ) from exc
    token_payload = payload.get("Token") or {}
    token = str(token_payload.get("Id") or "").strip()
    if not token:
        error_code = str(payload.get("ErrCode") or payload.get("Code") or "").strip()
        error_message = str(
            payload.get("ErrMsg") or payload.get("Message") or ""
        ).strip()
        if error_code == "40020503" or "no permission" in error_message.lower():
            raise RealtimeVoiceError(
                "阿里云 RAM 用户缺少智能语音交互 NLS 权限",
                code="aliyun_permission_denied",
            )
        if error_code in {"InvalidAccessKeyId.NotFound", "InvalidAccessKeyId"}:
            raise RealtimeVoiceError(
                "阿里云 AccessKey ID 无效",
                code="aliyun_access_key_invalid",
            )
        if error_code in {"SignatureDoesNotMatch", "InvalidAccessKeySecret"}:
            raise RealtimeVoiceError(
                "阿里云 AccessKey Secret 无效",
                code="aliyun_access_key_invalid",
            )
        raise RealtimeVoiceError(
            "阿里云实时语音 Token 为空", code="aliyun_token_invalid"
        )
    try:
        expires_at = float(token_payload.get("ExpireTime") or 0)
    except (TypeError, ValueError):
        expires_at = 0
    if expires_at > 10_000_000_000:
        expires_at /= 1000
    if expires_at <= now:
        expires_at = now + 23 * 60 * 60
    _ALIYUN_TOKEN_CACHE[cache_key] = (token, expires_at)
    return token


def _message(
    name: str,
    task_id: str,
    app_key: str,
    *,
    namespace: str = "SpeechTranscriber",
) -> dict[str, Any]:
    return {
        "header": {
            "appkey": app_key,
            "namespace": namespace,
            "name": name,
            "task_id": task_id,
            "message_id": uuid.uuid4().hex,
        }
    }


async def open_aliyun_transcriber(config: AliyunNlsConfig):
    if websockets is None:
        raise RealtimeVoiceError(
            "实时语音依赖未安装", code="realtime_dependency_missing"
        )
    if not config.configured:
        raise RealtimeVoiceError(
            "阿里云实时语音服务尚未配置", code="aliyun_not_configured"
        )
    token = await create_aliyun_token(config)
    task_id = uuid.uuid4().hex
    url = f"wss://{config.gateway}/ws/v1?token={quote(token, safe='')}"
    socket = None
    try:
        socket = await websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
            max_size=8 * 1024 * 1024,
        )
        start = _message("StartTranscription", task_id, config.app_key)
        start["payload"] = {
            "format": "pcm",
            "sample_rate": config.sample_rate,
            "enable_intermediate_result": True,
            "enable_punctuation_prediction": True,
            "enable_inverse_text_normalization": True,
            "max_sentence_silence": int(
                os.getenv("ALIYUN_NLS_MAX_SENTENCE_SILENCE", "800") or 800
            ),
        }
        await socket.send(json.dumps(start, ensure_ascii=False))
        try:
            start_timeout = float(os.getenv("ALIYUN_NLS_START_TIMEOUT", "10") or 10)
        except ValueError:
            start_timeout = 10
        while True:
            raw = await asyncio.wait_for(socket.recv(), timeout=max(1.0, start_timeout))
            event = parse_transcriber_event(raw)
            if not event:
                continue
            if event.get("type") == "provider_ready":
                break
            if event.get("type") == "provider_error":
                provider_code = str(event.get("provider_code") or "").strip()
                detail = str(event.get("message") or "阿里云拒绝启动实时识别").strip()
                raise RealtimeVoiceError(
                    detail,
                    code=f"aliyun_start_failed{':' + provider_code if provider_code else ''}",
                )
        return socket, task_id, config.app_key
    except RealtimeVoiceError:
        if socket is not None:
            await close_aliyun_socket(socket)
        raise
    except Exception as exc:
        if socket is not None:
            await close_aliyun_socket(socket)
        raise RealtimeVoiceError(
            "无法连接阿里云实时语音服务", code="aliyun_connect_failed"
        ) from exc


def parse_transcriber_event(raw: str | bytes) -> dict[str, Any] | None:
    if isinstance(raw, bytes):
        return None
    try:
        event = json.loads(raw)
    except (TypeError, ValueError):
        return None
    header = event.get("header") or {}
    payload = event.get("payload") or {}
    name = str(header.get("name") or "")
    if name in {"TranscriptionResultChanged", "SentenceEnd"}:
        text = str(payload.get("result") or "").strip()
        if not text:
            return None
        return {
            "type": "transcript",
            "text": text,
            "final": name == "SentenceEnd",
            "sentence_id": payload.get("sentence_id"),
            "event": name,
        }
    if name == "TranscriptionStarted":
        return {"type": "provider_ready", "task_id": header.get("task_id")}
    if name in {"TranscriptionCompleted", "TaskFailed", "TranscriptionFailed"}:
        status_code = (
            header.get("status")
            or header.get("status_code")
            or payload.get("status")
            or payload.get("status_code")
        )
        status_text = (
            header.get("status_text")
            or header.get("message")
            or payload.get("status_text")
            or payload.get("message")
            or ""
        )
        return {
            "type": "provider_done"
            if name == "TranscriptionCompleted"
            else "provider_error",
            "message": str(status_text),
            "provider_code": str(status_code) if status_code is not None else "",
            "event": name,
        }
    return None


async def stop_aliyun_transcriber(socket: Any, task_id: str, app_key: str) -> None:
    # NLS validates StopTranscription as a complete control request. Omitting
    # appkey or payload makes the gateway reject an otherwise healthy session
    # with MESSAGE_INVALID (40000002).
    stop = _message("StopTranscription", task_id, app_key)
    stop["payload"] = {}
    try:
        await socket.send(json.dumps(stop, ensure_ascii=False))
    except Exception:
        return


async def close_aliyun_socket(socket: Any) -> None:
    try:
        await socket.close()
    except Exception:
        return


async def stream_aliyun_speech(
    text: str, config: AliyunNlsTtsConfig | None = None
) -> AsyncIterator[bytes]:
    """Stream NLS synthesis audio without buffering the whole answer."""
    if websockets is None:
        raise RealtimeVoiceError(
            "实时语音依赖未安装", code="realtime_dependency_missing"
        )
    config = config or AliyunNlsTtsConfig.from_env()
    clean = str(text or "").strip()
    if not clean:
        raise RealtimeVoiceError("没有可播放的回答", code="empty_text")
    if not config.configured:
        raise RealtimeVoiceError(
            "阿里云语音合成服务尚未配置", code="aliyun_tts_not_configured"
        )
    token = await create_aliyun_token(
        AliyunNlsConfig(
            access_key_id=config.access_key_id,
            access_key_secret=config.access_key_secret,
            app_key=config.app_key,
            region=config.region,
            gateway=config.gateway,
            token=config.token,
            sample_rate=config.sample_rate,
        )
    )
    task_id = uuid.uuid4().hex
    url = f"wss://{config.gateway}/ws/v1?token={quote(token, safe='')}"
    emitted = False
    try:
        async with websockets.connect(
            url, ping_interval=20, ping_timeout=20, close_timeout=5
        ) as socket:
            start = _message(
                "StartSynthesis",
                task_id,
                config.app_key,
                namespace="SpeechSynthesizer",
            )
            start["payload"] = {
                "voice": config.voice,
                "format": config.audio_format,
                "sample_rate": config.sample_rate,
                "volume": 50,
                "speech_rate": 0,
                "pitch_rate": 0,
                "text": clean[:16000],
            }
            await socket.send(json.dumps(start, ensure_ascii=False))
            async for raw in socket:
                if isinstance(raw, bytes):
                    if raw:
                        emitted = True
                        yield raw
                    continue
                event = parse_transcriber_event(raw)
                if event and event.get("type") == "provider_error":
                    raise RealtimeVoiceError(
                        str(event.get("message") or "阿里云语音合成失败"),
                        code="aliyun_tts_failed",
                    )
                try:
                    header = (json.loads(raw) or {}).get("header") or {}
                except (TypeError, ValueError):
                    continue
                name = str(header.get("name") or "")
                if name in {"SynthesisCompleted", "TaskFailed", "SynthesisFailed"}:
                    if name != "SynthesisCompleted":
                        raise RealtimeVoiceError(
                            "阿里云语音合成失败", code="aliyun_tts_failed"
                        )
                    break
    except RealtimeVoiceError:
        raise
    except Exception as exc:
        raise RealtimeVoiceError(
            "阿里云语音合成连接失败", code="aliyun_tts_failed"
        ) from exc
    if not emitted:
        raise RealtimeVoiceError("阿里云语音合成返回空音频", code="aliyun_tts_empty")
