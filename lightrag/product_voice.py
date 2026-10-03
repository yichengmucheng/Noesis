"""配置驱动的 ASR/TTS 适配层。

供应商只通过 OpenAI-compatible HTTP 接口接入。未配置时显式失败，不能把
占位文本或空音频当成成功结果。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx


ALLOWED_AUDIO_TYPES = {
    "audio/webm",
    "audio/ogg",
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp4",
    "audio/aac",
}
MAX_AUDIO_BYTES = int(os.getenv("VOICE_MAX_AUDIO_MB", "25") or 25) * 1024 * 1024


class VoiceProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 503,
        code: str = "voice_provider_error",
    ):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


@dataclass(frozen=True)
class VoiceConfig:
    asr_base: str = ""
    asr_key: str = ""
    asr_model: str = ""
    tts_base: str = ""
    tts_key: str = ""
    tts_model: str = ""
    tts_voice: str = "alloy"
    timeout: float = 60.0

    @classmethod
    def from_env(cls) -> "VoiceConfig":
        return cls(
            asr_base=os.getenv("ASR_API_BASE", "").strip(),
            asr_key=os.getenv("ASR_API_KEY", "").strip(),
            asr_model=os.getenv("ASR_MODEL", "").strip(),
            tts_base=os.getenv("TTS_API_BASE", "").strip(),
            tts_key=os.getenv("TTS_API_KEY", "").strip(),
            tts_model=os.getenv("TTS_MODEL", "").strip(),
            tts_voice=os.getenv("TTS_VOICE", "alloy").strip() or "alloy",
            timeout=max(
                5.0, min(float(os.getenv("VOICE_TIMEOUT_SECONDS", "60") or 60), 180.0)
            ),
        )


def _endpoint(base: str, path: str) -> str:
    value = (base or "").rstrip("/")
    if value.endswith(path):
        return value
    return value + path


def _headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"} if key else {}


def _provider_message(response: httpx.Response, fallback: str) -> str:
    try:
        payload: Any = response.json()
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                return str(error.get("message") or fallback)
            return str(payload.get("message") or fallback)
    except Exception:
        pass
    return fallback


async def transcribe_audio(audio_bytes: bytes, filename: str, content_type: str) -> str:
    config = VoiceConfig.from_env()
    if not config.asr_base or not config.asr_key or not config.asr_model:
        raise VoiceProviderError("语音转文字服务尚未配置", code="asr_not_configured")
    if not audio_bytes:
        raise VoiceProviderError("录音内容为空", status_code=400, code="empty_audio")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise VoiceProviderError(
            "录音文件超过大小限制", status_code=413, code="audio_too_large"
        )
    if (
        content_type
        and content_type.split(";", 1)[0].lower() not in ALLOWED_AUDIO_TYPES
    ):
        raise VoiceProviderError(
            "不支持的音频格式", status_code=415, code="audio_type_not_supported"
        )
    try:
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            response = await client.post(
                _endpoint(config.asr_base, "/audio/transcriptions"),
                headers=_headers(config.asr_key),
                data={"model": config.asr_model},
                files={
                    "file": (
                        filename or "recording.webm",
                        audio_bytes,
                        content_type or "audio/webm",
                    )
                },
            )
    except httpx.HTTPError as exc:
        raise VoiceProviderError("语音转文字服务暂时不可用") from exc
    if response.status_code >= 400:
        raise VoiceProviderError(
            _provider_message(response, "语音转文字失败"), code="asr_failed"
        )
    try:
        payload = response.json()
    except Exception as exc:
        raise VoiceProviderError(
            "语音转文字返回格式错误", code="asr_invalid_response"
        ) from exc
    text = str(payload.get("text") if isinstance(payload, dict) else "").strip()
    if not text:
        raise VoiceProviderError("没有识别到可用语音内容", code="asr_empty_result")
    return text[:8000]


async def synthesize_speech(text: str, voice: str | None = None) -> tuple[bytes, str]:
    config = VoiceConfig.from_env()
    if not config.tts_base or not config.tts_key or not config.tts_model:
        raise VoiceProviderError("语音合成服务尚未配置", code="tts_not_configured")
    clean = str(text or "").strip()
    if not clean:
        raise VoiceProviderError("没有可播放的回答", status_code=400, code="empty_text")
    try:
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            response = await client.post(
                _endpoint(config.tts_base, "/audio/speech"),
                headers={
                    **_headers(config.tts_key),
                    "Content-Type": "application/json",
                },
                json={
                    "model": config.tts_model,
                    "voice": voice or config.tts_voice,
                    "input": clean[:16000],
                    "response_format": "mp3",
                },
            )
    except httpx.HTTPError as exc:
        raise VoiceProviderError("语音合成服务暂时不可用") from exc
    if response.status_code >= 400:
        raise VoiceProviderError(
            _provider_message(response, "语音合成失败"), code="tts_failed"
        )
    if not response.content:
        raise VoiceProviderError("语音合成返回空音频", code="tts_empty_result")
    return response.content, response.headers.get("content-type", "audio/mpeg").split(
        ";", 1
    )[0]


async def stream_speech(text: str, voice: str | None = None) -> AsyncIterator[bytes]:
    """Yield provider audio chunks as soon as the configured provider emits them.

    ``TTS_PROVIDER=aliyun_nls`` selects Alibaba NLS WebSocket synthesis. The
    default remains the existing OpenAI-compatible HTTP adapter for backwards
    compatibility; neither path silently falls back to the other.
    """
    if os.getenv("TTS_PROVIDER", "siliconflow").strip().lower() in {
        "aliyun",
        "aliyun_nls",
        "nls",
    }:
        from lightrag.product_realtime_voice import (
            AliyunNlsTtsConfig,
            stream_aliyun_speech,
        )

        config = AliyunNlsTtsConfig.from_env()
        if voice:
            config = AliyunNlsTtsConfig(**{**config.__dict__, "voice": voice})
        async for chunk in stream_aliyun_speech(text, config):
            yield chunk
        return

    config = VoiceConfig.from_env()
    if not config.tts_base or not config.tts_key or not config.tts_model:
        raise VoiceProviderError("语音合成服务尚未配置", code="tts_not_configured")
    clean = str(text or "").strip()
    if not clean:
        raise VoiceProviderError("没有可播放的回答", status_code=400, code="empty_text")
    try:
        timeout = httpx.Timeout(config.timeout, read=config.timeout)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                _endpoint(config.tts_base, "/audio/speech"),
                headers={
                    **_headers(config.tts_key),
                    "Content-Type": "application/json",
                },
                json={
                    "model": config.tts_model,
                    "voice": voice or config.tts_voice,
                    "input": clean[:16000],
                    "response_format": "mp3",
                    "stream": True,
                },
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    error = httpx.Response(response.status_code, content=body)
                    raise VoiceProviderError(
                        _provider_message(error, "语音合成失败"), code="tts_failed"
                    )
                emitted = False
                async for chunk in response.aiter_bytes():
                    if chunk:
                        emitted = True
                        yield chunk
                if not emitted:
                    raise VoiceProviderError(
                        "语音合成返回空音频", code="tts_empty_result"
                    )
    except VoiceProviderError:
        raise
    except httpx.HTTPError as exc:
        raise VoiceProviderError("语音合成服务暂时不可用") from exc
