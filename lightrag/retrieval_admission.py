"""各检索模式的准入规则。

raw_score 保留后端原值。ranking_score 只排序。
admission_score 是以该模式验证集门槛为中心的逻辑斯蒂概率，不是余弦本身。
概率达到 0.5 才返回。
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

_PROFILE_PATH = Path(__file__).with_name("retrieval_profiles.json")


@lru_cache(maxsize=1)
def _profiles() -> dict[str, Any]:
    return json.loads(_PROFILE_PATH.read_text(encoding="utf-8"))


_MODEL_PROFILES = (
    ("qwen3-embedding-4b", "qwen3-embedding-4b"),
    ("bge-m3", "bge-m3"),
)

_HYBRID_RATIO_KEYS = ("0", "0.25", "0.5", "0.75", "1")


def _app_env() -> str:
    return os.getenv("APP_ENV", "development").strip().lower()


def profile_name_for_model(model: str) -> str | None:
    key = (model or "").strip().lower()
    if not key:
        return None
    profiles = _profiles()
    for needle, name in _MODEL_PROFILES:
        if needle in key and name in profiles:
            return name
    for name in profiles:
        if name == "ci":
            continue
        if name.lower() in key:
            return name
    return None


def active_profile_name() -> str:
    """生产环境只接受与当前向量模型匹配的画像。开发和测试才允许回退到 ci。"""
    env = _app_env()
    forced = os.getenv("RETRIEVAL_PROFILE", "").strip()
    model = os.getenv("EMBEDDING_MODEL", "")
    profiles = _profiles()
    if forced:
        if env == "production" and (forced == "ci" or forced not in profiles):
            raise RuntimeError(
                "生产环境不能使用 ci 画像，也没有匹配的检索画像，拒绝启动"
            )
        return forced
    matched = profile_name_for_model(model)
    if matched:
        return matched
    if env == "production":
        shown = model.strip() or "未配置"
        raise RuntimeError(f"模型 {shown} 没有匹配的检索画像，拒绝启动")
    return "ci"


def retrieval_profile_startup_error() -> str | None:
    if _app_env() != "production":
        return None
    try:
        active_profile_name()
    except RuntimeError as exc:
        return str(exc)
    return None


def hybrid_ratio_key(ratio: float) -> str:
    value = min(1.0, max(0.0, float(ratio)))
    snapped = min((0.0, 0.25, 0.5, 0.75, 1.0), key=lambda item: abs(item - value))
    if snapped == 0:
        return "0"
    if snapped == 1:
        return "1"
    return (
        f"{snapped:.2f}".rstrip("0").rstrip(".")
        if snapped not in (0.25, 0.5, 0.75)
        else str(snapped)
    )


def mode_rule(
    mode: str, profile: str | None = None, ratio: float = 0.5
) -> dict[str, Any]:
    name = profile or active_profile_name()
    profiles = _profiles()
    if name not in profiles:
        if _app_env() == "production":
            raise RuntimeError(f"检索画像 {name} 不存在，拒绝启动")
        name = "ci"
    table = profiles[name]
    if mode not in table:
        if _app_env() == "production":
            raise RuntimeError(f"检索画像 {name} 没有模式 {mode}，拒绝启动")
        rule = dict(table["vector"])
    else:
        rule = dict(table[mode])
    floors = rule.get("floors")
    if mode == "hybrid" and isinstance(floors, dict) and floors:
        key = hybrid_ratio_key(ratio)
        if key not in floors:
            raise RuntimeError(f"检索画像 {name} 没有 hybrid 比例 {key} 的门槛")
        rule["floor"] = float(floors[key])
        rule["ratio"] = key
    return rule


def profile_binding(profile: str | None = None) -> dict[str, Any]:
    name = profile or active_profile_name()
    table = _profiles().get(name) or {}
    binding = table.get("binding")
    return dict(binding) if isinstance(binding, dict) else {}


def rerank_binding_matches(profile: str | None = None) -> bool:
    """换了重排模型、维度或供应商后，不能继续用上一份门槛。"""
    binding = profile_binding(profile)
    expected_model = str(binding.get("rerank_model") or "")
    if not expected_model:
        return True
    actual_model = os.getenv("RERANK_MODEL", "").strip()
    if actual_model and actual_model != expected_model:
        return False
    expected_dim = binding.get("embedding_dimension")
    actual_dim = os.getenv("EMBEDDING_DIM", "").strip()
    if (
        expected_dim not in (None, "")
        and actual_dim
        and str(expected_dim) != actual_dim
    ):
        return False
    expected_embed = str(binding.get("embedding_model") or "")
    actual_embed = os.getenv("EMBEDDING_MODEL", "").strip()
    if expected_embed and actual_embed and expected_embed != actual_embed:
        return False
    expected_provider = str(binding.get("provider") or "")
    actual_provider = os.getenv("RERANK_BINDING", "").strip()
    if expected_provider and actual_provider and expected_provider != actual_provider:
        return False
    return True


def admission_probability(
    raw: float | None, floor: float, temperature: float
) -> float | None:
    if raw is None:
        return None
    scale = temperature if temperature > 0 else 0.08
    return 1.0 / (1.0 + math.exp(-(float(raw) - float(floor)) / scale))


def signal_value(item: dict[str, Any], signal: str, ratio: float = 0.5) -> float | None:
    vector = item.get("vector_score")
    keyword = item.get("keyword_score")
    rerank = item.get("rerank_score")
    if signal == "rerank":
        return None if rerank is None else float(rerank)
    if signal == "keyword":
        return None if keyword is None else float(keyword)
    if signal == "blend":
        if vector is None and keyword is None:
            return None
        return float(ratio) * float(vector or 0) + (1 - float(ratio)) * float(
            keyword or 0
        )
    if signal == "max":
        values = [
            float(value) for value in (vector, keyword, rerank) if value is not None
        ]
        return max(values) if values else None
    if vector is None:
        return None
    return float(vector)


def annotate_admission(
    item: dict[str, Any], mode: str, ratio: float = 0.5
) -> dict[str, Any]:
    """写入 ranking_score 与 admission_score。没有信号时标记为不可信。"""
    selected = (
        "rerank"
        if item.get("reranked") and item.get("rerank_score") is not None
        else mode
    )
    if selected == "rerank" and not rerank_binding_matches():
        if _app_env() == "production":
            raise RuntimeError("重排模型与准入画像不一致，不能复用旧门槛")
        item["ranking_score"] = (
            None
            if item.get("ranking_score") is None
            else float(item.get("ranking_score") or item.get("score") or 0)
        )
        item["admission_score"] = None
        item["calibrated_score"] = None
        item["untrusted"] = True
        item["admission_signal"] = "rerank_rebinding_required"
        return item
    rule = mode_rule(selected, ratio=ratio)
    raw_signal = signal_value(item, str(rule.get("signal") or "vector"), ratio)
    probability = admission_probability(
        raw_signal, float(rule["floor"]), float(rule["temperature"])
    )
    ranking = item.get("ranking_score")
    if ranking is None:
        ranking = item.get("score")
    item["ranking_score"] = None if ranking is None else float(ranking)
    item["admission_score"] = probability
    item["admission_floor"] = float(rule["floor"])
    item["admission_signal"] = rule.get("signal")
    item["calibrated_score"] = probability
    item["untrusted"] = probability is None
    if probability is not None and item.get("ranking_score") is not None:
        item["score"] = float(item["ranking_score"])
    return item
