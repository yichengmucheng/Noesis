"""查询路由。精确查询和普通事实不调用模型。

HyDE 只用于候选召回，不能当作证据、引用或准入依据。
"""

from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable

MAX_VARIANTS = 3
MAX_SUBQUESTIONS = 4

_EXACT = re.compile(
    r"(\.pdf|\.docx|\.pptx|\.xlsx|\.txt|\.md|HT-\d|ZD-\d|INV-|JW-|AP-|POL-|"
    r"\d{4}\s*年|\d{4}-\d{2}-\d{2}|[A-Za-z]{1,8}-?\d{2,}[A-Za-z0-9-]*)",
    re.I,
)
_COMPARE = ("比较", "区别", "相比", "对比", "为什么", "如何影响", "导致", "因果", "关系", "跨文档", "以及")
_VAGUE = ("这个", "那个", "怎么办", "怎样", "相关内容", "大概", "有没有什么")
_CONCEPT = ("原理", "概念", "机制", "是什么意思", "如何理解")
_ANAPHORA = ("它", "他", "她", "这个", "那个", "上述", "刚才", "上面", "前面")
_NAME = re.compile(r"[\u4e00-\u9fff]{2,4}")

LlmFunc = Callable[[str], Awaitable[str] | str]


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _unique(items: list[str], limit: int) -> list[str]:
    seen: list[str] = []
    for item in items:
        text = _clean(item)
        if text and text not in seen:
            seen.append(text)
        if len(seen) >= limit:
            break
    return seen


def _looks_like_name(query: str) -> bool:
    if any(mark in query for mark in ("是谁", "叫什么", "姓名", "工号")):
        return bool(_NAME.search(query))
    return False


def _heuristic_subquestions(query: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"[和与及、]|以及|相比|比较", query) if part.strip()]
    if len(parts) >= 2:
        return _unique(parts, MAX_SUBQUESTIONS)
    return _unique([query + " 的直接事实", query + " 的关系或影响"], MAX_SUBQUESTIONS)


def _base(query: str, route: str, reason: str, *, variants: list[str] | None = None, subquestions: list[str] | None = None, hyde_text: str = "") -> dict[str, Any]:
    standalone = _clean(query)
    chosen = _unique([standalone, *(variants or [])], MAX_VARIANTS)
    if standalone not in chosen:
        chosen = _unique([standalone, *chosen], MAX_VARIANTS)
    return {
        "standalone_query": standalone,
        "route": route,
        "variants": chosen,
        "hyde_text": hyde_text if route == "hyde" else "",
        "subquestions": _unique(subquestions or [], MAX_SUBQUESTIONS),
        "reason": reason,
        "llm_used": False,
        "fallback_reason": "",
    }


def _parse_llm(raw: str, original: str) -> dict[str, Any] | None:
    text = str(raw or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    route = str(payload.get("route") or "")
    if route not in {"exact", "factual", "rewrite", "hyde", "subquestions"}:
        return None
    variants = payload.get("variants") if isinstance(payload.get("variants"), list) else []
    subquestions = payload.get("subquestions") if isinstance(payload.get("subquestions"), list) else []
    plan = _base(
        str(payload.get("standalone_query") or original),
        route,
        str(payload.get("reason") or "模型给出了结构化查询计划"),
        variants=[str(item) for item in variants],
        subquestions=[str(item) for item in subquestions],
        hyde_text=str(payload.get("hyde_text") or ""),
    )
    if original not in plan["variants"] and _clean(original):
        plan["variants"] = _unique([_clean(original), *plan["variants"]], MAX_VARIANTS)
    plan["llm_used"] = True
    return plan


async def _call_llm(llm: LlmFunc, prompt: str) -> str | None:
    import asyncio

    try:
        raw = llm(prompt)
        if asyncio.isfuture(raw) or asyncio.iscoroutine(raw):
            raw = await asyncio.wait_for(raw, timeout=8)
        return str(raw or "")
    except Exception:
        return None


def classify_route(query: str) -> str:
    text = _clean(query)
    if _EXACT.search(text) or _looks_like_name(text):
        return "exact"
    if any(mark in text for mark in _COMPARE):
        return "subquestions"
    if any(mark in text for mark in _VAGUE) and not any(mark in text for mark in _CONCEPT):
        return "rewrite"
    return "factual"


def needs_graph(plan: dict[str, Any], query: str) -> bool:
    if plan.get("route") == "subquestions":
        return True
    return any(mark in query for mark in ("为什么", "如何影响", "导致", "因果", "关系", "比较", "区别", "相比", "跨文档"))


def is_concept(query: str) -> bool:
    return any(mark in query for mark in _CONCEPT)


async def build_query_plan(
    query: str,
    *,
    history: list[str] | None = None,
    llm: LlmFunc | None = None,
    first_pass_weak: bool = False,
    force: str | None = None,
) -> dict[str, Any]:
    original = _clean(query)
    standalone = original
    coref_reason = ""
    if history and any(mark in original for mark in _ANAPHORA):
        topic = _clean(history[-1])
        if topic and topic not in standalone:
            standalone = _clean(f"{topic}。{original}")
            coref_reason = "会话指代已改写成独立问题"

    if force == "single":
        plan = _base(standalone, "factual", "消融：只保留原查询")
    elif force == "multi":
        plan = _base(standalone, "rewrite", "消融：原查询加两个改写", variants=[standalone + " 相关事实", standalone + " 关键内容"])
    elif force == "hyde":
        plan = _base(standalone, "hyde", "消融：HyDE 只参与召回", hyde_text="一段可能包含答案的说明：" + standalone)
    elif force == "subquestions":
        plan = _base(standalone, "subquestions", "消融：拆成子问题", subquestions=_heuristic_subquestions(standalone))
    else:
        route = classify_route(standalone)
        if route == "exact":
            plan = _base(standalone, "exact", "文件名、编号、型号、日期或人名，只保留原查询并走关键词召回")
        elif route == "subquestions":
            plan = _base(standalone, "subquestions", "比较、因果、关系或跨文档问题，拆成子问题", subquestions=_heuristic_subquestions(standalone))
        elif route == "rewrite":
            plan = _base(standalone, "rewrite", "表述模糊，在原查询之外最多再生成两个改写")
        elif first_pass_weak and is_concept(standalone):
            plan = _base(standalone, "hyde", "概念问题首轮召回弱，只追加一次 HyDE 召回")
        else:
            plan = _base(standalone, "factual", "普通事实，只运行原查询")
        needs_model = plan["route"] in {"rewrite", "hyde"} or (route == "subquestions" and llm is not None)
        if plan["route"] == "hyde" and llm is None:
            plan = _base(standalone, "factual", plan["reason"])
            plan["fallback_reason"] = "llm_unavailable"
        elif plan["route"] == "rewrite" and llm is None:
            plan["fallback_reason"] = "llm_unavailable"
            plan["variants"] = [standalone]
        elif needs_model and llm is not None:
            prompt = (
                "只返回 JSON，不要解释。字段：standalone_query, route, variants, hyde_text, subquestions, reason。"
                f"route 只能是 {plan['route']}。variants 最多 3 个且必须包含原始问题。subquestions 最多 4 个。"
                "hyde_text 只有 route=hyde 时填写，否则为空字符串。\n"
                f"原始问题：{original}\n独立问题：{standalone}"
            )
            raw = await _call_llm(llm, prompt)
            parsed = _parse_llm(raw or "", standalone)
            if parsed is None or parsed["route"] != plan["route"]:
                plan = _base(standalone, "factual", "模型输出无效，已退回原查询")
                plan["llm_used"] = bool(raw)
                plan["fallback_reason"] = "llm_invalid" if raw else "llm_unavailable"
            else:
                plan = parsed
                if plan["route"] != "hyde":
                    plan["hyde_text"] = ""
                plan["variants"] = [item for item in plan["variants"] if item != plan.get("hyde_text")]
                if standalone not in plan["variants"]:
                    plan["variants"] = _unique([standalone, *plan["variants"]], MAX_VARIANTS)
    if coref_reason:
        plan["reason"] = coref_reason + "。" + plan["reason"]
        plan["standalone_query"] = standalone
    if original and original not in plan["variants"] and not coref_reason:
        plan["variants"] = _unique([original, *plan["variants"]], MAX_VARIANTS)
    if coref_reason and standalone not in plan["variants"]:
        plan["variants"] = _unique([standalone, *plan["variants"]], MAX_VARIANTS)
    plan["variants"] = plan["variants"][:MAX_VARIANTS]
    plan["subquestions"] = plan["subquestions"][:MAX_SUBQUESTIONS]
    return plan
