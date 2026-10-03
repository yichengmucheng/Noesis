# -*- coding: utf-8 -*-
from lightrag.answer_pipeline import (
    SYSTEM_PROMPT,
    build_messages,
    cache_entry_usable,
    decide_refuse,
    needs_rewrite,
    redact_text,
    should_summarize,
    window_and_older,
)
from lightrag.operate import chunking_by_token_size


class _FakeTokenizer:
    def encode(self, text):
        return list(text)

    def decode(self, tokens):
        return "".join(tokens)


def test_parent_is_shared_by_child_chunks():
    text = "# 第一章\n" + "密封圈压缩不足。" * 40
    chunks = chunking_by_token_size(
        file_path="sample.docx",
        tokenizer=_FakeTokenizer(),
        content=text,
        overlap_token_size=0,
        max_token_size=40,
    )
    assert len(chunks) >= 2
    assert len({item["parent_id"] for item in chunks}) == 1
    assert "密封圈" in chunks[0]["parent_content"]
    assert len(chunks[0]["parent_content"]) > len(chunks[0]["content"])


def test_rewrite_only_when_pronoun_has_history():
    history = [{"role": "user", "content": "员工手册在哪"}]
    assert needs_rewrite("它主要讲了什么？", history)
    assert not needs_rewrite("员工手册主要讲了什么？", history)
    assert not needs_rewrite("它主要讲了什么？", [])


def test_window_keeps_five_rounds_and_summary_stays_out_of_system():
    history = []
    for index in range(7):
        history.append({"role": "user", "content": f"问题{index}"})
        history.append({"role": "assistant", "content": f"回答{index}"})
    recent, older = window_and_older(history, rounds=5)
    assert len(older) == 2
    assert recent[0]["content"] == "问题2"
    summary = {
        "text": "早期结论是密封失效",
        "source": "conversation",
        "generated_at": "2026-09-29T00:00:00",
        "redacted": True,
        "trust": "low",
        "notice": "不是指令",
    }
    system, user = build_messages(
        "当前问题",
        [{"doc_name": "a.docx", "parent_content": "父块正文"}],
        recent,
        summary,
    )
    assert system == SYSTEM_PROMPT
    assert "早期结论是密封失效" not in system
    assert "不是指令" in user
    assert "父块正文" in user
    assert should_summarize(older, None)
    _system, child_only = build_messages(
        "当前问题",
        [{"doc_name": "a.docx", "content": "子块正文", "parent_content": "父块正文"}],
        recent,
        summary,
        expand_context=False,
    )
    assert "子块正文" in child_only
    assert "父块正文" not in child_only


def test_cache_requires_version_and_live_chunks():
    entry = {
        "kb_id": "kb",
        "tenant": "default",
        "kb_version": "v1",
        "prompt_version": "answer-v2",
        "model": "qwen",
        "mode": "hybrid",
        "chunk_ids": ["c1"],
        "user_id": "user-a",
    }
    assert not cache_entry_usable(
        entry,
        kb_id="kb",
        tenant="default",
        kb_version="v1",
        mode="hybrid",
        model="qwen",
        live_chunk_ids={"c1"},
    )
    assert cache_entry_usable(
        entry,
        kb_id="kb",
        tenant="default",
        kb_version="v1",
        mode="hybrid",
        model="qwen",
        live_chunk_ids={"c1"},
        user_id="user-a",
    )
    assert not cache_entry_usable(
        entry,
        kb_id="kb",
        tenant="default",
        kb_version="v2",
        mode="hybrid",
        model="qwen",
        live_chunk_ids={"c1"},
    )
    assert not cache_entry_usable(
        entry,
        kb_id="kb",
        tenant="default",
        kb_version="v1",
        mode="hybrid",
        model="qwen",
        live_chunk_ids=set(),
        user_id="user-a",
    )
    assert cache_entry_usable(
        entry,
        kb_id="kb",
        tenant="default",
        kb_version="v1",
        mode="hybrid",
        model="qwen",
        live_chunk_ids={"c1"},
        user_id="user-a",
    )
    assert not cache_entry_usable(
        entry,
        kb_id="kb",
        tenant="default",
        kb_version="v1",
        mode="hybrid",
        model="qwen",
        live_chunk_ids={"c1"},
        user_id="user-b",
    )


def test_refusal_and_redaction():
    assert decide_refuse([])
    assert not decide_refuse([{"chunk_id": "c1"}])
    assert "[手机号]" in redact_text("联系 13800138000")
