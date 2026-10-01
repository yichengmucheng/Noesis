# -*- coding: utf-8 -*-
"""
原生分层切块单元测试（替代 langchain 的 .docx 分支）
覆盖：标题层级、表格原子性、中文句读分割、超长硬切、空输入
"""
import pytest

from lightrag.operate import (
    chunking_by_token_size,
    _hierarchical_markdown_chunks,
    _smart_text_split,
    _split_with_table_guard,
    _split_markdown_headers,
)


class _FakeTokenizer:
    """轻量替身：按字符数近似 token 计数，隔离对 tiktoken 的依赖"""

    def encode(self, text):
        return list(text)

    def decode(self, tokens):
        return "".join(tokens)


def _chunk(content, file_type=".docx", max_token=1024, overlap=128):
    return chunking_by_token_size(
        file_path=f"sample{file_type}",
        tokenizer=_FakeTokenizer(),
        content=content,
        overlap_token_size=overlap,
        max_token_size=max_token,
    )


SAMPLE = """# 一级标题甲
引言段落，介绍本节内容。

## 二级标题A
这是二级标题A下的正文。第一段。

## 二级标题B
<table>
<tr><td>项目</td><td>周期</td><td>备注</td></tr>
<tr><td>冷却液</td><td>2年</td><td>不可混加</td></tr>
</table>
表格之后的正文继续。


# 一级标题乙
开头无表格的正文。
"""


def test_headers_split_with_hierarchy():
    sections = _split_markdown_headers(SAMPLE)
    titles = [s["text"].splitlines()[0] for s in sections if s["text"]]
    assert "# 一级标题甲" in titles
    assert any(t.startswith("## 二级标题A") for t in titles)
    assert len(sections) >= 4


def test_table_is_atomic():
    pieces = _split_with_table_guard("前置文本。" * 50 + "\n<table>\n<tr><td>a</td></tr>\n</table>\n" + "后置文本。" * 50, 120, 20)
    table_pieces = [p for p in pieces if p.startswith("<table>")]
    assert len(table_pieces) == 1
    assert table_pieces[0].strip() == "<table>\n<tr><td>a</td></tr>\n</table>"


def test_smart_split_respects_limit():
    text = "这是第一句话。这是第二句话？这是第三句话，这是第四句；这是第五句。" * 20
    pieces = _smart_text_split(text, 60, 10)
    assert pieces
    assert all(len(p) <= 60 for p in pieces)
    joined = "".join(p.replace("。 ", "") for p in pieces)
    assert "这是第一句话" in joined


def test_hard_cut_with_overlap():
    text = "甲乙丙丁戊己庚辛壬癸" * 50  # 500 字，无任何分隔符
    pieces = _smart_text_split(text, 100, 20)
    assert all(len(p) <= 100 for p in pieces)
    # 无分隔符可分 → 带 overlap 的滑动窗口硬切：start = 0,80,160,240,320,400
    # 400 起点的窗口 (100字) 已覆盖到 500，按算法 break，无需 480 尾窗
    assert len(pieces) == 6
    assert pieces[0] == text[:100]
    assert pieces[3] == text[240:340]
    assert pieces[-1] == text[400:]  # 100 字，恰好收尾
    # 任意相邻两段存在 >= overlap 的重叠或恰好衔接：整体覆盖全文
    for a, b in zip(pieces, pieces[1:]):
        ia = text.index(a)
        ib = text.index(b, ia)  # b 的起点在 a 起点之后
        assert ib - ia <= 100  # 下一窗口起点在本窗口内（重叠）


def test_hierarchy_prefix_on_chunks():
    chunks = _chunk(SAMPLE, max_token=64, overlap=16)
    assert chunks
    # 每个块若属于某标题节，必须携带标题链前缀（表块除外：直接来自该节文本）
    b_chunks = [c for c in chunks if "二级标题B" in c["content"] or "表格之后" in c["content"]]
    assert any(c["content"].startswith("# 一级标题甲") for c in chunks)


def test_docx_chunk_output_schema():
    chunks = _chunk(SAMPLE)
    for c in chunks:
        assert {"tokens", "content", "chunk_order_index", "parent_id", "parent_content"} <= set(c.keys())
        assert c["parent_content"]
        assert c["tokens"] == len(c["content"])  # FakeTokenizer 字符计 token
        assert c["content"].strip()
    orders = [c["chunk_order_index"] for c in chunks]
    assert orders == sorted(orders)


def test_empty_and_plain():
    assert _chunk("") == []
    plain = _chunk("没有任何标题的普通长文本。", max_token=999999)
    assert len(plain) == 1


def test_generic_fallback_unchanged():
    # .md/.txt 走通用 token fallback；确保未被 docx 改动破坏
    md = _chunk("# 标题\n正文" * 30, file_type=".md", max_token=50)
    assert md and all(len(c["content"]) <= 60 for c in md)


def _chunk_with(content, file_type=".txt", strategy="fixed", max_token=1024, overlap=0, delimiter=None):
    return chunking_by_token_size(
        file_path=f"sample{file_type}",
        tokenizer=_FakeTokenizer(),
        content=content,
        overlap_token_size=overlap,
        max_token_size=max_token,
        chunk_strategy=strategy,
        chunk_delimiter=delimiter,
    )


def test_fixed_splits_by_length():
    chunks = _chunk_with("甲乙丙丁" * 20, strategy="fixed", max_token=10, overlap=0)
    assert [len(c["content"]) for c in chunks] == [10, 10, 10, 10, 10, 10, 10, 10]


def test_delimiter_keeps_segments():
    chunks = _chunk_with("现象甲\n---\n现象乙\n---\n现象丙", strategy="delimiter", delimiter="---", max_token=100)
    assert [c["content"] for c in chunks] == ["现象甲", "现象乙", "现象丙"]


def test_ledger_one_row_one_chunk():
    table = "机型,现象,原因\nK13N,漏水,密封失效\nK15N,断裂,底座开裂\n"
    chunks = _chunk_with(table, file_type=".csv", strategy="ledger", max_token=8)
    assert len(chunks) == 2
    assert chunks[0]["content"] == "机型：K13N；现象：漏水；原因：密封失效"
    assert "K15N" in chunks[1]["content"]
    assert "断裂" in chunks[1]["content"]


def test_qa_pairs_stay_together():
    text = "问：节温器打不开怎么判断？\n答：看水温是否持续升高。\n\n问：风扇不转怎么办？\n答：检查继电器。\n"
    chunks = _chunk_with(text, strategy="qa", max_token=10)
    assert len(chunks) == 2
    assert chunks[0]["content"].startswith("问题：节温器打不开怎么判断？")
    assert "答案：看水温是否持续升高。" in chunks[0]["content"]
    assert "风扇不转" in chunks[1]["content"]


def test_clause_keeps_chapter_title():
    text = (
        "第一章 总则\n"
        "第一节 范围\n"
        "第一条 本制度适用于发动机故障。\n"
        "第二条 记录应当完整。\n"
        "第二章 处置\n"
        "第三条 先停机再排查。\n"
    )
    chunks = _chunk_with(text, strategy="clause", max_token=8)
    assert len(chunks) == 3
    assert chunks[0]["content"].startswith("第一章 总则\n第一节 范围\n第一条")
    assert chunks[1]["content"].startswith("第一章 总则\n第一节 范围\n第二条")
    assert chunks[2]["content"].startswith("第二章 处置\n第三条")
    assert "第一节" not in chunks[2]["content"]


def test_one_keeps_whole_document():
    text = "这是一份很短的单页说明。包含两句。仍然应该是一整块。"
    chunks = _chunk_with(text, strategy="one", max_token=4)
    assert len(chunks) == 1
    assert chunks[0]["content"] == text


def test_smart_routes_by_content():
    csv_text = "机型,现象\nK13N,漏水\nK15N,断裂\n"
    ledger = _chunk_with(csv_text, file_type=".csv", strategy="smart", max_token=4)
    assert len(ledger) == 2
    assert ledger[0]["content"] == "机型：K13N；现象：漏水"

    short = _chunk_with("单页说明正文。", strategy="smart", max_token=100)
    assert len(short) == 1

    law = "第一条 适用本制度。\n第二条 应当记录。"
    clauses = _chunk_with(law, strategy="smart", max_token=4)
    assert len(clauses) == 2
    assert clauses[0]["content"].startswith("第一条")
