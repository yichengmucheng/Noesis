# -*- coding: utf-8 -*-
import sys

sys.argv = ["lightrag"]

from lightrag.api.routers.product_shell import (
    _chunk_mentions,
    _kb_filenames,
    _record_in_kb,
    _relation_labels,
)


def test_entity_scope_follows_current_kb():
    bindings = {"a.docx": "a3-default", "b.docx": "other-kb"}
    mixed = r"docs\a.docx<SEP>docs/b.docx"
    assert _kb_filenames(mixed, "a3-default", bindings) == ["a.docx"]
    assert _kb_filenames(mixed, "other-kb", bindings) == ["b.docx"]
    assert _record_in_kb("", "a3-default", bindings) is False
    assert _record_in_kb("", "other-kb", bindings) is False
    assert _record_in_kb("docs/b.docx", "a3-default", bindings) is False


def test_frequency_counts_distinct_chunks():
    assert _chunk_mentions("") == []
    assert _chunk_mentions("c1<SEP>c1<SEP> c2 ") == ["c1", "c2"]


def test_relation_labels_split_separators():
    assert _relation_labels("导致<SEP>属于") == ["导致", "属于"]
    assert _relation_labels("采取措施，验证于") == ["采取措施", "验证于"]
