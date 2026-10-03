# -*- coding: utf-8 -*-
import json
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

from lightrag.product_accounts import CURRENT_USER, LOCAL_OWNER_ID, visible_kbs
from lightrag.product_deletion import (
    apply_metadata_purge,
    begin_purge,
    mark_job,
    queue_retry,
)

sys.argv = ["lightrag"]
from lightrag.api.routers.product_shell import (
    ShellStore,
    _find_kb,
    scoped_graph_records,
)


def test_existing_kb_is_bound_to_migration_user(tmp_path: Path):
    path = tmp_path / "product_shell.json"
    path.write_text(
        json.dumps(
            {"kbs": [{"id": "a3-default", "name": "默认知识库"}], "file_bindings": {}}
        ),
        encoding="utf-8",
    )
    loaded = ShellStore(str(tmp_path)).load()
    assert loaded["kbs"][0]["owner_id"] == LOCAL_OWNER_ID
    assert loaded["device_sessions"] == []
    assert loaded["purge_jobs"] == []
    visible = visible_kbs(loaded, "someone-else")
    assert visible == []
    assert len(visible_kbs(loaded, LOCAL_OWNER_ID)) == 1


def test_other_user_gets_not_found():
    data = {"kbs": [{"id": "kb-b", "owner_id": "user-b"}]}
    token = CURRENT_USER.set({"user_id": "user-a"})
    try:
        with pytest.raises(HTTPException) as exc:
            _find_kb(data, "kb-b")
        assert exc.value.status_code == 404
    finally:
        CURRENT_USER.reset(token)


def test_graph_and_cache_do_not_cross_kb_files():
    bindings = {"a.txt": "kb-a", "b.txt": "kb-b"}
    nodes = [
        {"id": "甲", "file_path": "a.txt"},
        {"id": "乙", "file_path": "b.txt", "kb_id": "kb-b"},
    ]
    edges = [
        {"source": "甲", "target": "乙", "file_path": "b.txt"},
        {"source": "甲", "target": "甲", "file_path": "a.txt", "kb_id": "kb-a"},
    ]
    kept_nodes, kept_edges = scoped_graph_records(nodes, edges, "kb-a", bindings)
    assert [item["id"] for item in kept_nodes] == ["甲"]
    assert kept_edges[0]["file_path"] == "a.txt"
    assert all(
        "b.txt" not in str(item.get("file_path")) for item in kept_nodes + kept_edges
    )


def test_soft_delete_can_retry_after_failure():
    data = {
        "kbs": [{"id": "kb-a", "owner_id": "user-a"}],
        "file_bindings": {"a.txt": "kb-a", "b.txt": "kb-b"},
        "qa_pairs": [{"id": "q1", "kb_id": "kb-a"}, {"id": "q2", "kb_id": "kb-b"}],
        "comparisons": [],
        "audits": {},
        "sessions": {},
        "purge_jobs": [],
    }
    job = begin_purge(data, data["kbs"][0])
    assert data["kbs"][0]["deleted_at"]
    assert visible_kbs(data, "user-a") == []
    cache = apply_metadata_purge(
        data, "kb-a", [{"kb_id": "kb-a", "answer": "旧答案"}, {"kb_id": "kb-b"}]
    )
    assert data["file_bindings"] == {"b.txt": "kb-b"}
    assert data["qa_pairs"] == [{"id": "q2", "kb_id": "kb-b"}]
    assert cache == [{"kb_id": "kb-b"}]
    mark_job(job, ok=False, error="清理中断", leftover=["a.txt"])
    assert job["status"] == "failed"
    assert job["attempts"] == 1
    queue_retry(job)
    assert job["status"] == "pending"
    mark_job(job, ok=True, error="", leftover=[])
    assert job["status"] == "succeeded"
    with pytest.raises(ValueError):
        queue_retry(job)
