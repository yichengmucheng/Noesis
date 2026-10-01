# -*- coding: utf-8 -*-
"""任务持久化、租约、重试和上传接口。使用临时目录，不碰 9621 的开发数据。"""
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.argv = ["lightrag"]

from lightrag.product_db import SqliteStore, public_job
from lightrag.product_deletion import begin_purge
from lightrag.product_ingest import content_key, reset_stage_failures
from lightrag.product_scope import scope_visible
from lightrag.product_storage import _graph, _kv, _save_kv, _vdb_rows, load_shell, save_shell
from lightrag.product_storage_check import repair_safe, scan
from lightrag.product_uploads import allocate_upload
from lightrag.product_worker import import_pending_purges, serve, worker_loop

ROOT = Path(__file__).resolve().parents[1]


def _bind(monkeypatch, working: Path, inputs: Path, **extra: str) -> None:
    monkeypatch.setenv("WORKING_DIR", str(working))
    monkeypatch.setenv("INPUT_DIR", str(inputs))
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("PRODUCT_AUTH", "0")
    monkeypatch.setenv("WORKER_POLL_INTERVAL", "0.2")
    monkeypatch.setenv("JOB_RETRY_BACKOFF", "0")
    monkeypatch.setenv("JOB_LEASE_SECONDS", "30")
    monkeypatch.setenv("JOB_HEARTBEAT_SECONDS", "10")
    monkeypatch.setenv("JOB_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("MAX_CONCURRENT_JOBS_PER_USER", "2")
    monkeypatch.setenv("MAX_CONCURRENT_JOBS_PER_KB", "2")
    monkeypatch.setenv("INGEST_RUNTIME", "lightrag.testing_index_runtime:runtime")
    for name in ("JOB_FAIL_STAGE", "JOB_FAIL_TIMES", "JOB_FAIL_CODE", "JOB_PAUSE_STAGE", "JOB_PAUSE_SECONDS", "JOB_FORCE_CONSISTENCY_FAIL", "WORKER_MAX_JOBS"):
        monkeypatch.delenv(name, raising=False)
    for key, value in extra.items():
        monkeypatch.setenv(key, value)
    reset_stage_failures()


def _workspace(tmp_path: Path, user: str = "user-a", kb: str = "kb-a"):
    working = tmp_path / "rag"
    inputs = tmp_path / "inputs"
    working.mkdir()
    inputs.mkdir()
    data = load_shell(working)
    data["kbs"] = [{"id": kb, "owner_id": user, "name": kb, "settings": {}, "graph_config": {}}]
    save_shell(working, data)
    store = SqliteStore(working / "product_jobs.sqlite")
    return working, inputs, store


def _queue(store, working: Path, inputs: Path, text: str, user="user-a", kb="kb-a", name="note.txt"):
    data = load_shell(working)
    record = allocate_upload(data, user, kb, name)
    record["status"] = "queued"
    dest = inputs / record["storage_key"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = text.encode("utf-8")
    dest.write_bytes(payload)
    save_shell(working, data)
    job = store.create_job({
        "job_type": "ingestion",
        "user_id": user,
        "kb_id": kb,
        "doc_id": record["doc_id"],
        "file_name": name,
        "file_path": record["storage_key"],
        "idempotency_key": content_key(user, kb, payload),
        "input_snapshot": {"owner_id": user, "storage_key": record["storage_key"]},
    })
    return job


def _counts(working: Path, doc_id: str) -> tuple[int, int, int]:
    chunks = sum(1 for row in _kv(working, "text_chunks").values() if str(row.get("doc_id") or "") == doc_id)
    vectors = sum(1 for row in _vdb_rows(working / "vdb_chunks.json") if str(row.get("doc_id") or "") == doc_id)
    graph, _path = _graph(working)
    edges = sum(1 for _src, _tgt, attrs in graph.edges(data=True) if str(attrs.get("doc_id") or "") == doc_id)
    return chunks, vectors, edges


def _hidden(working: Path, doc_id: str, kb: str) -> bool:
    data = load_shell(working)
    record = {"doc_id": doc_id, "kb_id": kb, "content": "半成品"}
    return scope_visible(record, kb, data.get("file_bindings"), data.get("doc_index")) is False


def _drive(store, worker_id: str, stop: threading.Event) -> threading.Thread:
    thread = threading.Thread(target=worker_loop, args=(store, worker_id, stop), daemon=True)
    thread.start()
    return thread


def _wait_status(store, job_id: str, statuses: set[str], timeout: float = 20):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = store.get_job(job_id)
        if last and last["status"] in statuses:
            return last
        time.sleep(0.05)
    return last


def test_same_idempotency_key_returns_the_existing_job(tmp_path, monkeypatch):
    working, inputs, store = _workspace(tmp_path)
    _bind(monkeypatch, working, inputs)
    payload = "同一份内容".encode("utf-8")
    fields = {
        "job_type": "ingestion",
        "user_id": "user-a",
        "kb_id": "kb-a",
        "doc_id": "doc-1",
        "file_name": "a.txt",
        "idempotency_key": content_key("user-a", "kb-a", payload),
    }
    first = store.create_job(fields)
    second = store.create_job({**fields, "job_id": "another", "doc_id": "doc-1"})
    assert first["job_id"] == second["job_id"]
    assert store.list_jobs(user_id="user-a")["total"] == 1
    assert store.list_jobs(user_id="user-b")["total"] == 0
    assert store.request_cancel(first["job_id"], "user-b") is None
    leaked = public_job({**first, "error_message": "traceback sk-secret C:\\secret\\file"})
    assert leaked["error_message"] == "处理失败"
    assert "sk-" not in leaked["error_message"]
    assert "file_path" not in leaked


def test_worker_processes_queued_job_without_duplicate_outputs(tmp_path, monkeypatch):
    working, inputs, store = _workspace(tmp_path)
    _bind(monkeypatch, working, inputs)
    job = _queue(store, working, inputs, "第一段\n\n第二段")
    assert job["status"] == "queued"
    stop = threading.Event()
    thread = _drive(store, "worker-a", stop)
    done = _wait_status(store, job["job_id"], {"succeeded"})
    stop.set()
    thread.join(5)
    assert done["status"] == "succeeded"
    assert done["stage"] == "ready"
    chunks, vectors, edges = _counts(working, job["doc_id"])
    assert (chunks, vectors, edges) == (2, 2, 0)
    data = load_shell(working)
    assert data["doc_index"][job["doc_id"]]["status"] == "ready"
    assert scope_visible(
        {"doc_id": job["doc_id"], "kb_id": "kb-a", "content": "第一段"},
        "kb-a",
        data["file_bindings"],
        data["doc_index"],
    )


def test_stage_failures_retry_without_duplicating_records(tmp_path, monkeypatch):
    working, inputs, store = _workspace(tmp_path)
    _bind(monkeypatch, working, inputs, JOB_FAIL_TIMES="1")
    for stage, expected in (("parsing", (2, 2, 0)), ("embedding", (2, 2, 0)), ("graphing", (2, 2, 0))):
        reset_stage_failures()
        monkeypatch.setenv("JOB_FAIL_STAGE", stage)
        text = f"{stage} 段落一\n\n{stage} 段落二"
        job = _queue(store, working, inputs, text, name=f"{stage}.txt")
        stop = threading.Event()
        thread = _drive(store, f"worker-{stage}", stop)
        done = _wait_status(store, job["job_id"], {"succeeded", "failed"})
        stop.set()
        thread.join(5)
        assert done["status"] == "succeeded", done
        assert _counts(working, job["doc_id"]) == expected


def test_cancel_timeout_and_failed_check_are_not_searchable(tmp_path, monkeypatch):
    working, inputs, store = _workspace(tmp_path)
    _bind(monkeypatch, working, inputs, JOB_PAUSE_STAGE="parsing", JOB_PAUSE_SECONDS="1.2")
    cancelled = _queue(store, working, inputs, "取消这段\n\n不要入库", name="cancel.txt")
    stop = threading.Event()
    thread = _drive(store, "worker-cancel", stop)
    deadline = time.time() + 10
    while time.time() < deadline and not (working / "stage.log").exists():
        time.sleep(0.05)
    store.request_cancel(cancelled["job_id"], "user-a")
    done = _wait_status(store, cancelled["job_id"], {"cancelled"})
    stop.set()
    thread.join(5)
    assert done["status"] == "cancelled"
    assert _counts(working, cancelled["doc_id"]) == (0, 0, 0)
    assert _hidden(working, cancelled["doc_id"], "kb-a")

    monkeypatch.delenv("JOB_PAUSE_STAGE", raising=False)
    monkeypatch.setenv("STAGE_PARSE_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("JOB_PAUSE_STAGE", "parsing")
    monkeypatch.setenv("JOB_PAUSE_SECONDS", "3")
    timed = _queue(store, working, inputs, "超时这段\n\n不要入库", name="timeout.txt")
    stop = threading.Event()
    thread = _drive(store, "worker-timeout", stop)
    done = _wait_status(store, timed["job_id"], {"timeout", "failed"})
    stop.set()
    thread.join(5)
    assert done["status"] == "timeout"
    assert _hidden(working, timed["doc_id"], "kb-a")

    monkeypatch.delenv("JOB_PAUSE_STAGE", raising=False)
    monkeypatch.setenv("JOB_FORCE_CONSISTENCY_FAIL", "1")
    checked = _queue(store, working, inputs, "校验失败\n\n不可检索", name="check.txt")
    stop = threading.Event()
    thread = _drive(store, "worker-check", stop)
    done = _wait_status(store, checked["job_id"], {"consistency_failed"})
    stop.set()
    thread.join(5)
    assert done["status"] == "consistency_failed"
    assert done["error_code"] == "consistency"
    assert load_shell(working)["doc_index"][checked["doc_id"]]["status"] == "consistency_failed"
    assert _hidden(working, checked["doc_id"], "kb-a")


def test_stage_timeout_drops_partials_and_blocks_late_writes(tmp_path, monkeypatch):
    working, inputs, store = _workspace(tmp_path)
    _bind(
        monkeypatch,
        working,
        inputs,
        STAGE_EMBED_TIMEOUT_SECONDS="1",
        JOB_PAUSE_STAGE="embedding",
        JOB_PAUSE_SECONDS="3",
    )
    job = _queue(store, working, inputs, "节温器导致水温升高", name="late.txt")
    stop = threading.Event()
    thread = _drive(store, "worker-late-embed", stop)
    deadline = time.time() + 10
    ir = working / "document_ir" / f"{job['doc_id']}.json"
    while time.time() < deadline and not (working / "stage.log").exists():
        time.sleep(0.05)
    assert ir.is_file()
    done = _wait_status(store, job["job_id"], {"timeout"})
    stop.set()
    thread.join(5)
    assert done["status"] == "timeout"
    time.sleep(3)
    assert not ir.exists()
    assert _counts(working, job["doc_id"]) == (0, 0, 0)
    if (working / "vdb_chunks.json").exists():
        assert all(row.get("doc_id") != job["doc_id"] for row in _vdb_rows(working / "vdb_chunks.json"))
    if (working / "graph_chunk_entity_relation.graphml").exists():
        graph, _path = _graph(working)
        assert all(job["doc_id"] not in str(attrs.get("doc_id") or "") for _node, attrs in graph.nodes(data=True))


def test_expired_lease_can_be_taken_over_and_restart_recovers(tmp_path, monkeypatch):
    working, inputs, store = _workspace(tmp_path)
    _bind(monkeypatch, working, inputs, JOB_LEASE_SECONDS="2", JOB_HEARTBEAT_SECONDS="30")
    job = _queue(store, working, inputs, "租约接管\n\n继续")
    held = store.acquire("worker-old")
    assert held["status"] == "running"
    store._exec(
        "UPDATE product_jobs SET lease_until = ? WHERE job_id = ?",
        ("2000-01-01T00:00:00+00:00", held["job_id"]),
    )
    assert store.recover_expired() == 1
    recovered = store.get_job(held["job_id"])
    assert recovered["status"] == "queued"
    assert recovered["worker_id"] == ""
    stop = threading.Event()
    thread = _drive(store, "worker-new", stop)
    done = _wait_status(store, held["job_id"], {"succeeded"})
    stop.set()
    thread.join(5)
    assert done["status"] == "succeeded"
    assert done["worker_id"] == ""


def test_graceful_stop_releases_the_lease(tmp_path, monkeypatch):
    working, inputs, store = _workspace(tmp_path)
    _bind(monkeypatch, working, inputs, JOB_PAUSE_STAGE="parsing", JOB_PAUSE_SECONDS="1.5")
    job = _queue(store, working, inputs, "优雅退出\n\n释放")
    stop = threading.Event()
    thread = _drive(store, "worker-stop", stop)
    deadline = time.time() + 10
    while time.time() < deadline and not (working / "stage.log").exists():
        time.sleep(0.05)
    stop.set()
    thread.join(8)
    released = store.get_job(job["job_id"])
    assert released["status"] == "queued"
    assert released["worker_id"] == ""
    assert released["lease_until"] == ""


def test_two_workers_do_not_run_the_same_job(tmp_path, monkeypatch):
    working, inputs, store = _workspace(tmp_path)
    _bind(monkeypatch, working, inputs, JOB_PAUSE_STAGE="parsing", JOB_PAUSE_SECONDS="0.8")
    job = _queue(store, working, inputs, "只跑一次\n\n不要并行")
    stop = threading.Event()
    first = _drive(store, "worker-1", stop)
    second = _drive(store, "worker-2", stop)
    done = _wait_status(store, job["job_id"], {"succeeded"})
    stop.set()
    first.join(5)
    second.join(5)
    assert done["status"] == "succeeded"
    log = (working / "stage.log").read_text(encoding="utf-8").strip().splitlines()
    assert log == ["parsing"]
    assert _counts(working, job["doc_id"]) == (2, 2, 0)


def test_per_user_concurrency_limit(tmp_path, monkeypatch):
    working, inputs, store = _workspace(tmp_path)
    _bind(
        monkeypatch,
        working,
        inputs,
        MAX_CONCURRENT_JOBS_PER_USER="1",
        JOB_PAUSE_STAGE="parsing",
        JOB_PAUSE_SECONDS="1.2",
    )
    first = _queue(store, working, inputs, "先处理\n\n甲", name="one.txt")
    second = _queue(store, working, inputs, "后处理\n\n乙", name="two.txt")
    stop = threading.Event()
    left = _drive(store, "worker-left", stop)
    right = _drive(store, "worker-right", stop)
    deadline = time.time() + 8
    saw_queue = False
    while time.time() < deadline:
        jobs = [store.get_job(first["job_id"]), store.get_job(second["job_id"])]
        states = {item["status"] for item in jobs}
        if "running" in states and "queued" in states:
            saw_queue = True
            break
        time.sleep(0.05)
    assert saw_queue
    monkeypatch.delenv("JOB_PAUSE_STAGE", raising=False)
    done_first = _wait_status(store, first["job_id"], {"succeeded"})
    done_second = _wait_status(store, second["job_id"], {"succeeded"})
    stop.set()
    left.join(5)
    right.join(5)
    assert done_first["status"] == done_second["status"] == "succeeded"


def test_upload_and_delete_jobs_do_not_clobber(tmp_path, monkeypatch):
    working, inputs, store = _workspace(tmp_path)
    _bind(monkeypatch, working, inputs)
    data = load_shell(working)
    data["kbs"].append({"id": "kb-b", "owner_id": "user-a", "name": "kb-b"})
    save_shell(working, data)
    upload = _queue(store, working, inputs, "保留下来\n\n甲库")
    data = load_shell(working)
    kb = next(item for item in data["kbs"] if item["id"] == "kb-b")
    shell_job = begin_purge(data, kb)
    save_shell(working, data)
    purge = store.create_job({
        "job_id": shell_job["job_id"],
        "job_type": "purge",
        "user_id": "user-a",
        "kb_id": "kb-b",
        "doc_id": "",
        "idempotency_key": f"purge:kb-b:{shell_job['job_id']}",
        "input_snapshot": {"kb_id": "kb-b"},
    })
    stop = threading.Event()
    left = _drive(store, "worker-upload", stop)
    right = _drive(store, "worker-delete", stop)
    uploaded = _wait_status(store, upload["job_id"], {"succeeded", "failed"})
    deleted = _wait_status(store, purge["job_id"], {"succeeded", "failed", "consistency_failed"})
    stop.set()
    left.join(5)
    right.join(5)
    assert uploaded["status"] == "succeeded"
    assert deleted["status"] == "succeeded"
    assert load_shell(working)["doc_index"][upload["doc_id"]]["status"] == "ready"
    assert _counts(working, upload["doc_id"]) == (2, 2, 0)


def test_delete_is_idempotent_and_crash_handoff_uses_a_real_process(tmp_path, monkeypatch):
    working, inputs, store = _workspace(tmp_path)
    _bind(monkeypatch, working, inputs)
    job = _queue(store, working, inputs, "崩溃后继续\n\n第二段")
    env = os.environ.copy()
    for key in list(env):
        if any(word in key.upper() for word in ("KEY", "SECRET", "TOKEN", "PASSWORD", "COOKIE")):
            env.pop(key, None)
    env.update({
        "WORKING_DIR": str(working),
        "INPUT_DIR": str(inputs),
        "DATABASE_URL": "",
        "APP_ENV": "development",
        "PRODUCT_AUTH": "0",
        "WORKER_COUNT": "1",
        "WORKER_POLL_INTERVAL": "0.2",
        "JOB_LEASE_SECONDS": "2",
        "JOB_HEARTBEAT_SECONDS": "30",
        "JOB_PAUSE_STAGE": "parsing",
        "JOB_PAUSE_SECONDS": "8",
        "PYTHONPATH": str(ROOT),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    })
    log = tmp_path / "worker.log"
    with log.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            [sys.executable, "-m", "lightrag.product_worker"],
            cwd=str(ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    deadline = time.time() + 20
    while time.time() < deadline and not (working / "stage.log").exists():
        if proc.poll() is not None:
            break
        time.sleep(0.05)
    proc.kill()
    proc.wait(timeout=10)
    time.sleep(3)
    follow = env.copy()
    follow.pop("JOB_PAUSE_STAGE", None)
    follow.pop("JOB_PAUSE_SECONDS", None)
    follow["WORKER_MAX_JOBS"] = "1"
    follow["JOB_LEASE_SECONDS"] = "30"
    with log.open("a", encoding="utf-8") as handle:
        done_proc = subprocess.run(
            [sys.executable, "-m", "lightrag.product_worker"],
            cwd=str(ROOT),
            env=follow,
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=40,
        )
    assert done_proc.returncode == 0
    finished = store.get_job(job["job_id"])
    assert finished["status"] == "succeeded"
    assert _counts(working, job["doc_id"]) == (2, 2, 0)

    again = store.create_job({
        "job_type": "delete_document",
        "user_id": "user-a",
        "kb_id": "kb-a",
        "doc_id": job["doc_id"],
        "idempotency_key": f"delete-document:kb-a:{job['doc_id']}",
    })
    repeat = store.create_job({
        "job_type": "delete_document",
        "user_id": "user-a",
        "kb_id": "kb-a",
        "doc_id": job["doc_id"],
        "idempotency_key": f"delete-document:kb-a:{job['doc_id']}",
    })
    assert again["job_id"] == repeat["job_id"]
    stop = threading.Event()
    thread = _drive(store, "worker-delete-doc", stop)
    deleted = _wait_status(store, again["job_id"], {"succeeded", "consistency_failed"})
    stop.set()
    thread.join(5)
    assert deleted["status"] == "succeeded"
    assert _counts(working, job["doc_id"]) == (0, 0, 0)
    third = store.create_job({
        "job_type": "delete_document",
        "user_id": "user-a",
        "kb_id": "kb-a",
        "doc_id": job["doc_id"],
        "idempotency_key": f"delete-document:kb-a:{job['doc_id']}",
    })
    assert third["job_id"] == again["job_id"]
    assert third["status"] == "succeeded"


def test_unknown_owner_stays_quarantined_and_backup_can_be_restored(tmp_path, monkeypatch):
    working, inputs, store = _workspace(tmp_path)
    _bind(monkeypatch, working, inputs)
    data = load_shell(working)
    data["purge_jobs"] = [{"job_id": "job-orphan", "kb_id": "kb-missing", "status": "pending"}]
    data["kbs"].append({"id": "kb-known", "owner_id": "user-a", "name": "known"})
    data["purge_jobs"].append({"job_id": "job-known", "kb_id": "kb-known", "status": "pending"})
    save_shell(working, data)
    assert import_pending_purges(working, store) == 1
    assert store.get_job("job-orphan") is None
    assert store.get_job("job-known")["status"] == "queued"
    _save_kv(working, "text_chunks", {"loose": {"content": "无归属"}})
    (working / "semantic_cache.json").write_text("{", encoding="utf-8")
    (working / "semantic_cache.json.bak").write_text("[]", encoding="utf-8")
    before = scan(working, "kb-a")
    assert any(item["kind"] == "quarantine" for item in before["issues"])
    assert any(item["kind"] == "corrupt_json" for item in before["issues"])
    assert scope_visible({"content": "无归属"}, "kb-a", {}, {}) is False
    repaired = repair_safe(working, "kb-a")
    assert "semantic_cache.json" in repaired["restored"]
    assert (working / "semantic_cache.json").read_text(encoding="utf-8") == "[]"
    assert _kv(working, "text_chunks")["loose"].get("kb_id", "") == ""


def test_production_worker_refuses_to_start_without_postgres(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DATABASE_URL", "")
    with pytest.raises(SystemExit):
        serve()


def test_jobs_are_listed_filtered_and_sorted(tmp_path, monkeypatch):
    working, inputs, store = _workspace(tmp_path)
    _bind(monkeypatch, working, inputs)
    older = store.create_job({
        "job_type": "ingestion",
        "user_id": "user-a",
        "kb_id": "kb-a",
        "doc_id": "doc-old",
        "idempotency_key": "ingestion:user-a:kb-a:old",
    })
    time.sleep(0.01)
    newer = store.create_job({
        "job_type": "ingestion",
        "user_id": "user-a",
        "kb_id": "kb-a",
        "doc_id": "doc-new",
        "idempotency_key": "ingestion:user-a:kb-a:new",
    })
    page = store.list_jobs(user_id="user-a", status="queued", page=1, page_size=1)
    assert page["total"] == 2
    assert page["items"][0]["job_id"] == newer["job_id"]
    assert store.list_jobs(user_id="user-a", doc_id="doc-old")["items"][0]["job_id"] == older["job_id"]
    failed = store.transition(older["job_id"], "failed", error_code="user_input", error_message="文件内容为空")
    assert store.request_retry(failed["job_id"], "user-a")["status"] == "queued"


def test_api_returns_job_id_immediately_and_refresh_uses_saved_status(tmp_path, monkeypatch):
    working, inputs, _store = _workspace(tmp_path)
    _bind(monkeypatch, working, inputs, PRODUCT_AUTH="1", TOKEN_SECRET="unit-test-secret-for-jobs")

    class Rag:
        def __init__(self):
            self.working_dir = str(working)
            self.addon_params = {}

        async def get_docs_by_status(self, _status):
            return {}

    class Docs:
        def __init__(self):
            self.input_dir = str(inputs)

        def is_supported_file(self, name: str) -> bool:
            return name.endswith(".txt")

    from lightrag.api.routers.product_shell import create_product_shell_routes

    app = FastAPI()
    app.include_router(create_product_shell_routes(Rag(), Docs()))
    client = TestClient(app)
    csrf = {"X-KB-Request": "1"}
    created = client.post("/api/v1/auth/register", json={"email": "owner@example.com", "password": "correct-horse"}, headers=csrf)
    assert created.status_code == 200
    owner = {"Authorization": f"Bearer {created.json()['access_token']}", **csrf}
    other = client.post("/api/v1/auth/register", json={"email": "other@example.com", "password": "correct-horse"}, headers=csrf)
    stranger = {"Authorization": f"Bearer {other.json()['access_token']}", **csrf}
    kb = client.post("/api/v1/kb", json={"name": "任务库"}, headers=owner)
    assert kb.status_code == 200
    kb_id = kb.json()["id"]
    uploaded = client.post(
        "/api/v1/documents",
        data={"kb_id": kb_id, "chunk_strategy": "one"},
        files={"file": ("note.txt", "接口立即返回\n\n第二段".encode("utf-8"), "text/plain")},
        headers=owner,
    )
    assert uploaded.status_code == 200
    body = uploaded.json()
    assert body["job_id"]
    assert body["status"] == "queued"
    assert "traceback" not in uploaded.text
    assert "sk-" not in uploaded.text
    hidden = client.get(f"/api/v1/jobs/{body['job_id']}", headers=stranger)
    assert hidden.status_code == 404
    assert client.get("/api/v1/jobs", headers=stranger).json()["total"] == 0
    assert client.post(f"/api/v1/jobs/{body['job_id']}/cancel", headers=stranger).status_code == 404
    visible = client.get(f"/api/v1/jobs/{body['job_id']}", headers=owner)
    assert visible.status_code == 200
    assert visible.json()["status"] == "queued"
    listed = client.get("/api/v1/jobs", params={"kb_id": kb_id, "status": "queued"}, headers=owner)
    assert listed.json()["total"] == 1
    stop = threading.Event()
    thread = _drive(SqliteStore(working / "product_jobs.sqlite"), "api-worker", stop)
    deadline = time.time() + 20
    final = {}
    while time.time() < deadline:
        final = client.get(f"/api/v1/jobs/{body['job_id']}", headers=owner).json()
        if final["status"] in {"succeeded", "failed", "consistency_failed"}:
            break
        time.sleep(0.05)
    stop.set()
    thread.join(5)
    assert final["status"] == "succeeded"
    docs = client.get("/api/v1/documents", params={"kb_id": kb_id}, headers=owner).json()
    match = next(item for item in docs["items"] if item["id"] == body["doc_id"])
    assert match["status"] == "ready"
    assert match["progress"] == 100
    page = (ROOT / "product_webui" / "src" / "pages" / "Documents" / "index.tsx").read_text(encoding="utf-8")
    assert "jobApi.retry" in page
    assert "consistency_failed" in page
    assert "失败" in page
