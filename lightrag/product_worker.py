"""独立任务进程。

API 只创建任务。这个进程按 job_type 领取租约，心跳续期，退出时释放租约。
崩溃后租约过期，其他 worker 可以把 running 任务收回队列。
"""

from __future__ import annotations

import os
import signal
import socket
import threading
from pathlib import Path
from typing import Any

from lightrag.product_db import job_settings, open_store, production_database_error
from lightrag.product_ingest import run_delete_document, run_ingestion, run_purge


def _dirs() -> tuple[Path, Path]:
    working = Path(os.getenv("WORKING_DIR", "./data/rag_storage"))
    inputs = Path(os.getenv("INPUT_DIR", "./data/inputs"))
    working.mkdir(parents=True, exist_ok=True)
    inputs.mkdir(parents=True, exist_ok=True)
    return working, inputs


class _Guard:
    def __init__(self, store: Any, job_id: str, worker_id: str, stop: threading.Event):
        self.store = store
        self.job_id = job_id
        self.worker_id = worker_id
        self.stop = stop
        self.lost = False
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _loop(self) -> None:
        interval = job_settings()["heartbeat_seconds"]
        while not self._done.wait(interval):
            if not self.store.heartbeat(self.job_id, self.worker_id):
                self.lost = True
                return

    def alive(self) -> bool:
        return not self.stop.is_set() and not self.lost

    def close(self) -> None:
        self._done.set()
        self._thread.join(timeout=2)


_STORAGE_LOCK = threading.Lock()


def import_pending_purges(working: Path, store: Any | None = None) -> int:
    """把仍未完成、且能确认归属的旧删除任务导入任务库。无法确认归属的不分配。"""
    from lightrag.product_storage import load_shell

    data = load_shell(working)
    store = store or open_store(working / "product_jobs.sqlite")
    imported = 0
    for job in data.get("purge_jobs") or []:
        if str(job.get("status") or "") not in {"pending", "running"}:
            continue
        kb_id = str(job.get("kb_id") or "")
        owner = ""
        for kb in data.get("kbs") or []:
            if kb.get("id") == kb_id:
                owner = str(kb.get("owner_id") or "")
                break
        if not owner or not kb_id or not job.get("job_id"):
            continue
        before = store.get_job(str(job["job_id"]))
        store.create_job(
            {
                "job_id": job["job_id"],
                "job_type": "purge",
                "user_id": owner,
                "kb_id": kb_id,
                "doc_id": "",
                "idempotency_key": f"purge:{kb_id}:{job['job_id']}",
                "input_snapshot": {"kb_id": kb_id},
            }
        )
        if before is None:
            imported += 1
    return imported


def process_job(
    store: Any, job: dict[str, Any], working: Path, inputs: Path, stop: threading.Event
) -> None:
    guard = _Guard(store, job["job_id"], str(job.get("worker_id") or ""), stop)
    guard.start()
    try:
        with _STORAGE_LOCK:
            kind = job.get("job_type")
            if kind == "purge":
                run_purge(store, job, working, inputs)
            elif kind == "delete_document":
                run_delete_document(store, job, working, inputs)
            elif kind == "index_rebuild":
                from lightrag.index_manifest import run_index_rebuild

                run_index_rebuild(store, job, working)
            else:
                run_ingestion(store, job, working, inputs, guard.alive)
    finally:
        guard.close()
        if stop.is_set():
            store.release(job["job_id"], str(job.get("worker_id") or ""))


def worker_loop(
    store: Any,
    worker_id: str,
    stop: threading.Event,
    job_types: list[str] | None = None,
) -> None:
    working, inputs = _dirs()
    settings = job_settings()
    handled = 0
    limit = int(os.getenv("WORKER_MAX_JOBS", "0") or "0")
    while not stop.is_set():
        job = store.acquire(worker_id, job_types)
        if job is None:
            if stop.wait(settings["poll_interval"]):
                return
            continue
        process_job(store, job, working, inputs, stop)
        handled += 1
        if limit and handled >= limit:
            return


def serve(job_types: list[str] | None = None) -> None:
    from lightrag.retrieval_admission import retrieval_profile_startup_error

    profile_blocked = retrieval_profile_startup_error()
    if profile_blocked:
        raise SystemExit(profile_blocked)
    blocked = production_database_error()
    if os.getenv("APP_ENV", "").strip().lower() == "production" and not os.getenv(
        "DATABASE_URL", ""
    ).strip().startswith("postgres"):
        raise SystemExit(
            blocked or "生产环境的 Worker 必须配置 PostgreSQL DATABASE_URL"
        )
    store = open_store()
    store.migrate()
    working, _inputs = _dirs()
    store.recover_expired()
    import_pending_purges(working, store)
    stop = threading.Event()

    def request_stop(signum: int, _frame: Any) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)
    settings = job_settings()
    prefix = f"{socket.gethostname()}-{os.getpid()}"
    threads = []
    for index in range(settings["worker_count"]):
        thread = threading.Thread(
            target=worker_loop,
            args=(store, f"{prefix}-{index}", stop, job_types),
            daemon=True,
        )
        thread.start()
        threads.append(thread)
    while any(thread.is_alive() for thread in threads):
        stop.wait(0.2)
        if stop.is_set():
            break
    for thread in threads:
        thread.join(timeout=settings["lease_seconds"])


def main() -> None:
    raw = os.getenv("WORKER_JOB_TYPES", "").strip()
    types = [item.strip() for item in raw.split(",") if item.strip()] or None
    serve(types)


if __name__ == "__main__":
    main()
