# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import threading
from pathlib import Path

sys.argv = ["lightrag"]

from lightrag.api.routers.product_shell import ShellStore
from lightrag.product_storage import load_shell, save_shell

PROJECT = Path(__file__).resolve().parents[1]


def test_permission_error_retries_and_reads_backup(tmp_path, monkeypatch):
    working = tmp_path / "rag"
    working.mkdir()
    save_shell(
        working, {"kbs": [{"id": "kb-a"}], "upload_jobs": {"job": {"status": "ready"}}}
    )
    calls = {"n": 0}
    original = Path.replace

    def flaky(self, target):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError("product_shell.json")
        return original(self, target)

    monkeypatch.setattr(Path, "replace", flaky)
    current = load_shell(working)
    current["upload_jobs"]["job"]["status"] = "embedding"
    save_shell(working, current)
    loaded = ShellStore(str(working)).load()
    assert loaded["upload_jobs"]["job"]["status"] == "embedding"


def test_api_poll_and_worker_update_twenty_rounds(tmp_path):
    working = tmp_path / "rag"
    working.mkdir()
    initial = load_shell(working)
    initial["kbs"] = [
        {
            "id": "kb-a",
            "owner_id": "user-a",
            "name": "个人库",
            "settings": {},
            "graph_config": {},
        }
    ]
    initial["upload_jobs"] = {"job-1": {"status": "queued", "n": 0}}
    save_shell(working, initial)
    errors: list[str] = []

    def worker():
        try:
            for step in range(15):
                current = load_shell(working)
                current["upload_jobs"]["job-1"] = {"status": "embedding", "n": step}
                save_shell(working, current)
        except Exception as exc:
            errors.append(f"worker:{exc}")

    def api_poll():
        store = ShellStore(str(working))
        try:
            for _ in range(30):
                data = store.load()
                job = (data.get("upload_jobs") or {}).get("job-1")
                if not isinstance(data.get("kbs"), list) or not isinstance(job, dict):
                    errors.append("incomplete")
        except Exception as exc:
            errors.append(f"api:{exc}")

    for _round in range(20):
        threads = [threading.Thread(target=worker)]
        threads.extend(threading.Thread(target=api_poll) for _ in range(3))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert errors == []
    final = load_shell(working)
    assert final["upload_jobs"]["job-1"]["status"] == "embedding"


def test_two_processes_keep_distinct_fields_twenty_times(tmp_path):
    working = tmp_path / "rag"
    working.mkdir()
    save_shell(working, load_shell(working))
    code = """
import sys
from pathlib import Path
from lightrag.product_storage import mutate_shell
working = Path(sys.argv[1])
field = sys.argv[2]
for step in range(25):
    def editor(data, field=field, step=step):
        bucket = data.get(field)
        if not isinstance(bucket, dict):
            bucket = {}
            data[field] = bucket
        bucket["writer"] = field
        bucket["n"] = step
    mutate_shell(working, editor)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT) + os.pathsep + env.get("PYTHONPATH", "")
    for _round in range(20):
        processes = [
            subprocess.Popen(
                [sys.executable, "-c", code, str(working), field],
                cwd=str(PROJECT),
                env=env,
            )
            for field in ("upload_jobs", "audits")
        ]
        codes = [process.wait(timeout=60) for process in processes]
        assert codes == [0, 0], codes
        final = load_shell(working)
        assert final["upload_jobs"]["writer"] == "upload_jobs"
        assert final["audits"]["writer"] == "audits"
        assert final["upload_jobs"]["n"] == 24
        assert final["audits"]["n"] == 24
