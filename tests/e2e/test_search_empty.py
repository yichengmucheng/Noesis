# -*- coding: utf-8 -*-
"""检索测试页在 chunks=[] 时结束加载并显示 empty_reason。不占用 9621。"""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[2]
PORT = 9645


def _free(port: int) -> None:
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", port))
    except OSError as exc:
        raise RuntimeError(f"端口 {port} 已被占用，测试不会使用 9621") from exc
    finally:
        sock.close()


def _wait(port: int, proc: subprocess.Popen, log_path: Path) -> None:
    import urllib.request

    deadline = time.time() + 120
    last = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            raise RuntimeError(tail or last)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last = str(exc)
        time.sleep(1)
    raise RuntimeError(last or "health timeout")


def test_search_page_clears_results_and_shows_empty_reason(tmp_path):
    if os.getenv("APP_ENV", "").strip().lower() == "production" and "9621" in os.getenv("PORT", ""):
        pytest.fail("这个测试不能连接 9621")
    _free(PORT)
    root = tmp_path / "empty-ui"
    working = root / "rag"
    inputs = root / "inputs"
    working.mkdir(parents=True)
    inputs.mkdir()
    from lightrag.product_storage import save_shell

    save_shell(working, {
        "kbs": [{
            "id": "kb-empty",
            "name": "空态库",
            "owner_id": "local-owner",
            "description": "",
            "settings": {"retrieval_mode": "vector", "graph_enabled": True},
            "graph_config": {},
        }],
    })
    env = os.environ.copy()
    for key in list(env):
        if any(word in key.upper() for word in ("KEY", "SECRET", "TOKEN", "PASSWORD", "COOKIE")):
            env.pop(key, None)
    env.update({
        "APP_ENV": "development",
        "PRODUCT_AUTH": "0",
        "HOST": "127.0.0.1",
        "PORT": str(PORT),
        "WORKING_DIR": str(working),
        "INPUT_DIR": str(inputs),
        "DATABASE_URL": "",
        "EMBEDDING_BINDING": "openai",
        "EMBEDDING_MODEL": "test-embed",
        "EMBEDDING_DIM": "8",
        "EMBEDDING_BINDING_HOST": "http://127.0.0.1:9/v1",
        "LLM_BINDING": "openai",
        "LLM_MODEL": "test-model",
        "LLM_BINDING_HOST": "http://127.0.0.1:9/v1",
        "PYTHONPATH": str(PROJECT),
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "NO_PROXY": "127.0.0.1,localhost",
    })
    (root / ".env").write_text(
        "\n".join(f"{key}={env[key]}" for key in (
            "APP_ENV", "PRODUCT_AUTH", "HOST", "PORT", "WORKING_DIR", "INPUT_DIR", "DATABASE_URL",
            "EMBEDDING_BINDING", "EMBEDDING_MODEL", "EMBEDDING_DIM",
        )),
        encoding="utf-8",
    )
    log_path = root / "server.log"
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        [sys.executable, "-m", "lightrag.api.lightrag_server"],
        cwd=str(root),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    shot = Path(os.environ.get("TEMP", str(root))) / "kb-search-empty.png"
    try:
        os.environ["NO_PROXY"] = "127.0.0.1,localhost"
        _wait(PORT, proc, log_path)
        from playwright.sync_api import sync_playwright

        calls = {"n": 0}

        def fulfill(route):
            calls["n"] += 1
            body = json.loads(route.request.post_data or "{}")
            query = body.get("query") or ""
            if "津门" in query:
                payload = {
                    "chunks": [{
                        "chunk_id": "doc-edu-c0001",
                        "content": "林知夏2018至2022年就读津门财经大学",
                        "excerpt": "林知夏2018至2022年就读津门财经大学",
                        "document_id": "doc-edu",
                        "unit_id": "t0001",
                        "kb_id": "kb-empty",
                        "owner_id": "local-owner",
                        "score": 0.91,
                        "ranking_score": 0.8,
                        "admission_score": 0.91,
                        "raw_score": 0.8,
                        "channel": "dense",
                        "path": "",
                    }],
                    "empty_reason": None,
                    "status": "ok",
                    "query_plan": {
                        "route": "exact",
                        "reason": "编号查询只保留原问题",
                        "variants": ["津门财经大学"],
                        "hyde_text": "",
                        "llm_used": False,
                    },
                    "retrieval": {
                        "dense_top": 40,
                        "bm25_top": 30,
                        "graph_top": 0,
                        "rerank_pool": 0,
                        "children": 1,
                        "parents": 1,
                    },
                }
            else:
                payload = {"chunks": [], "empty_reason": "没有达到相关度要求的内容"}
            route.fulfill(status=200, content_type="application/json", body=json.dumps(payload, ensure_ascii=False))

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(args=["--no-proxy-server"])
            page = browser.new_page()
            page.route("**/api/v1/chat/search-test", fulfill)
            page.goto(f"http://127.0.0.1:{PORT}/console/kb/kb-empty/search", wait_until="domcontentloaded")
            page.get_by_role("heading", name="检索测试").wait_for()
            box = page.get_by_placeholder("例如：这份资料的核心结论是什么")

            def search(text):
                box.fill(text)
                page.get_by_test_id("search-run").click()

            search("津门财经大学")
            page.get_by_text("林知夏2018至2022年就读津门财经大学").wait_for()
            page.get_by_test_id("search-plan").wait_for()
            assert "exact" in page.get_by_test_id("search-plan").inner_text()
            assert page.get_by_test_id("search-loading").count() == 0

            search("量子计算芯片原理")
            page.get_by_test_id("search-empty").wait_for()
            page.get_by_text("没有达到相关度要求的内容").wait_for()
            assert page.get_by_test_id("search-loading").count() == 0
            assert page.get_by_text("林知夏2018至2022年就读津门财经大学").count() == 0

            search("津门财经大学信息管理")
            page.get_by_text("林知夏2018至2022年就读津门财经大学").wait_for()
            search("今天天气怎么样")
            page.get_by_test_id("search-empty").wait_for()
            page.get_by_text("没有达到相关度要求的内容").wait_for()
            assert page.get_by_test_id("search-loading").count() == 0
            assert page.get_by_text("林知夏2018至2022年就读津门财经大学").count() == 0
            page.screenshot(path=str(shot), full_page=True)
            browser.close()
        assert calls["n"] >= 4
        assert shot.exists()
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        log.close()
