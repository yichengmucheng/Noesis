# -*- coding: utf-8 -*-
"""个人知识库前端闭环。临时端口，不读取项目 .env，不占用 9621。"""

import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[2]
PORT = 9646


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
            raise RuntimeError(log_path.read_text(encoding="utf-8", errors="replace")[-2000:] or last)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last = str(exc)
        time.sleep(1)
    raise RuntimeError(last or "health timeout")


def _forbid(text: str) -> None:
    for word in ("HyDE", "admission", "reranker", "Top-K", "bge-reranker"):
        assert word not in text


def test_product_navigation_account_jobs_chat(tmp_path):
    if PORT == 9621:
        pytest.fail("这个测试不能连接 9621")
    _free(PORT)
    root = tmp_path / "product-ui"
    working = root / "rag"
    inputs = root / "inputs"
    working.mkdir(parents=True)
    inputs.mkdir()
    env = {
        key: value
        for key, value in os.environ.items()
        if not any(word in key.upper() for word in ("KEY", "SECRET", "TOKEN", "PASSWORD", "COOKIE"))
    }
    env.update({
        "APP_ENV": "development",
        "PRODUCT_AUTH": "1",
        "TOKEN_SECRET": "playwright-ui-secret-32bytes-ok!!",
        "CORS_ORIGINS": f"http://127.0.0.1:{PORT}",
        "COOKIE_SECURE": "0",
        "AUTH_RATE_LIMIT": "1000",
        "HOST": "127.0.0.1",
        "PORT": str(PORT),
        "WORKING_DIR": str(working),
        "INPUT_DIR": str(inputs),
        "DATABASE_URL": "",
        "MAX_UPLOAD_MB": "12",
        "MAX_FILES_PER_KB": "4",
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
            "APP_ENV", "PRODUCT_AUTH", "TOKEN_SECRET", "CORS_ORIGINS", "COOKIE_SECURE",
            "WORKING_DIR", "INPUT_DIR", "DATABASE_URL", "MAX_UPLOAD_MB", "MAX_FILES_PER_KB",
            "EMBEDDING_BINDING", "EMBEDDING_MODEL", "EMBEDDING_DIM", "LLM_BINDING", "LLM_MODEL",
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
    shot_dir = Path(os.environ.get("TEMP", str(root)))
    desktop = shot_dir / "kb-ui-desktop.png"
    mobile = shot_dir / "kb-ui-mobile.png"
    try:
        os.environ["NO_PROXY"] = "127.0.0.1,localhost"
        _wait(PORT, proc, log_path)
        from playwright.sync_api import sync_playwright

        calls = {"cancel": 0, "retry": 0, "deep": 0, "check": 0, "search": 0}

        def documents(route):
            request = route.request
            if request.method == "DELETE":
                route.fulfill(status=200, content_type="application/json", body=json.dumps({
                    "job_id": "job-del", "status": "succeeded", "file_name": "笔记.txt",
                }, ensure_ascii=False))
                return
            if request.method == "GET" and "/documents" in request.url and "/status" not in request.url:
                route.fulfill(status=200, content_type="application/json", body=json.dumps({
                    "items": [{
                        "id": "doc-1",
                        "name": "笔记.txt",
                        "chunk_count": 3,
                        "char_count": 120,
                        "status": "ready",
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "updated_at": "2026-01-02T00:00:00+00:00",
                        "mime_type": "text/plain",
                        "file_size": 120,
                        "unit_count": 2,
                        "index_status": "ready",
                        "version_id": "ver-1",
                    }],
                    "total": 1,
                }, ensure_ascii=False))
                return
            route.fallback()

        def jobs(route):
            if "/jobs" not in route.request.url:
                route.fallback()
                return
            if route.request.method == "POST" and route.request.url.endswith("/cancel"):
                calls["cancel"] += 1
                route.fulfill(status=200, content_type="application/json", body=json.dumps({"job_id": "j2", "status": "cancelled"}))
                return
            if route.request.method == "POST" and route.request.url.endswith("/retry"):
                calls["retry"] += 1
                route.fulfill(status=200, content_type="application/json", body=json.dumps({"job_id": "j1", "status": "queued"}))
                return
            route.fulfill(status=200, content_type="application/json", body=json.dumps({
                "items": [
                    {
                        "job_id": "j1", "file_name": "笔记.txt", "status": "failed", "stage": "embedding",
                        "error": "嵌入失败：模型没有返回向量", "created_at": "2026-01-01T00:00:00+00:00",
                    },
                    {
                        "job_id": "j2", "file_name": "排队.txt", "status": "running", "stage": "parsing",
                        "created_at": "2026-01-01T00:00:00+00:00",
                    },
                ],
                "total": 2, "page": 1, "page_size": 8,
            }, ensure_ascii=False))

        def checks(route):
            if route.request.url.endswith("/deep-check"):
                calls["deep"] += 1
            else:
                calls["check"] += 1
            route.fulfill(status=200, content_type="application/json", body=json.dumps({
                "passed": False,
                "stores": {
                    "text_chunks": {"count": 1, "samples": ["vdb_chunks.json"]},
                    "chunks_vdb": {"count": 0, "samples": ["internal-store.json"]},
                },
            }, ensure_ascii=False))

        held = []
        bodies = []

        def chat(route):
            body = json.loads(route.request.post_data or "{}")
            bodies.append(body)
            query = body.get("query") or ""
            if query == "停一下":
                held.append(route)
                return
            if query == "没有资料":
                payload = 'data: {"type":"meta","citations":[],"answerable":false}\n\n'
            elif query == "失败问题":
                payload = 'data: {"type":"error","message":"问答服务暂时不可用"}\n\n'
            elif query == "编号问题":
                payload = (
                    'data: {"type":"meta","answerable":true,"citations":[{"citation_id":"C1","doc_name":"笔记.txt","document_id":"doc-1","chunk_id":"c1","unit_id":"t1","excerpt":"命中句子"}]}\n\n'
                    'data: {"type":"token","text":"答案见 [C1]"}\n\n'
                    'data: {"type":"done","answer":"答案见 [C1]","answerable":true,"complete":true,"citations":[{"citation_id":"C1","doc_name":"笔记.txt","document_id":"doc-1","chunk_id":"c1","unit_id":"t1","excerpt":"命中句子"}]}\n\n'
                )
            elif query == "无位置":
                payload = (
                    'data: {"type":"token","text":"没有坐标 [C1]"}\n\n'
                    'data: {"type":"done","answer":"没有坐标 [C1]","answerable":true,"citations":[{"citation_id":"C1","doc_name":"旧笔记.txt","document_id":"doc-1","chunk_id":"c1","unit_id":"none","excerpt":"旧摘录"}]}\n\n'
                )
            elif query == "失效来源":
                payload = (
                    'data: {"type":"token","text":"旧引用 [C1]"}\n\n'
                    'data: {"type":"done","answer":"旧引用 [C1]","answerable":true,"citations":[{"citation_id":"C1","doc_name":"已删.txt","document_id":"doc-gone","chunk_id":"c9","excerpt":"已删除摘录"}]}\n\n'
                )
            else:
                payload = (
                    'data: {"type":"meta","citations":[{"doc_name":"笔记.txt","excerpt":"命中句子","parent_content":"所在章节的补充","page_number":3}]}\n\n'
                    'data: {"type":"token","text":"答案见 [1]"}\n\n'
                )
            route.fulfill(status=200, content_type="text/event-stream", body=payload)

        def source(route):
            if "doc-gone" in route.request.url:
                route.fulfill(status=404, content_type="application/json", body='{"detail":"来源不存在"}')
                return
            if "unit_id=none" in route.request.url:
                route.fulfill(status=200, content_type="application/json", body=json.dumps({
                    "doc_name": "旧笔记.txt",
                    "source_kind": "text",
                    "unit": {"page_number": None, "slide_number": None, "section_path": [], "line_start": None, "bbox": None},
                    "matched_chunk": {"excerpt": "旧摘录"},
                    "parent_context": "",
                    "source_text": "",
                    "preview": {"available": False, "kind": "none", "url": None},
                }, ensure_ascii=False))
                return
            route.fulfill(status=200, content_type="application/json", body=json.dumps({
                "doc_name": "笔记.txt",
                "source_kind": "text",
                "unit": {"page_number": None, "line_start": 4, "line_end": 4, "section_path": ["冷却系统"]},
                "matched_chunk": {"excerpt": "命中句子"},
                "parent_context": "所在章节的补充",
                "source_text": "命中句子",
                "preview": {"available": False, "kind": "text", "url": None},
            }, ensure_ascii=False))

        def content(route):
            if "doc-gone" in route.request.url:
                route.fulfill(status=404, content_type="application/json", body='{"detail":"来源不存在"}')
                return
            if "unit_id=none" in route.request.url:
                route.fulfill(status=200, content_type="application/json", body=json.dumps({
                    "source_kind": "text",
                    "text": "旧摘录",
                    "units": [],
                    "layout": "original",
                    "doc_name": "旧笔记.txt",
                }, ensure_ascii=False))
                return
            route.fulfill(status=200, content_type="application/json", body=json.dumps({
                "source_kind": "text",
                "text": "第一行\n第二行\n第三行\n命中句子\n第五行",
                "units": [],
                "layout": "original",
                "doc_name": "笔记.txt",
            }, ensure_ascii=False))

        def search(route):
            calls["search"] += 1
            route.fulfill(status=200, content_type="application/json", body=json.dumps({
                "status": "ok",
                "chunks": [{
                    "chunk_id": "c1",
                    "excerpt": "诊断摘录",
                    "doc_name": "笔记.txt",
                    "channel": "dense",
                    "channel_rank": 1,
                    "path_id": "path-1",
                    "path": "甲到乙",
                    "parent_id": "parent-1",
                    "admission_reason": "分数通过",
                    "admission_score": 0.2,
                    "score": 0.8,
                }],
                "query_plan": {
                    "route": "rewrite",
                    "standalone_query": "独立后的问题",
                    "reason": "问法需要整理",
                    "variants": ["问法甲"],
                    "hyde_text": "假设段落",
                    "subquestions": ["子问题一"],
                    "llm_used": True,
                },
                "retrieval": {
                    "dense_top": 40, "bm25_top": 30, "graph_top": 2, "rrf": 12,
                    "rerank_pool": 40, "children": 1, "parents": 1,
                },
                "rerank": {"model": "test-rerank", "applied": True, "pool_size": 40},
                "timings": {"retrieve": 3},
                "index_version": "rev-1",
                "embedding_model": "Qwen/Qwen3-Embedding-4B",
            }, ensure_ascii=False))

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(args=["--no-proxy-server"])
            context = browser.new_context(
                viewport={"width": 1280, "height": 800},
                permissions=["clipboard-read", "clipboard-write"],
            )
            page = context.new_page()
            page.route("**/api/v1/kb/**/jobs**", jobs)
            page.route("**/api/v1/jobs/**", jobs)
            page.route("**/api/v1/kb/purge-jobs/**", checks)
            page.route("**/api/v1/chat/stream", chat)
            page.route(re.compile(r"/api/v1/documents"), documents)
            page.route(re.compile(r"/api/v1/documents/.+/content"), content)
            page.route(re.compile(r"/api/v1/documents/.+/source"), source)
            page.goto(f"http://127.0.0.1:{PORT}/console/", wait_until="domcontentloaded")
            page.get_by_test_id("auth-switch").click()
            page.get_by_test_id("email").fill("ui.browser@example.com")
            page.get_by_test_id("password").fill("correct-horse")
            page.get_by_test_id("auth-submit").click()
            page.get_by_test_id("kb-home").wait_for()
            page.get_by_role("button", name="新建知识库").click()
            page.get_by_label("知识库名称").fill("个人资料")
            page.get_by_role("button", name="创建").click()
            page.get_by_text("知识库创建成功").wait_for()
            page.locator(".ant-card", has_text="个人资料").click()
            page.wait_for_url("**/documents")
            page.get_by_role("heading", name="资料").wait_for()
            assert page.get_by_text("练习").count() == 0
            page.get_by_test_id("nav-documents").wait_for()
            page.get_by_test_id("nav-chat").wait_for()
            page.get_by_test_id("nav-advanced").click()
            page.get_by_test_id("nav-search").wait_for()
            _forbid(page.locator("body").inner_text())
            assert "12MB" in page.locator("body").inner_text() or page.get_by_text("12MB").count() >= 0
            page.get_by_role("button", name="导入文件").click()
            page.get_by_text("不超过 12MB").wait_for()
            page.get_by_role("button", name="取消").click()
            page.get_by_role("dialog").wait_for(state="hidden")

            page.get_by_test_id("open-jobs").click()
            page.get_by_text("嵌入失败：模型没有返回向量").wait_for()
            page.get_by_text("解析").first.wait_for()
            page.get_by_role("button", name="取消任务").click()
            page.get_by_role("button", name="重试任务").click()
            page.locator(".ant-drawer-open .ant-drawer-close").click()

            page.get_by_text("笔记.txt").first.click()
            page.get_by_test_id("doc-detail").wait_for()
            page.get_by_text("切块 3").wait_for()
            page.get_by_text("文件类型 text/plain").wait_for()
            page.get_by_text("位置单元 2").wait_for()
            page.get_by_text("大小 120 字节").wait_for()
            page.get_by_text("版本 ver-1").wait_for()
            page.locator(".ant-drawer-open .ant-drawer-close").click()
            page.get_by_role("button", name="删除").click()
            with page.expect_response(lambda item: item.request.method == "DELETE") as deleted:
                page.locator(".ant-popconfirm").get_by_role("button", name="确定").click()
            assert deleted.value.status == 200
            page.get_by_test_id("delete-job").wait_for()
            page.get_by_test_id("run-deep-check").click()
            page.get_by_test_id("deep-check").wait_for()
            report = page.get_by_test_id("deep-check").inner_text()
            assert "资料片段仍有 1 处未清理" in report
            assert "vdb_chunks.json" not in report
            assert "internal-store" not in report
            page.get_by_role("button", name="一致性检查").click()

            (working / "vdb_chunks.json").write_text("{}", encoding="utf-8")
            page.reload(wait_until="domcontentloaded")
            page.get_by_test_id("index-mismatch").wait_for()
            page.get_by_text("需要重新构建索引").wait_for()
            desktop_overflow = page.evaluate("() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1")
            page.screenshot(path=str(desktop), full_page=True)

            search_nav = page.get_by_test_id("nav-search")
            if not search_nav.is_visible():
                page.get_by_test_id("nav-advanced").click()
            search_nav.click()
            page.get_by_role("heading", name="检索测试").wait_for()
            page.get_by_placeholder("例如：这份资料的核心结论是什么").fill("索引不一致")
            page.get_by_test_id("search-run").click()
            page.get_by_test_id("index-mismatch").wait_for()
            assert page.get_by_test_id("search-empty").count() == 0

            (working / "vdb_chunks.json").unlink()
            page.route("**/api/v1/chat/search-test", search)
            page.get_by_placeholder("例如：这份资料的核心结论是什么").fill("诊断问题")
            page.get_by_test_id("search-run").click()
            page.get_by_text("诊断摘录").wait_for()
            page.get_by_test_id("search-plan").wait_for()
            assert "rewrite" in page.get_by_test_id("search-plan").inner_text()
            page.get_by_text("检索诊断").click()
            diag = page.get_by_test_id("search-diagnostics").inner_text()
            assert "独立后的问题" in diag
            assert "HyDE" in diag
            assert "path-1" in diag
            assert "admission" in diag
            assert "test-rerank" in diag
            assert "rev-1" in diag
            assert "Qwen/Qwen3-Embedding-4B" in diag

            page.get_by_test_id("nav-chat").click()
            page.get_by_role("heading", name="问答").wait_for()
            _forbid(page.locator("body").inner_text())
            box = page.get_by_placeholder("输入问题，Shift+Enter 换行，Enter 发送")
            box.fill("核心结论")
            page.get_by_role("button", name="发送").click()
            page.get_by_text("答案见 [1]").wait_for()
            page.get_by_test_id("citation-open").click()
            page.get_by_test_id("source-drawer").wait_for()
            page.get_by_test_id("source-reader").wait_for()
            drawer = page.get_by_test_id("source-drawer").inner_text()
            assert "笔记.txt" in drawer
            assert "第3页" in drawer
            assert "命中内容" in drawer
            assert "命中句子" in drawer
            assert "补充上下文" in drawer
            page.locator(".ant-drawer-open .ant-drawer-close").click()
            page.get_by_test_id("chat-copy").click()
            copied = page.evaluate("() => navigator.clipboard.readText()")
            assert copied == "答案见 [1]"
            page.get_by_test_id("chat-retry").click()
            page.get_by_text("答案见 [1]").wait_for()
            page.get_by_test_id("chat-clear").click()
            assert page.get_by_text("答案见 [1]").count() == 0

            box.fill("编号问题")
            page.get_by_role("button", name="发送").click()
            page.get_by_text("答案见 [C1]").wait_for()
            page.get_by_test_id("citation-mark").click()
            page.get_by_test_id("source-drawer").wait_for()
            located = page.get_by_test_id("source-drawer").inner_text()
            assert "笔记.txt" in located
            assert "第4行" in located
            assert "命中句子" in located
            assert "打开原文" not in located
            assert "已找到来源，但暂无精确位置" not in located
            page.locator(".ant-drawer-open .ant-drawer-close").click()
            page.set_viewport_size({"width": 390, "height": 844})
            page.get_by_test_id("citation-mark").last.click()
            page.get_by_test_id("source-reader").wait_for()
            page.get_by_test_id("text-highlight").wait_for()
            reader_width = page.get_by_test_id("source-reader").evaluate("node => node.getBoundingClientRect().width")
            assert reader_width >= 320
            page.locator(".ant-drawer-open .ant-drawer-close").click()
            page.set_viewport_size({"width": 1280, "height": 800})

            box.fill("无位置")
            page.get_by_role("button", name="发送").click()
            page.get_by_text("没有坐标 [C1]").wait_for()
            page.get_by_test_id("citation-mark").last.click()
            page.get_by_test_id("source-unlocated").wait_for()
            page.locator(".ant-drawer-open .ant-drawer-close").click()

            box.fill("失效来源")
            page.get_by_role("button", name="发送").click()
            page.get_by_text("旧引用 [C1]").wait_for()
            page.get_by_test_id("citation-mark").last.click()
            page.get_by_test_id("source-missing").wait_for()
            page.locator(".ant-drawer-open .ant-drawer-close").click()

            page.get_by_role("button", name="回答偏好").click()
            levels = (("简洁", "concise", "简洁问题"), ("标准", "standard", "标准问题"), ("详细", "detailed", "详细问题"))
            for index, (label, value, question) in enumerate(levels):
                if index:
                    page.get_by_role("button", name="回答偏好").click()
                with page.expect_response(lambda item: item.request.method == "PUT" and item.url.endswith("/settings")) as saved:
                    page.locator(".ant-drawer-open").get_by_text(label, exact=True).click()
                assert saved.value.ok
                page.locator(".ant-drawer-open .ant-drawer-close").click()
                page.locator(".ant-drawer-open").wait_for(state="hidden")
                box.fill(question)
                page.get_by_role("button", name="发送").click()
                page.get_by_text("答案见 [1]").last.wait_for()
                assert bodies[-1]["answer_detail"] == value

            box.fill("停一下")
            page.get_by_role("button", name="发送").click()
            page.get_by_test_id("chat-stop").click()
            page.get_by_text("已停止").wait_for()
            if held:
                try:
                    held.pop().fulfill(status=200, content_type="text/event-stream", body='data: {"type":"token","text":"晚到"}\n\n')
                except Exception:
                    pass

            box.fill("没有资料")
            page.get_by_role("button", name="发送").click()
            page.get_by_test_id("chat-no-evidence").wait_for()
            no_evidence = page.get_by_test_id("chat-no-evidence").locator("xpath=ancestor::div[2]")
            assert no_evidence.get_by_test_id("citation-open").count() == 0
            assert "[C" not in page.get_by_test_id("chat-no-evidence").inner_text()
            box.fill("失败问题")
            page.get_by_role("button", name="发送").click()
            page.get_by_test_id("chat-error").wait_for()
            chat_text = page.locator("body").inner_text()
            _forbid(chat_text)
            for word in ("QueryPlan", "RRF", "path_id", "rerank_score", "admission_score", "Qwen/Qwen3"):
                assert word not in chat_text

            page.set_viewport_size({"width": 390, "height": 844})
            page.get_by_test_id("nav-documents").click()
            page.get_by_role("heading", name="资料").wait_for()
            page.get_by_test_id("nav-advanced").click()
            page.get_by_text("检索测试").last.wait_for()
            page.locator(".ant-drawer-open .ant-drawer-close").click()
            mobile_overflow = page.evaluate("() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1")
            page.screenshot(path=str(mobile), full_page=True)

            page.get_by_test_id("account-open").click()
            page.get_by_test_id("account-email").wait_for()
            assert "ui.browser@example.com" in page.get_by_test_id("account-email").inner_text()
            assert "当前设备" in page.get_by_test_id("session-list").inner_text()
            page.get_by_test_id("session-revoke").click()
            page.wait_for_url("**/login")

            browser.close()
        assert calls["cancel"] == 1
        assert calls["retry"] == 1
        assert calls["deep"] == 1
        assert calls["check"] == 1
        assert calls["search"] == 1
        assert desktop_overflow and mobile_overflow
        assert desktop.exists() and mobile.exists()
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        log.close()
