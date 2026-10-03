# -*- coding: utf-8 -*-
"""用真实浏览器验证 PRODUCT_AUTH=1 的登录闭环。不读取项目 .env，不占用 9621。"""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[2]
AUTH_PORT = 9622
OPEN_PORT = 9623


def _free(port: int) -> None:
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", port))
    except OSError as exc:
        raise RuntimeError(f"端口 {port} 已被占用，测试不会使用 9621") from exc
    finally:
        sock.close()


def _env(root: Path, port: int, auth: str, app_env: str) -> dict[str, str]:
    blocked = ("KEY", "SECRET", "TOKEN", "PASSWORD", "COOKIE")
    env = {
        key: value
        for key, value in os.environ.items()
        if not any(word in key.upper() for word in blocked)
    }
    working = root / "rag"
    inputs = root / "inputs"
    working.mkdir(parents=True, exist_ok=True)
    inputs.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "APP_ENV": app_env,
            "PRODUCT_AUTH": auth,
            "TOKEN_SECRET": "playwright-test-secret-32bytes-ok!!",
            "CORS_ORIGINS": f"http://127.0.0.1:{port}",
            "COOKIE_SECURE": "0",
            "AUTH_RATE_LIMIT": "1000",
            "HOST": "127.0.0.1",
            "PORT": str(port),
            "WORKING_DIR": str(working),
            "INPUT_DIR": str(inputs),
            "LLM_BINDING": "openai",
            "LLM_BINDING_HOST": "http://127.0.0.1:9/v1",
            "LLM_MODEL": "test-model",
            "EMBEDDING_BINDING": "openai",
            "EMBEDDING_BINDING_HOST": "http://127.0.0.1:9/v1",
            "EMBEDDING_MODEL": "BAAI/bge-m3"
            if app_env == "production"
            else "test-embed",
            "EMBEDDING_DIM": "8",
            "PYTHONPATH": str(PROJECT),
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    (root / ".env").write_text(
        "\n".join(
            f"{key}={env[key]}"
            for key in (
                "APP_ENV",
                "PRODUCT_AUTH",
                "TOKEN_SECRET",
                "CORS_ORIGINS",
                "COOKIE_SECURE",
                "WORKING_DIR",
                "INPUT_DIR",
                "LLM_BINDING",
                "LLM_MODEL",
                "EMBEDDING_BINDING",
                "EMBEDDING_DIM",
            )
        ),
        encoding="utf-8",
    )
    return env


def _start(root: Path, port: int, auth: str, app_env: str) -> subprocess.Popen:
    _free(port)
    env = _env(root, port, auth, app_env)
    log = open(root / "server.log", "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        [sys.executable, "-m", "lightrag.api.lightrag_server"],
        cwd=str(root),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    proc._gate_log = log
    return proc


def _log_tail(proc: subprocess.Popen) -> str:
    handle = getattr(proc, "_gate_log", None)
    if handle:
        handle.flush()
    path = Path(handle.name) if handle else None
    if path and path.exists():
        return path.read_text(encoding="utf-8", errors="replace")[-2000:]
    return ""


def _wait_health(port: int, proc: subprocess.Popen) -> None:
    import urllib.request

    deadline = time.time() + 90
    last = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(_log_tail(proc))
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2
            ) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last = str(exc)
        time.sleep(1)
    raise RuntimeError(last or _log_tail(proc))


def _stop(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="session")
def auth_base(tmp_path_factory):
    root = tmp_path_factory.mktemp("auth-on")
    proc = _start(root, AUTH_PORT, "1", "production")
    try:
        _wait_health(AUTH_PORT, proc)
        yield f"http://127.0.0.1:{AUTH_PORT}"
    finally:
        _stop(proc)


@pytest.fixture(scope="session")
def open_base(tmp_path_factory):
    root = tmp_path_factory.mktemp("auth-off")
    proc = _start(root, OPEN_PORT, "0", "development")
    try:
        _wait_health(OPEN_PORT, proc)
        yield f"http://127.0.0.1:{OPEN_PORT}"
    finally:
        _stop(proc)


def _playwright():
    from playwright.sync_api import sync_playwright

    return sync_playwright()


def _open_login(page, base: str) -> None:
    page.goto(base + "/console/login", wait_until="domcontentloaded")
    page.get_by_test_id("login-card").wait_for()


def _to_register(page) -> None:
    page.get_by_test_id("auth-switch").click()
    page.get_by_test_id("auth-submit").filter(has_text="注册").wait_for()


def _submit_auth(page) -> None:
    page.get_by_test_id("auth-submit").click()


def test_production_refuses_auth_off(tmp_path):
    proc = _start(tmp_path, 9624, "0", "production")
    try:
        code = proc.wait(timeout=40)
    finally:
        _stop(proc)
    output = _log_tail(proc)
    assert code != 0
    assert "必须开启产品登录" in output
    assert "sk-" not in output


def test_unauthenticated_console_goes_to_login(auth_base):
    with _playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.goto(auth_base + "/console/", wait_until="networkidle")
        page.wait_for_url("**/login")
        assert page.get_by_test_id("kb-home").count() == 0
        status = page.request.get(auth_base + "/api/v1/kb")
        assert status.status == 401
        browser.close()


def test_register_stores_refresh_only_in_cookie(auth_base):
    email = "ada.browser@example.com"
    with _playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context()
        page = context.new_page()
        _open_login(page, auth_base)
        _to_register(page)
        page.get_by_label("邮箱").fill(email)
        page.get_by_label("密码").fill("correct-horse")
        with page.expect_response(
            lambda item: item.url.endswith("/auth/register") and item.status == 200
        ) as caught:
            _submit_auth(page)
        body = caught.value.json()
        assert "refresh_token" not in body
        assert body.get("access_token")
        page.get_by_test_id("kb-home").wait_for()
        cookie = next(
            item for item in context.cookies() if item["name"] == "kb_refresh"
        )
        assert cookie["httpOnly"] is True
        assert cookie["sameSite"] in {"Lax", "lax"}
        stored = page.evaluate(
            "() => JSON.stringify(localStorage) + JSON.stringify(sessionStorage)"
        )
        html = page.content()
        assert "kb-refresh" not in stored
        assert cookie["value"] not in html
        assert cookie["value"] not in stored
        browser.close()


def test_register_errors_do_not_leak_internals(auth_base):
    with _playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_page()
        page = context
        _open_login(page, auth_base)
        _to_register(page)
        page.get_by_label("邮箱").fill("ada.browser@example.com")
        page.get_by_label("密码").fill("correct-horse")
        _submit_auth(page)
        page.get_by_text("该邮箱已注册").wait_for()
        page.get_by_label("邮箱").fill("not-an-email")
        _submit_auth(page)
        page.get_by_text("请输入邮箱").wait_for()
        page.get_by_label("邮箱").fill("weak.browser@example.com")
        page.get_by_label("密码").fill("12345678")
        _submit_auth(page)
        page.get_by_text("密码至少 8 位，且包含字母").wait_for()
        denied = page.request.post(
            auth_base + "/api/v1/auth/register",
            data=json.dumps(
                {"email": "other.browser@example.com", "password": "correct-horse"}
            ),
            headers={"Content-Type": "application/json"},
        )
        assert denied.status == 403
        text = denied.text()
        assert "缺少请求头" in text
        assert "Traceback" not in text
        assert "token" not in text.lower()
        browser.close()


def test_login_isolation_reload_and_single_refresh(auth_base):
    email = "isle.browser@example.com"
    with _playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context()
        page = context.new_page()
        _open_login(page, auth_base)
        page.get_by_label("邮箱").fill("missing.browser@example.com")
        page.get_by_label("密码").fill("correct-horse")
        _submit_auth(page)
        page.get_by_text("邮箱或密码错误").wait_for()
        page.get_by_label("邮箱").fill(email)
        page.get_by_label("密码").fill("wrong-horse-1")
        _submit_auth(page)
        assert page.get_by_text("邮箱或密码错误").count() >= 1
        page.get_by_label("邮箱").fill(email)
        page.get_by_label("密码").fill("correct-horse")
        _to_register(page)
        with page.expect_response(
            lambda item: item.url.endswith("/auth/register")
        ) as caught:
            _submit_auth(page)
        assert caught.value.status == 200
        page.get_by_test_id("kb-home").wait_for()
        page.get_by_role("button", name="新建知识库").click()
        page.get_by_label("知识库名称").fill("A-private")
        page.get_by_role("button", name="创建", exact=True).click()
        page.get_by_text("A-private").wait_for()
        other = browser.new_context()
        other_page = other.new_page()
        _open_login(other_page, auth_base)
        _to_register(other_page)
        other_page.get_by_label("邮箱").fill("beau.browser@example.com")
        other_page.get_by_label("密码").fill("correct-horse")
        _submit_auth(other_page)
        other_page.get_by_test_id("kb-home").wait_for()
        assert other_page.get_by_text("A-private").count() == 0
        refreshes = []
        page.on(
            "request",
            lambda request: refreshes.append(request.url)
            if "/auth/refresh" in request.url
            else None,
        )
        page.reload(wait_until="networkidle")
        page.get_by_test_id("kb-home").wait_for()
        assert page.url.endswith("/console/") or page.url.rstrip("/").endswith(
            "/console"
        )
        assert len(refreshes) == 1
        seen = {"kb": 0, "me": 0}

        def handle(route):
            url = route.request.url
            if any(
                part in url
                for part in (
                    "/auth/refresh",
                    "/auth/login",
                    "/auth/register",
                    "/auth/status",
                )
            ):
                route.continue_()
                return
            key = (
                "me"
                if url.endswith("/auth/me") or "/auth/me" in url
                else "kb"
                if "/api/v1/kb" in url
                else ""
            )
            if key and seen[key] == 0:
                seen[key] = 1
                route.fulfill(
                    status=401,
                    content_type="application/json",
                    body='{"detail":"未登录"}',
                )
                return
            route.continue_()

        before = len(refreshes)
        page.route("**/api/v1/**", handle)
        page.reload(wait_until="networkidle")
        page.get_by_test_id("kb-home").wait_for()
        assert len(refreshes) - before == 2
        page.unroute("**/api/v1/**")

        def reject(route):
            if (
                "/auth/refresh" in route.request.url
                or "/api/v1/kb" in route.request.url
            ):
                route.fulfill(
                    status=401,
                    content_type="application/json",
                    body='{"detail":"未登录"}',
                )
                return
            route.continue_()

        page.route("**/api/v1/**", reject)
        failure_before = len(refreshes)
        page.reload(wait_until="networkidle")
        page.wait_for_url("**/login")
        assert len(refreshes) - failure_before < 5
        browser.close()


def test_logout_password_and_all_devices(auth_base):
    email = "cara.browser@example.com"
    with _playwright() as playwright:
        browser = playwright.chromium.launch()
        first = browser.new_context()
        second = browser.new_context()
        page = first.new_page()
        other = second.new_page()
        for target, address in ((page, email), (other, email)):
            _open_login(target, auth_base)
            if address == email and target == page:
                _to_register(target)
                target.get_by_label("邮箱").fill(address)
                target.get_by_label("密码").fill("correct-horse")
                _submit_auth(target)
            else:
                target.get_by_label("邮箱").fill(address)
                target.get_by_label("密码").fill("correct-horse")
                _submit_auth(target)
            target.get_by_test_id("kb-home").wait_for()
        page.get_by_test_id("account-open").click()
        page.get_by_role("button", name="退出", exact=True).click()
        page.wait_for_url("**/login")
        assert not any(item["name"] == "kb_refresh" for item in first.cookies())
        other.reload(wait_until="networkidle")
        other.get_by_test_id("kb-home").wait_for()
        blocked = page.request.get(auth_base + "/api/v1/kb")
        assert blocked.status == 401
        _open_login(page, auth_base)
        page.get_by_label("邮箱").fill(email)
        page.get_by_label("密码").fill("correct-horse")
        _submit_auth(page)
        page.get_by_test_id("kb-home").wait_for()
        page.get_by_test_id("account-open").click()
        page.get_by_role("button", name="全部退出").click()
        page.wait_for_url("**/login")
        other.reload(wait_until="networkidle")
        other.wait_for_url("**/login")
        stale = other.request.post(
            auth_base + "/api/v1/auth/refresh",
            headers={"X-KB-Request": "1"},
        )
        assert stale.status == 401
        page.get_by_test_id("login-card").wait_for()
        page.get_by_label("邮箱").fill(email)
        page.get_by_label("密码").fill("correct-horse")
        _submit_auth(page)
        page.get_by_test_id("kb-home").wait_for()
        _open_login(other, auth_base)
        other.get_by_label("邮箱").fill(email)
        other.get_by_label("密码").fill("correct-horse")
        _submit_auth(other)
        other.get_by_test_id("kb-home").wait_for()
        page.get_by_test_id("account-open").click()
        page.get_by_role("button", name="修改密码").click()
        page.get_by_label("当前密码").fill("correct-horse")
        page.get_by_label("新密码").fill("correct-horse-2")
        page.get_by_role("button", name="保存", exact=True).click()
        page.wait_for_url("**/login")
        other.reload(wait_until="networkidle")
        other.wait_for_url("**/login")
        _open_login(page, auth_base)
        page.get_by_label("邮箱").fill(email)
        page.get_by_label("密码").fill("correct-horse")
        _submit_auth(page)
        page.get_by_text("邮箱或密码错误").wait_for()
        page.get_by_label("密码").fill("correct-horse-2")
        _submit_auth(page)
        page.get_by_test_id("kb-home").wait_for()
        browser.close()


def test_auth_off_console_stays_open(open_base):
    with _playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.goto(open_base + "/console/", wait_until="networkidle")
        page.get_by_test_id("kb-home").wait_for()
        assert "/login" not in page.url
        browser.close()
