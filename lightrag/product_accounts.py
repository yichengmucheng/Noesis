"""产品账户、登录会话和启动检查。

密码使用标准库 scrypt，明文和刷新令牌原文都不写入 product_shell.json。
PRODUCT_AUTH 未开启时，登录接口返回 404，现有控制台保持匿名可用。
"""

from __future__ import annotations

import hashlib
import hmac
import codecs
import io
import logging
import os
import re
import secrets
import time
import zipfile
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Header, HTTPException, Request, Response
from pydantic import BaseModel

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_PUBLIC_FIELDS = ("user_id", "email", "created_at", "disabled")
LOCAL_OWNER_ID = "local-owner"
_DEFAULT_TOKEN_SECRET = "lightrag-jwt-default-secret"
_ACCESS_SECONDS = 15 * 60
_REFRESH_DAYS = 7
CURRENT_USER: ContextVar[dict[str, Any] | None] = ContextVar("product_current_user", default=None)
WRITE_KB_ID: ContextVar[str] = ContextVar("product_write_kb", default="")
WRITE_OWNER_ID: ContextVar[str] = ContextVar("product_write_owner", default="")
WRITE_DOC_ID: ContextVar[str] = ContextVar("product_write_doc", default="")
logger = logging.getLogger("lightrag.product")


class CredentialBody(BaseModel):
    email: str
    password: str
    device_label: str = ""


class RefreshBody(BaseModel):
    refresh_token: str = ""


class PasswordBody(BaseModel):
    old_password: str
    new_password: str


class AccountError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def ensure_users(data: dict[str, Any]) -> list[dict[str, Any]]:
    users = data.get("users")
    if not isinstance(users, list):
        users = []
        data["users"] = users
    return users


def public_user(user: dict[str, Any]) -> dict[str, Any]:
    return {field: user.get(field) for field in _PUBLIC_FIELDS}


def hash_password(password: str) -> str:
    if len(password) < 8 or not re.search(r"[A-Za-z]", password):
        raise AccountError("weak_password", "密码至少 8 位，且包含字母")
    salt = os.urandom(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=32,
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n_text, r_text, p_text, salt_hex, digest_hex = str(stored).split("$")
        if scheme != "scrypt":
            return False
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=bytes.fromhex(salt_hex),
            n=int(n_text),
            r=int(r_text),
            p=int(p_text),
            dklen=len(expected),
        )
        return hmac.compare_digest(actual, expected)
    except (TypeError, ValueError):
        return False


def register_user(data: dict[str, Any], email: str, password: str) -> dict[str, Any]:
    normalized = email.strip().lower()
    if not _EMAIL_RE.match(normalized):
        raise AccountError("invalid_email", "邮箱格式不正确")
    users = ensure_users(data)
    if any(str(item.get("email") or "").lower() == normalized for item in users):
        raise AccountError("email_taken", "该邮箱已注册")
    user = {
        "user_id": uuid4().hex,
        "email": normalized,
        "password_hash": hash_password(password),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "disabled": False,
        "session_version": 0,
    }
    users.append(user)
    return public_user(user)


def authenticate(data: dict[str, Any], email: str, password: str) -> dict[str, Any] | None:
    normalized = email.strip().lower()
    for user in ensure_users(data):
        if str(user.get("email") or "").lower() != normalized:
            continue
        if user.get("disabled"):
            return None
        if verify_password(password, str(user.get("password_hash") or "")):
            return public_user(user)
        return None
    return None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime | None = None) -> str:
    return (moment or _now()).isoformat()


def product_auth_enabled() -> bool:
    flag = os.getenv("PRODUCT_AUTH", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    if flag in {"0", "false", "no", "off"}:
        return False
    return os.getenv("APP_ENV", "development").strip().lower() == "production"


def production_startup_error() -> str | None:
    if os.getenv("APP_ENV", "development").strip().lower() != "production":
        return None
    if not product_auth_enabled():
        return "生产模式必须开启产品登录（PRODUCT_AUTH）"
    secret = os.getenv("TOKEN_SECRET", "").strip()
    if not secret or secret == _DEFAULT_TOKEN_SECRET:
        return "生产模式必须设置独立的 TOKEN_SECRET"
    if os.getenv("CORS_ORIGINS", "*").strip() in {"", "*"}:
        return "生产模式不能把 CORS_ORIGINS 设为 *"
    return None


def token_secret() -> str:
    return os.getenv("TOKEN_SECRET", "").strip() or _DEFAULT_TOKEN_SECRET


def upload_limits() -> tuple[int, int]:
    try:
        megabytes = int(os.getenv("MAX_UPLOAD_MB", "20") or "20")
    except ValueError:
        megabytes = 20
    try:
        count = int(os.getenv("MAX_FILES_PER_KB", "100") or "100")
    except ValueError:
        count = 100
    return max(1, megabytes) * 1024 * 1024, max(1, count)


def upload_capability() -> dict[str, Any]:
    max_bytes, max_files = upload_limits()
    try:
        max_pages = int(os.getenv("MAX_PDF_PAGES", "200") or "200")
    except ValueError:
        max_pages = 200
    dimension = str(os.getenv("EMBEDDING_DIM") or "").strip()
    return {
        "max_upload_bytes": max_bytes,
        "max_files_per_kb": max_files,
        "max_pdf_pages": max(1, max_pages),
        "extensions": [
            ".pdf", ".png", ".jpg", ".jpeg", ".docx", ".xlsx", ".pptx",
            ".txt", ".md", ".markdown", ".csv", ".json", ".html", ".htm",
        ],
        "embedding_model": os.getenv("EMBEDDING_MODEL") or "",
        "rerank_model": os.getenv("RERANK_MODEL") or "",
        "llm_model": os.getenv("LLM_MODEL") or "",
        "embedding_dimension": int(dimension) if dimension.isdigit() else 0,
    }


def content_matches_extension(name: str, head: bytes) -> bool:
    sample = head[:64]
    if sample.startswith(b"MZ") or sample.startswith(b"\x7fELF"):
        return False
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        return sample.startswith(b"%PDF")
    if ext == ".png":
        return sample.startswith(b"\x89PNG")
    if ext in {".jpg", ".jpeg"}:
        return sample.startswith(b"\xff\xd8\xff")
    if ext in {".docx", ".xlsx", ".pptx"}:
        return sample.startswith(b"PK")
    if ext in {".txt", ".md", ".markdown", ".csv", ".json", ".html", ".htm"}:
        if b"\x00" in sample:
            return False
        try:
            codecs.getincrementaldecoder("utf-8")().decode(sample, final=False)
        except UnicodeDecodeError:
            return False
    return True


def pdf_page_count(payload: bytes) -> int:
    return len(re.findall(br"/Type\s*/Page(?!s)", payload))


def office_package_safe(payload: bytes) -> bool:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as package:
            names = set(package.namelist())
            if "[Content_Types].xml" not in names:
                return False
            infos = package.infolist()
            if len(infos) > 2000:
                return False
            if sum(item.file_size for item in infos) > 100 * 1024 * 1024:
                return False
    except zipfile.BadZipFile:
        return False
    return True


def inspect_upload(name: str, payload: bytes) -> None:
    max_bytes, _count = upload_limits()
    try:
        max_pages = int(os.getenv("MAX_PDF_PAGES", "200") or "200")
    except ValueError:
        max_pages = 200
    if len(payload) > max_bytes:
        raise AccountError("too_large", "文件超过大小限制")
    ext = Path(name).suffix.lower()
    if ext in {".zip", ".exe", ".dll", ".js", ".bat", ".cmd", ".ps1"}:
        raise AccountError("rejected", "不接受该文件类型")
    if not content_matches_extension(name, payload[:64]):
        raise AccountError("mismatch", "文件内容和扩展名不一致")
    if ext == ".pdf" and pdf_page_count(payload) > max(1, max_pages):
        raise AccountError("too_many_pages", "PDF 页数超过限制")
    if ext in {".docx", ".xlsx", ".pptx"} and not office_package_safe(payload):
        raise AccountError("rejected", "文件包不安全或内容不符")


_RATE_HITS: dict[str, list[float]] = {}


def rate_limited(bucket: str, *, record: bool = True) -> bool:
    try:
        limit = int(os.getenv("AUTH_RATE_LIMIT", "8") or "8")
    except ValueError:
        limit = 8
    now = time.monotonic()
    recent = [item for item in _RATE_HITS.get(bucket, []) if now - item < 60]
    if len(recent) >= max(1, limit):
        _RATE_HITS[bucket] = recent
        return True
    if record:
        recent.append(now)
    _RATE_HITS[bucket] = recent
    return False


def transfer_local_owner(
    data: dict[str, Any],
    target_user_id: str,
    cache_entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not target_user_id or target_user_id == LOCAL_OWNER_ID:
        raise AccountError("invalid_target", "需要一个真实用户")
    moved: list[str] = []
    for kb in data.get("kbs") or []:
        if not isinstance(kb, dict):
            continue
        if (kb.get("owner_id") or LOCAL_OWNER_ID) == LOCAL_OWNER_ID:
            kb["owner_id"] = target_user_id
            moved.append(str(kb.get("id") or ""))
    moved_ids = set(moved)
    updated = 0
    for entry in cache_entries or []:
        if entry.get("kb_id") in moved_ids:
            entry["user_id"] = target_user_id
            updated += 1
    return {"moved_kb_ids": moved, "cache_updated": updated}


def transfer_local_owner_if_configured(
    data: dict[str, Any],
    user: dict[str, Any],
    cache_entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    email = os.getenv("PRODUCT_OWNER_EMAIL", "").strip().lower()
    if not email or str(user.get("email") or "") != email:
        return {"moved_kb_ids": [], "cache_updated": 0}
    return transfer_local_owner(data, str(user.get("user_id") or ""), cache_entries)


def cookie_secure() -> bool:
    flag = os.getenv("COOKIE_SECURE", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    if flag in {"0", "false", "no", "off"}:
        return False
    return os.getenv("APP_ENV", "development").strip().lower() == "production"


REFRESH_COOKIE = "kb_refresh"


def local_owner_view() -> dict[str, Any]:
    return {
        "user_id": LOCAL_OWNER_ID,
        "email": "local@localhost",
        "created_at": "",
        "disabled": False,
    }


def actor_id() -> str:
    user = CURRENT_USER.get()
    if user and user.get("user_id"):
        return str(user["user_id"])
    return LOCAL_OWNER_ID


def stamp_kb(payload: dict[str, Any]) -> dict[str, Any]:
    stamped = dict(payload)
    kb_id = WRITE_KB_ID.get()
    if kb_id:
        parts = [part for part in str(stamped.get("kb_id") or "").split("<SEP>") if part]
        if kb_id not in parts:
            parts.append(kb_id)
        stamped["kb_id"] = "<SEP>".join(parts)
    owner = WRITE_OWNER_ID.get()
    if owner and not stamped.get("owner_id"):
        stamped["owner_id"] = owner
    doc_id = WRITE_DOC_ID.get() or str(stamped.get("full_doc_id") or "")
    if doc_id and not stamped.get("doc_id"):
        stamped["doc_id"] = doc_id
    return stamped


def migrate_shell_data(data: dict[str, Any]) -> bool:
    changed = False
    if not isinstance(data.get("users"), list):
        ensure_users(data)
        changed = True
    if not isinstance(data.get("device_sessions"), list):
        data["device_sessions"] = []
        changed = True
    if not isinstance(data.get("purge_jobs"), list):
        data["purge_jobs"] = []
        changed = True
    for kb in data.get("kbs") or []:
        if isinstance(kb, dict) and not kb.get("owner_id"):
            kb["owner_id"] = LOCAL_OWNER_ID
            changed = True
            logger.info("已有知识库 %s 归属迁移用户 %s", kb.get("id"), LOCAL_OWNER_ID)
    return changed


def visible_kbs(data: dict[str, Any], owner_id: str) -> list[dict[str, Any]]:
    items = []
    for kb in data.get("kbs") or []:
        if not isinstance(kb, dict) or kb.get("deleted_at"):
            continue
        if (kb.get("owner_id") or LOCAL_OWNER_ID) == owner_id:
            items.append(kb)
    return items


def issue_access_token(user_id: str, session_id: str = "", version: int = 0) -> str:
    import jwt

    now = int(_now().timestamp())
    payload = {
        "sub": user_id,
        "role": "product",
        "sid": session_id,
        "ver": int(version),
        "iat": now,
        "exp": now + _ACCESS_SECONDS,
    }
    token = jwt.encode(payload, token_secret(), algorithm="HS256")
    return token if isinstance(token, str) else token.decode("utf-8")


def decode_access_payload(token: str) -> dict[str, Any]:
    import jwt

    try:
        payload = jwt.decode(token, token_secret(), algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise AccountError("invalid_token", "未登录") from exc
    if payload.get("role") != "product" or not payload.get("sub"):
        raise AccountError("invalid_token", "未登录")
    return payload


def decode_access_token(token: str) -> str:
    return str(decode_access_payload(token)["sub"])


def access_is_current(data: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any] | None:
    user_id = str(payload.get("sub") or "")
    user = next(
        (item for item in ensure_users(data) if item.get("user_id") == user_id and not item.get("disabled")),
        None,
    )
    if user is None:
        return None
    if int(payload.get("ver") or 0) != int(user.get("session_version") or 0):
        return None
    session_id = str(payload.get("sid") or "")
    if not session_id:
        return None
    for session in _sessions(data):
        if session.get("session_id") == session_id and session.get("user_id") == user_id and _session_active(session):
            return user
    return None


def _sessions(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = data.get("device_sessions")
    if not isinstance(rows, list):
        rows = []
        data["device_sessions"] = rows
    return rows


def _session_active(session: dict[str, Any]) -> bool:
    if session.get("revoked_at"):
        return False
    try:
        expires = datetime.fromisoformat(str(session.get("expires_at") or ""))
    except ValueError:
        return False
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    return expires > _now()


def issue_refresh_token(data: dict[str, Any], user_id: str, device_label: str = "") -> tuple[str, str]:
    raw = secrets.token_urlsafe(32)
    session_id = uuid4().hex
    _sessions(data).append(
        {
            "session_id": session_id,
            "user_id": user_id,
            "token_hash": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "device_label": (device_label or "未知设备")[:80],
            "created_at": _iso(),
            "last_used_at": _iso(),
            "expires_at": _iso(_now() + timedelta(days=_REFRESH_DAYS)),
            "revoked_at": None,
        }
    )
    return raw, session_id


def _find_refresh_any(data: dict[str, Any], raw: str) -> dict[str, Any] | None:
    digest = hashlib.sha256(str(raw).encode("utf-8")).hexdigest()
    for session in _sessions(data):
        if session.get("token_hash") == digest:
            return session
    return None


def revoke_all_sessions(data: dict[str, Any], user_id: str) -> None:
    for user in ensure_users(data):
        if user.get("user_id") == user_id:
            user["session_version"] = int(user.get("session_version") or 0) + 1
    for session in _sessions(data):
        if session.get("user_id") == user_id and not session.get("revoked_at"):
            session["revoked_at"] = _iso()


def rotate_refresh_token(data: dict[str, Any], raw: str) -> tuple[dict[str, Any], str, str]:
    session = _find_refresh_any(data, raw)
    if session is None:
        raise AccountError("invalid_token", "未登录")
    user_id = str(session.get("user_id") or "")
    if session.get("revoked_at") or not _session_active(session):
        if session.get("revoked_at"):
            revoke_all_sessions(data, user_id)
        raise AccountError("invalid_token", "未登录")
    session["revoked_at"] = _iso()
    user = next((item for item in ensure_users(data) if item.get("user_id") == user_id), None)
    if user is None or user.get("disabled"):
        raise AccountError("invalid_token", "未登录")
    refresh, session_id = issue_refresh_token(data, user_id, str(session.get("device_label") or ""))
    return public_user(user), refresh, session_id


def revoke_refresh_token(data: dict[str, Any], user_id: str, raw: str) -> None:
    session = _find_refresh_any(data, raw)
    if session is None or not _session_active(session):
        return
    if session.get("user_id") != user_id:
        raise AccountError("not_found", "会话不存在")
    session["revoked_at"] = _iso()


def change_password(data: dict[str, Any], user_id: str, old_password: str, new_password: str) -> None:
    user = next((item for item in ensure_users(data) if item.get("user_id") == user_id), None)
    if user is None or not verify_password(old_password, str(user.get("password_hash") or "")):
        raise AccountError("bad_login", "邮箱或密码错误")
    user["password_hash"] = hash_password(new_password)
    revoke_all_sessions(data, user_id)


def list_device_sessions(
    data: dict[str, Any],
    user_id: str,
    current_session_id: str = "",
) -> list[dict[str, Any]]:
    rows = []
    for session in _sessions(data):
        if session.get("user_id") != user_id or not _session_active(session):
            continue
        created = session.get("created_at")
        rows.append(
            {
                "session_id": session.get("session_id"),
                "device_label": session.get("device_label") or "未知设备",
                "created_at": created,
                "last_used_at": session.get("last_used_at") or created,
                "expires_at": session.get("expires_at"),
                "is_current": bool(current_session_id) and session.get("session_id") == current_session_id,
            }
        )
    return rows


def revoke_device_session(data: dict[str, Any], user_id: str, session_id: str) -> None:
    for session in _sessions(data):
        if session.get("session_id") != session_id:
            continue
        if session.get("user_id") != user_id:
            raise AccountError("not_found", "会话不存在")
        session["revoked_at"] = _iso()
        return
    raise AccountError("not_found", "会话不存在")


def _apply_refresh_cookie(response, raw: str) -> None:
    response.set_cookie(
        REFRESH_COOKIE,
        raw,
        httponly=True,
        secure=cookie_secure(),
        samesite="lax",
        max_age=_REFRESH_DAYS * 24 * 3600,
        path="/api/v1/auth",
    )


def _clear_refresh_cookie(response) -> None:
    response.delete_cookie(
        REFRESH_COOKIE,
        path="/api/v1/auth",
        httponly=True,
        samesite="lax",
        secure=cookie_secure(),
    )


def _issue_for_user(data: dict[str, Any], user: dict[str, Any], device_label: str) -> tuple[str, str]:
    stored = next(item for item in ensure_users(data) if item.get("user_id") == user["user_id"])
    refresh, session_id = issue_refresh_token(data, user["user_id"], device_label)
    access = issue_access_token(user["user_id"], session_id, int(stored.get("session_version") or 0))
    return access, refresh


def create_auth_router(read_data, write_data):
    auth = APIRouter()

    def _http(exc: AccountError) -> HTTPException:
        status = 400
        if exc.code == "email_taken":
            status = 409
        elif exc.code in {"bad_login", "invalid_token"}:
            status = 401
        elif exc.code == "rate_limited":
            status = 429
        elif exc.code in {"too_large"}:
            status = 413
        elif exc.code == "not_found":
            status = 404
        return HTTPException(status_code=status, detail=str(exc))

    def _require_auth_enabled() -> None:
        if not product_auth_enabled():
            raise HTTPException(status_code=404, detail="登录未开启")

    def _require_csrf(header: str | None) -> None:
        if header != "1":
            raise HTTPException(status_code=403, detail="缺少请求头")

    async def _user_from_header(authorization: str | None) -> dict[str, Any]:
        if not product_auth_enabled():
            return local_owner_view()
        raw = ""
        if authorization and authorization.lower().startswith("bearer "):
            raw = authorization[7:].strip()
        try:
            payload = decode_access_payload(raw) if raw else {}
        except AccountError as exc:
            raise _http(exc) from exc
        if not payload:
            raise HTTPException(status_code=401, detail="未登录")
        data = await read_data()
        user = access_is_current(data, payload)
        if user is None:
            raise HTTPException(status_code=401, detail="未登录")
        return public_user(user)

    @auth.get("/auth/status")
    async def auth_status():
        return {"enabled": product_auth_enabled()}

    @auth.post("/auth/register")
    async def register(body: CredentialBody, response: Response, x_kb_request: str | None = Header(default=None)):
        _require_auth_enabled()
        _require_csrf(x_kb_request)
        if rate_limited(f"register:{body.email.strip().lower()}"):
            raise HTTPException(status_code=429, detail="请求过于频繁")
        data = await read_data()
        try:
            user = register_user(data, body.email, body.password)
            access, refresh = _issue_for_user(data, user, body.device_label)
            transfer_local_owner_if_configured(data, user)
        except AccountError as exc:
            raise _http(exc) from exc
        await write_data(data)
        _apply_refresh_cookie(response, refresh)
        return {"access_token": access, "user": user}

    @auth.post("/auth/login")
    async def login(body: CredentialBody, response: Response, x_kb_request: str | None = Header(default=None)):
        _require_auth_enabled()
        _require_csrf(x_kb_request)
        bucket = f"login:{body.email.strip().lower()}"
        if rate_limited(bucket, record=False):
            raise HTTPException(status_code=429, detail="请求过于频繁")
        data = await read_data()
        user = authenticate(data, body.email, body.password)
        if user is None:
            rate_limited(bucket)
            raise HTTPException(status_code=401, detail="邮箱或密码错误")
        access, refresh = _issue_for_user(data, user, body.device_label)
        transfer_local_owner_if_configured(data, user)
        await write_data(data)
        _apply_refresh_cookie(response, refresh)
        return {"access_token": access, "user": user}

    @auth.post("/auth/refresh")
    async def refresh(request: Request, response: Response, x_kb_request: str | None = Header(default=None)):
        _require_auth_enabled()
        _require_csrf(x_kb_request)
        data = await read_data()
        try:
            user, rotated, session_id = rotate_refresh_token(data, request.cookies.get(REFRESH_COOKIE, ""))
            stored = next(item for item in ensure_users(data) if item.get("user_id") == user["user_id"])
            access = issue_access_token(user["user_id"], session_id, int(stored.get("session_version") or 0))
        except AccountError as exc:
            raise _http(exc) from exc
        await write_data(data)
        _apply_refresh_cookie(response, rotated)
        return {"access_token": access, "user": user}

    @auth.post("/auth/logout")
    async def logout(request: Request, response: Response, authorization: str | None = Header(default=None), x_kb_request: str | None = Header(default=None)):
        _require_auth_enabled()
        _require_csrf(x_kb_request)
        user = await _user_from_header(authorization)
        data = await read_data()
        revoke_refresh_token(data, user["user_id"], request.cookies.get(REFRESH_COOKIE, ""))
        await write_data(data)
        _clear_refresh_cookie(response)
        return {"status": "ok"}

    @auth.post("/auth/logout-all")
    async def logout_all(response: Response, authorization: str | None = Header(default=None), x_kb_request: str | None = Header(default=None)):
        _require_auth_enabled()
        _require_csrf(x_kb_request)
        user = await _user_from_header(authorization)
        data = await read_data()
        revoke_all_sessions(data, user["user_id"])
        await write_data(data)
        _clear_refresh_cookie(response)
        return {"status": "ok"}

    @auth.post("/auth/password")
    async def update_password(body: PasswordBody, response: Response, authorization: str | None = Header(default=None), x_kb_request: str | None = Header(default=None)):
        _require_auth_enabled()
        _require_csrf(x_kb_request)
        user = await _user_from_header(authorization)
        data = await read_data()
        try:
            change_password(data, user["user_id"], body.old_password, body.new_password)
        except AccountError as exc:
            raise _http(exc) from exc
        await write_data(data)
        _clear_refresh_cookie(response)
        return {"status": "ok"}

    @auth.get("/auth/me")
    async def me(authorization: str | None = Header(default=None)):
        return await _user_from_header(authorization)

    @auth.get("/auth/sessions")
    async def sessions(authorization: str | None = Header(default=None)):
        user = await _user_from_header(authorization)
        session_id = ""
        if authorization and authorization.lower().startswith("bearer "):
            try:
                session_id = str(decode_access_payload(authorization[7:].strip()).get("sid") or "")
            except AccountError:
                session_id = ""
        data = await read_data()
        return {"items": list_device_sessions(data, user["user_id"], session_id)}

    @auth.delete("/auth/sessions/{session_id}")
    async def delete_session(session_id: str, authorization: str | None = Header(default=None)):
        user = await _user_from_header(authorization)
        data = await read_data()
        try:
            revoke_device_session(data, user["user_id"], session_id)
        except AccountError as exc:
            raise _http(exc) from exc
        await write_data(data)
        return {"status": "ok"}

    return auth
