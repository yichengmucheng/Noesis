"""产品登录开启后的请求门禁。

公开路径可以匿名访问。产品接口必须带有效 access token。
原生查询、文档、图谱和 Ollama 接口即使带了 token 也不放行，避免绕过知识库归属。
"""

from __future__ import annotations

from lightrag.product_accounts import (
    AccountError,
    decode_access_token,
    product_auth_enabled,
)

_PUBLIC_EXACT = {
    "/",
    "/health",
    "/auth-status",
    "/api/v1/auth/status",
    "/api/v1/auth/register",
    "/api/v1/auth/login",
    "/api/v1/auth/refresh",
}
_NATIVE_PREFIXES = (
    "/documents",
    "/query",
    "/graphs",
    "/graph",
)


def _clean(path: str) -> str:
    path = (path or "/").split("?", 1)[0]
    if len(path) > 1:
        path = path.rstrip("/")
    return path or "/"


def _is_public(path: str) -> bool:
    if path in _PUBLIC_EXACT or path.startswith("/console"):
        return True
    return False


def _is_native_data(path: str) -> bool:
    if path.startswith("/api/") and not path.startswith("/api/v1"):
        return True
    return any(
        path == prefix or path.startswith(prefix + "/") for prefix in _NATIVE_PREFIXES
    )


def _bearer_user(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        return ""
    try:
        return decode_access_token(authorization[7:].strip())
    except AccountError:
        return ""


def access_decision(path: str, authorization: str | None, method: str = "GET") -> str:
    """返回 allow、unauthorized 或 forbidden。认证关闭时一律 allow。"""
    if method.upper() == "OPTIONS" or not product_auth_enabled():
        return "allow"
    cleaned = _clean(path)
    if _is_public(cleaned):
        return "allow"
    user_id = _bearer_user(authorization)
    if _is_native_data(cleaned):
        return "forbidden" if user_id else "unauthorized"
    if not user_id:
        return "unauthorized"
    return "allow"


def install_product_guard(app) -> None:
    from fastapi.responses import JSONResponse

    @app.middleware("http")
    async def product_guard(request, call_next):
        decision = access_decision(
            request.url.path,
            request.headers.get("authorization"),
            request.method,
        )
        if decision == "allow":
            return await call_next(request)
        if decision == "forbidden":
            return JSONResponse(status_code=403, content={"detail": "请使用产品接口"})
        return JSONResponse(status_code=401, content={"detail": "未登录"})
