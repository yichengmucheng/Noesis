# -*- coding: utf-8 -*-
import json
import sys
from pathlib import Path

from lightrag.product_accounts import (
    AccountError,
    authenticate,
    ensure_users,
    hash_password,
    public_user,
    register_user,
    verify_password,
)

sys.argv = ["lightrag"]
from lightrag.api.routers.product_shell import ShellStore


def test_register_stores_hash_and_hides_it_from_public_view():
    data = {}
    created = register_user(data, " Ada@Example.com ", "correct-horse")
    assert created["email"] == "ada@example.com"
    assert "password_hash" not in created
    stored = data["users"][0]
    assert stored["password_hash"] != "correct-horse"
    assert stored["password_hash"].startswith("scrypt$")
    assert verify_password("correct-horse", stored["password_hash"])
    assert public_user(stored) == created


def test_duplicate_email_and_weak_password_are_rejected():
    data = {}
    register_user(data, "ada@example.com", "correct-horse")
    try:
        register_user(data, "ADA@example.com", "another-password")
    except AccountError as exc:
        assert exc.code == "email_taken"
    else:
        raise AssertionError("重复邮箱应被拒绝")
    try:
        hash_password("short")
    except AccountError as exc:
        assert exc.code == "weak_password"
    else:
        raise AssertionError("过短密码应被拒绝")


def test_authenticate_rejects_wrong_password_and_disabled_user():
    data = {}
    register_user(data, "ada@example.com", "correct-horse")
    assert authenticate(data, "ada@example.com", "wrong-password") is None
    assert (
        authenticate(data, "ada@example.com", "correct-horse")["email"]
        == "ada@example.com"
    )
    data["users"][0]["disabled"] = True
    assert authenticate(data, "ada@example.com", "correct-horse") is None


def test_shell_store_adds_users_to_old_file(tmp_path: Path):
    path = tmp_path / "product_shell.json"
    path.write_text(json.dumps({"kbs": [], "file_bindings": {}}), encoding="utf-8")
    store = ShellStore(str(tmp_path))
    loaded = store.load()
    assert loaded["users"] == []
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["users"] == []
    assert ensure_users({}) == []
