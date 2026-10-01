"""把 local-owner 的知识库交给指定邮箱。可以重复执行。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lightrag.product_accounts import (
    LOCAL_OWNER_ID,
    ensure_users,
    migrate_shell_data,
    transfer_local_owner,
)
from lightrag.product_deletion import consistency_report


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def migrate(working_dir: Path, email: str) -> dict:
    shell_path = working_dir / "product_shell.json"
    cache_path = working_dir / "semantic_cache.json"
    data = _load(shell_path)
    migrate_shell_data(data)
    normalized = email.strip().lower()
    user = next((item for item in ensure_users(data) if item.get("email") == normalized), None)
    if user is None:
        raise SystemExit(f"找不到用户 {normalized}。请先注册，再执行迁移。")
    if user.get("user_id") == LOCAL_OWNER_ID:
        raise SystemExit("不能迁移给 local-owner")
    cache = _load(cache_path)
    entries = cache if isinstance(cache, list) else []
    result = transfer_local_owner(data, str(user["user_id"]), entries)
    _save(shell_path, data)
    if entries:
        _save(cache_path, entries)
    result["email"] = normalized
    return result


def check(working_dir: Path, kb_id: str) -> dict:
    data = _load(working_dir / "product_shell.json")
    cache = _load(working_dir / "semantic_cache.json")
    entries = cache if isinstance(cache, list) else []
    return consistency_report(data, kb_id, entries)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="迁移 local-owner 知识库，或检查删除后是否还有残留")
    parser.add_argument("--working-dir", required=True)
    parser.add_argument("--email", default="")
    parser.add_argument("--check-kb", default="")
    parser.add_argument("--deep-check-kb", default="")
    parser.add_argument("--input-dir", default="")
    args = parser.parse_args(argv)
    folder = Path(args.working_dir)
    if args.deep_check_kb:
        from lightrag.product_storage import deep_check

        report = deep_check(folder, args.deep_check_kb, Path(args.input_dir) if args.input_dir else None)
        print(json.dumps(report, ensure_ascii=False))
        return
    if args.check_kb:
        print(json.dumps(check(folder, args.check_kb), ensure_ascii=False))
        return
    if not args.email:
        raise SystemExit("请提供 --email 或 --check-kb")
    print(json.dumps(migrate(folder, args.email), ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
