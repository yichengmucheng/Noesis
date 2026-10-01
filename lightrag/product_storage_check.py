"""扫描任务和存储是否一致。

--repair-safe 只恢复能够证明的备份，或把过期租约收回队列。
不会猜测 owner_id 或 kb_id，也不会把无归属数据分配给用户。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lightrag.product_db import open_store
from lightrag.product_scope import scope_visible
from lightrag.product_storage import _STORE_FILES, _read_json, load_shell


def _corrupt(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return True
    return False


def scan(working_dir: Path, kb_id: str = "") -> dict:
    working_dir = Path(working_dir)
    shell = load_shell(working_dir)
    issues = []
    for name in ("product_shell.json", "semantic_cache.json", *_STORE_FILES.values()):
        path = working_dir / name
        if _corrupt(path):
            bak = path.with_name(path.name + ".bak")
            issues.append({
                "kind": "corrupt_json",
                "file": name,
                "repair": "restore_backup" if bak.exists() and not _corrupt(bak) else "",
            })
    quarantined = 0
    chunks = _read_json(working_dir / "kv_store_text_chunks.json") or {}
    if isinstance(chunks, dict):
        for key, row in chunks.items():
            if not isinstance(row, dict):
                continue
            if scope_visible(row, kb_id or str(row.get("kb_id") or "unknown"), shell.get("file_bindings"), shell.get("doc_index")):
                continue
            if not row.get("kb_id") and not row.get("doc_id") and not row.get("file_path"):
                quarantined += 1
                issues.append({"kind": "quarantine", "id": str(key), "repair": ""})
    try:
        store = open_store(working_dir / "product_jobs.sqlite")
        expired = store.count_expired()
    except Exception:
        expired = 0
    if expired:
        issues.append({"kind": "expired_lease", "count": expired, "repair": "requeue"})
    return {"working_dir": str(working_dir), "kb_id": kb_id, "issues": issues, "quarantine": quarantined}


def repair_safe(working_dir: Path, kb_id: str = "") -> dict:
    working_dir = Path(working_dir)
    restored = []
    for name in ("product_shell.json", "semantic_cache.json", *_STORE_FILES.values()):
        path = working_dir / name
        bak = path.with_name(path.name + ".bak")
        if _corrupt(path) and bak.exists() and not _corrupt(bak):
            path.write_bytes(bak.read_bytes())
            restored.append(name)
    try:
        open_store(working_dir / "product_jobs.sqlite").recover_expired()
    except Exception:
        pass
    report = scan(working_dir, kb_id)
    report["restored"] = restored
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="检查产品存储和任务是否一致")
    parser.add_argument("--working-dir", required=True)
    parser.add_argument("--kb-id", default="")
    parser.add_argument("--repair-safe", action="store_true")
    args = parser.parse_args(argv)
    folder = Path(args.working_dir)
    report = repair_safe(folder, args.kb_id) if args.repair_safe else scan(folder, args.kb_id)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
