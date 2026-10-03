"""应用数据库的备份、恢复和完整性检查命令。"""

from __future__ import annotations

import argparse
from pathlib import Path

from lightrag.product_appdb import appdb_integrity_check, backup_appdb, restore_appdb


def main() -> int:
    parser = argparse.ArgumentParser(description="备份或恢复 Noesis 应用数据库")
    parser.add_argument("--working-dir", required=True, type=Path)
    parser.add_argument("--backup", type=Path, help="输出或输入的 SQLite 备份文件")
    parser.add_argument("--restore", action="store_true", help="从 --backup 恢复")
    parser.add_argument("--check", action="store_true", help="运行 integrity_check")
    args = parser.parse_args()
    if args.restore:
        if not args.backup:
            parser.error("--restore 必须同时提供 --backup")
        restore_appdb(args.working_dir, args.backup)
        print(f"restored: {args.working_dir / 'product_app.sqlite'}")
    elif args.backup:
        backup_appdb(args.working_dir, args.backup)
        print(f"backup: {args.backup}")
    if args.check or not args.restore and not args.backup:
        print(f"integrity_check: {appdb_integrity_check(args.working_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
