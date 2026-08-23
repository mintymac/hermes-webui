#!/usr/bin/env python3
"""Migrate Hermes WebUI sessions from JSON sidecars to SQLite.

Usage:
    python3 scripts/migrate_sessions_to_sqlite.py /path/to/webui-mvp/sessions [--commit]

Dry-run (default) is fully non-mutating: the staged database is built in a
temporary directory, never inside the session directory.

With --commit the real sessions.db is created inside the session directory
staged as ``created_by=migration, migration_complete=0`` from the very first
statement, so a crash at ANY point leaves a database the WebUI refuses to
activate. Only after every session round-trip-verifies is the database
published (``migration_complete=1``), and only then are the original JSON
files moved to ``json-backup/``.

Crash-point test hooks: setting HERMES_MIGRATE_CRASH_AFTER to one of
``create``, ``copy``, ``verify``, or ``publish`` exits with code 42 at that
stage (after the named stage has completed) — used by
tests/test_migration_staged.py.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Allow running from repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from api.webui_session_sqlite import WebUISqliteSessionDB, _is_safe_session_id

CRASH_EXIT = 42


def _crash_check(stage: str) -> None:
    if os.environ.get("HERMES_MIGRATE_CRASH_AFTER") == stage:
        print(f"CRASH-INJECTED after {stage}")
        sys.exit(CRASH_EXIT)


def _load_json_session(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"SKIP {path.name}: cannot read/parse: {e}")
        return None
    if not isinstance(data, dict):
        print(f"SKIP {path.name}: not a dict")
        return None
    sid = data.get("session_id") or path.stem
    if not _is_safe_session_id(sid):
        print(f"SKIP {path.name}: unsafe session_id {sid!r}")
        return None
    if "session_id" not in data:
        data["session_id"] = sid
    return data


def _canon(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sessions_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Structural round-trip comparison, preserving key presence.

    Explicit-None keys must exist in both sides (the store preserves
    presence via null_fields_json / JSONB semantics); the previous
    implementation skipped None values and could not catch dropped keys.
    """
    for key in ("messages", "tool_calls", "context_messages"):
        if _canon(a.get(key)) != _canon(b.get(key)):
            return False
    ignored = {"messages", "tool_calls", "context_messages", "last_message_at", "message_count"}
    for key, value in a.items():
        if key in ignored:
            continue
        if key not in b:
            return False
        if _canon(value) != _canon(b[key]):
            return False
    return True


def _run_copy(db: WebUISqliteSessionDB, json_files: list[Path]) -> tuple[int, int]:
    migrated = 0
    failed = 0
    for path in json_files:
        data = _load_json_session(path)
        if data is None:
            failed += 1
            continue
        sid = data["session_id"]
        # force: this offline tool deliberately rebuilds rows from the JSON
        # sidecars, which are the source of truth for the cutover.
        db.write_session(data, force=True)
        loaded = db.read_session(sid)
        if loaded is None or not _sessions_equal(data, loaded):
            print(f"FAIL {sid}: round-trip verification failed")
            failed += 1
            continue
        migrated += 1
        print(f"OK   {sid}")
        _crash_check("copy")  # fires after the first successful row
    return migrated, failed


def _finish(db: WebUISqliteSessionDB, json_files: list[Path], session_dir: Path, *, commit: bool) -> int:
    if commit:
        # Publish BEFORE moving sidecars: a crash after publication leaves a
        # fully-populated active database with the JSON files still in place.
        db.set_meta("migration_complete", "1")
        _crash_check("publish")
        backup_dir = session_dir / "json-backup"
        backup_dir.mkdir(exist_ok=True)
        for path in json_files:
            dest = backup_dir / path.name
            shutil.move(str(path), str(dest))
        print(f"Moved {len(json_files)} JSON files to {backup_dir}")
    else:
        print("Dry-run complete. Pass --commit to move JSON files to json-backup/.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate WebUI sessions from JSON to SQLite")
    parser.add_argument("session_dir", type=Path, help="Path to webui-mvp/sessions directory")
    parser.add_argument("--commit", action="store_true", help="Move JSON files to json-backup/ after verification")
    parser.add_argument("--db-name", default="sessions.db", help="SQLite database filename")
    args = parser.parse_args()

    session_dir = args.session_dir.expanduser().resolve()
    if not session_dir.is_dir():
        print(f"ERROR: {session_dir} is not a directory")
        return 1

    json_files = sorted(p for p in session_dir.glob("*.json") if not p.name.startswith("_"))
    if not json_files:
        print("No session JSON files found.")
        return 0

    print(f"Found {len(json_files)} JSON session files in {session_dir}")

    if args.commit:
        # Staged from creation: created_by=migration, migration_complete=0 is
        # seeded in the constructor's first transaction, before any row.
        db = WebUISqliteSessionDB(
            session_dir=session_dir, db_name=args.db_name, created_by="migration"
        )
    else:
        # Dry-run is non-mutating: stage in a temp dir, never in session_dir.
        with tempfile.TemporaryDirectory() as td:
            db = WebUISqliteSessionDB(session_dir=Path(td), created_by="migration")
            _crash_check("create")
            t0 = time.time()
            migrated, failed = _run_copy(db, json_files)
            elapsed = time.time() - t0
            print(f"\nMigrated {migrated}/{len(json_files)} sessions in {elapsed:.2f}s")
            if failed:
                print(f"FAILURES: {failed}")
                return 1
            _crash_check("verify")
            return _finish(db, json_files, session_dir, commit=False)

    _crash_check("create")
    t0 = time.time()
    migrated, failed = _run_copy(db, json_files)
    elapsed = time.time() - t0
    print(f"\nMigrated {migrated}/{len(json_files)} sessions in {elapsed:.2f}s")
    if failed:
        print(f"FAILURES: {failed}")
        return 1
    _crash_check("verify")
    return _finish(db, json_files, session_dir, commit=True)


if __name__ == "__main__":
    raise SystemExit(main())
