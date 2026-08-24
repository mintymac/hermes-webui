#!/usr/bin/env python3
"""Migrate Hermes WebUI sessions from JSON sidecars to SQLite.

Usage:
    python3 scripts/migrate_sessions_to_sqlite.py /path/to/webui-mvp/sessions [--commit]

Dry-run (default) is fully non-mutating: the staged database is built in a
temporary directory, never inside the session directory.

With --commit the database is built STAGED as ``sessions.db.migrating`` next
to the final path (same directory, so the publish ``os.replace`` is atomic).
The staged file is seeded ``created_by=migration, migration_complete=0`` in
its own constructor commit, so a crash at ANY point leaves a database the
WebUI refuses to activate — and it never becomes ``sessions.db`` until it is
verified through a fresh connection, marked ``migration_complete=1``,
checkpointed, fsynced, and atomically renamed over the final path. Only then
are the original JSON files moved to ``json-backup/``.

Crash-point test hooks: setting HERMES_MIGRATE_CRASH_AFTER to one of
``pre_schema``, ``schema``, ``markers``, ``create``, ``copy``, ``verify``, or
``publish`` exits with code 42 at that stage (after the named stage has
completed) — used by tests/test_migration_staged.py. ``schema`` and
``markers`` fire inside the store constructor (api.webui_session_sqlite
exposes the same hook) so the pre-marker window is covered too.
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


def _fsync_dir(path: Path) -> None:
    """fsync a directory so a rename/creation inside it is durable."""
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_file(path: Path) -> None:
    with open(path, "rb") as f:
        os.fsync(f.fileno())


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


def _verify_fresh(json_files: list[Path], session_dir: Path, db_name: str) -> int:
    """Re-verify every session through a FRESH store (new connection, not
    the writer's thread-local). Returns the failure count."""
    verify = WebUISqliteSessionDB(session_dir=session_dir, db_name=db_name)
    try:
        failed = 0
        for path in json_files:
            data = _load_json_session(path)
            if data is None:
                failed += 1
                continue
            loaded = verify.read_session(data["session_id"])
            if loaded is None or not _sessions_equal(data, loaded):
                print(f"FAIL {data['session_id']}: fresh-connection verification failed")
                failed += 1
        return failed
    finally:
        verify.close()


def _publish(
    db: WebUISqliteSessionDB,
    json_files: list[Path],
    session_dir: Path,
    *,
    db_name: str,
    staged_name: str,
) -> int:
    """Publish the staged database: verify fresh, mark complete, make it
    durable, then atomically rename it onto the final path. The staged file
    never becomes sessions.db before every step here has succeeded."""
    failed = _verify_fresh(json_files, session_dir, staged_name)
    if failed:
        print(f"FAILURES: {failed}")
        return 1
    _crash_check("verify")

    db.set_meta("migration_complete", "1")
    conn = db._conn()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.commit()
    staged_path = db.db_path
    _fsync_file(staged_path)
    wal_path = Path(str(staged_path) + "-wal")
    if wal_path.exists():
        _fsync_file(wal_path)
    db.close()
    _fsync_dir(session_dir)

    # Publish BEFORE moving sidecars: a crash after publication leaves a
    # fully-populated active database with the JSON files still in place.
    _crash_check("publish")
    os.replace(staged_path, session_dir / db_name)
    _fsync_dir(session_dir)

    backup_dir = session_dir / "json-backup"
    backup_dir.mkdir(exist_ok=True)
    for path in json_files:
        dest = backup_dir / path.name
        shutil.move(str(path), str(dest))
    print(f"Moved {len(json_files)} JSON files to {backup_dir}")
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

    if not args.commit:
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
            print("Dry-run complete. Pass --commit to move JSON files to json-backup/.")
            return 0

    final_path = session_dir / args.db_name
    staged_name = args.db_name + ".migrating"
    staged_path = session_dir / staged_name

    # Staged leftovers from an interrupted run carry no authority and are
    # rebuilt from scratch.
    for suffix in ("", "-wal", "-shm"):
        Path(str(staged_path) + suffix).unlink(missing_ok=True)

    if final_path.exists():
        probe = WebUISqliteSessionDB(session_dir=session_dir, db_name=args.db_name)
        try:
            active = probe.is_active()
        finally:
            probe.close()
        if active:
            print(f"ERROR: {final_path} already exists and is active; refusing to commit over it")
            return 1
        # Inactive final (e.g. an interrupted older in-place migration):
        # leave it alone; publication replaces it atomically only after the
        # staged database verifies.

    _crash_check("pre_schema")
    # Staged from creation: created_by=migration, migration_complete=0 is
    # seeded in the constructor's own commit, before any row.
    db = WebUISqliteSessionDB(
        session_dir=session_dir, db_name=staged_name, created_by="migration"
    )
    _crash_check("create")
    t0 = time.time()
    migrated, failed = _run_copy(db, json_files)
    elapsed = time.time() - t0
    print(f"\nMigrated {migrated}/{len(json_files)} sessions in {elapsed:.2f}s")
    if failed:
        print(f"FAILURES: {failed}")
        return 1
    return _publish(db, json_files, session_dir, db_name=args.db_name, staged_name=staged_name)


if __name__ == "__main__":
    raise SystemExit(main())
