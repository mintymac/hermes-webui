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

Cutover coordination:

- An exclusive, non-blocking cross-process lock (``<session_dir>/.migrating.lock``)
  serializes concurrent migration runs (flock / msvcrt, mirroring
  ``api.models._cleanup_manifest_process_lock``). The lock file is never
  unlinked: unlinking under contention splits later acquirers across inodes.
  Both dry-run and --commit take it so a dry-run never reports against a
  moving target. The live WebUI does NOT take this lock; defense against a
  live writer is the source-identity CAS below.
- Each copied sidecar's exact bytes are hashed (SHA-256) at copy time. A stat
  signature would false-positive on the WebUI's temp-file + os.replace writer
  (every save swaps the inode and bumps mtime_ns even for identical content);
  the content hash gives exact-preimage semantics with no false positives.
- Immediately before publication, every migrated sidecar is re-hashed: any
  missing/changed file refuses the whole run (exit 1), because a drift proves
  a live writer is serving this session dir mid-cutover — publishing a mixed
  snapshot is precisely the lost-write hole. The staged file is retained for
  the next run; no sidecar is moved.
- After publication, each migrated sidecar is re-hashed again and moved to
  ``json-backup/`` only if it is still the exact preimage; drifted files stay
  in place as operator-visible evidence (exit 3) and are NOT auto-re-imported
  (a force re-import would launder an arbitrary intermediate writer over a
  fenced row, breaking the writer-token story; the SQL row is authoritative).
- Sidecars that fail to parse or verify are skipped per-session (the staged
  rows are excised: both the sessions row AND the session_incarnations row,
  so the sid keeps its pristine absent-row first-create path) and are left
  in place; publication proceeds on the migrated set.

Exit codes: 0 success (a SKIP report is allowed) - 1 refusal (drift
pre-publish, active db, lock held, nothing migrated, bad dir) - 3 published
but retirement incomplete - 42 crash hooks.

Crash-point test hooks: setting HERMES_MIGRATE_CRASH_AFTER to one of
``pre_schema``, ``schema``, ``markers``, ``create``, ``copy``, ``verify``, or
``publish`` exits with code 42 at that stage (after the named stage has
completed) — used by tests/test_migration_staged.py. ``schema`` and
``markers`` fire inside the store constructor (api.webui_session_sqlite
exposes the same hook) so the pre-marker window is covered too.

Test-only mutation hooks (same pattern as the crash hooks):
``HERMES_MIGRATE_TEST_MUTATE=<sid>`` rewrites that sidecar (byte-different,
still valid JSON) immediately BEFORE the pre-publish CAS, and
``HERMES_MIGRATE_TEST_MUTATE_AFTER_PUBLISH=<sid>`` does the same immediately
AFTER the publish ``os.replace``, before the retirement move loop. These are
the only deterministic way to exercise the copy→CAS and CAS→move windows
from a subprocess test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    import fcntl as _fcntl
except ImportError:  # Windows
    _fcntl = None
try:
    import msvcrt as _msvcrt
except ImportError:  # POSIX
    _msvcrt = None

# Allow running from repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from api.webui_session_sqlite import WebUISqliteSessionDB, _is_safe_session_id

CRASH_EXIT = 42

# A (sid, path, size, sha256) source-identity snapshot for one migrated
# sidecar; the hash covers the exact bytes read during the copy pass.
Snapshot = tuple[str, Path, int, str]


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


@contextmanager
def _migration_lock(session_dir: Path):
    """Exclusive, non-blocking cross-process lock for the migration cutover.

    A held lock means a live concurrent run: refuse (exit 1). flock releases
    automatically on process death, so there is no stale-lock recovery. The
    lock file is NEVER unlinked — unlinking while another process is waiting
    can split later acquirers across different inodes and defeat the lock
    (same rationale as api.models._cleanup_manifest_process_lock). If no
    lock primitive exists, fail closed rather than running an uncoordinated
    cutover.
    """
    lock_path = session_dir / ".migrating.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(fd, "r+b", buffering=0) as lock_file:
        if _fcntl is not None:
            try:
                _fcntl.flock(lock_file.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
            except OSError:
                print("ERROR: another migration holds .migrating.lock")
                raise SystemExit(1)
            try:
                yield
            finally:
                _fcntl.flock(lock_file.fileno(), _fcntl.LOCK_UN)
            return

        if _msvcrt is not None:
            if os.fstat(lock_file.fileno()).st_size == 0:
                lock_file.write(b"\0")
            lock_file.seek(0)
            try:
                _msvcrt.locking(  # type: ignore[attr-defined]
                    lock_file.fileno(), _msvcrt.LK_LOCK, 1  # type: ignore[attr-defined]
                )
            except OSError:
                print("ERROR: another migration holds .migrating.lock")
                raise SystemExit(1)
            try:
                yield
            finally:
                lock_file.seek(0)
                _msvcrt.locking(  # type: ignore[attr-defined]
                    lock_file.fileno(), _msvcrt.LK_UNLCK, 1  # type: ignore[attr-defined]
                )
            return

        print("ERROR: no cross-process lock primitive available; refusing uncoordinated cutover")
        raise SystemExit(1)


def _load_json_session(path: Path) -> tuple[dict[str, Any], bytes] | None:
    """Parse a sidecar; returns (data, raw_bytes) or None (already SKIP-logged)."""
    try:
        raw = path.read_bytes()
        data = json.loads(raw.decode("utf-8"))
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
    return data, raw


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


def _excise(db: WebUISqliteSessionDB, sid: str) -> None:
    """Remove a session from the STAGED artifact entirely: the authority row
    AND the session_incarnations row (direct SQL — the staged file is this
    tool's own). Both must go: leaving only the authority row retired would
    brick the sid post-cutover (RetiredSessionWriteError on its first
    JSON-fallback save); removing both restores the pristine absent-row
    first-create path. The sessions delete cascades to the transcript tables
    (the store connection enables PRAGMA foreign_keys)."""
    conn = db._conn()
    with conn:
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
        conn.execute("DELETE FROM session_incarnations WHERE session_id = ?", (sid,))


def _run_copy(db: WebUISqliteSessionDB, json_files: list[Path]) -> tuple[list[Snapshot], int]:
    """Copy pass, partitioned: each sidecar either migrates (recorded with a
    source-identity snapshot) or is skipped with a SKIP line. A single bad
    legacy file must not abort the whole run."""
    migrated: list[Snapshot] = []
    skipped = 0
    for path in json_files:
        loaded = _load_json_session(path)
        if loaded is None:
            skipped += 1
            continue
        data, raw = loaded
        sid = data["session_id"]
        # force: this offline tool deliberately rebuilds rows from the JSON
        # sidecars, which are the source of truth for the cutover.
        db.write_session(data, force=True)
        loaded_row = db.read_session(sid)
        if loaded_row is None or not _sessions_equal(data, loaded_row):
            print(f"SKIP {path.name}: round-trip verification failed")
            _excise(db, sid)
            skipped += 1
            continue
        migrated.append((sid, path, len(raw), hashlib.sha256(raw).hexdigest()))
        print(f"OK   {sid}")
        _crash_check("copy")  # fires after the first successful row
    return migrated, skipped


def _verify_fresh(
    db: WebUISqliteSessionDB,
    migrated: list[Snapshot],
    session_dir: Path,
    db_name: str,
) -> list[Snapshot]:
    """Re-verify every migrated session through a FRESH store (new
    connection, not the writer's thread-local). Sessions that fail are
    excised from the staged db and dropped from the migrated set (a
    per-session skip, not a run abort); publication proceeds on the
    survivors."""
    verify = WebUISqliteSessionDB(session_dir=session_dir, db_name=db_name)
    survivors: list[Snapshot] = []
    try:
        for sid, path, size, sha in migrated:
            loaded = _load_json_session(path)
            ok = False
            if loaded is not None:
                data, _raw = loaded
                if data.get("session_id") == sid:
                    row = verify.read_session(sid)
                    ok = row is not None and _sessions_equal(data, row)
            if ok:
                survivors.append((sid, path, size, sha))
            else:
                if loaded is not None:
                    print(f"SKIP {path.name}: fresh-connection verification failed")
                _excise(db, sid)
        return survivors
    finally:
        verify.close()


def _matches_preimage(path: Path, size: int, sha: str) -> bool:
    """Exact-preimage check: the file still holds the bytes that were copied."""
    try:
        raw = path.read_bytes()
    except OSError:
        return False
    return len(raw) == size and hashlib.sha256(raw).hexdigest() == sha


def _test_mutate(env_var: str, session_dir: Path) -> None:
    """TEST-ONLY hook: rewrite one sidecar (byte-different, still valid JSON)
    so subprocess tests can exercise the copy→CAS and CAS→move windows."""
    sid = os.environ.get(env_var)
    if not sid:
        return
    path = session_dir / f"{sid}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    data["_mutated"] = time.time_ns()
    path.write_text(json.dumps(data), encoding="utf-8")


def _publish(
    db: WebUISqliteSessionDB,
    migrated: list[Snapshot],
    session_dir: Path,
    *,
    db_name: str,
    staged_name: str,
) -> int:
    """Publish the staged database: verify fresh, mark complete, make it
    durable, CAS the sources, then atomically rename onto the final path and
    retire only exact-preimage sidecars. The staged file never becomes
    sessions.db before every step here has succeeded."""
    survivors = _verify_fresh(db, migrated, session_dir, staged_name)
    _crash_check("verify")
    if not survivors:
        # Activating an empty store over live sidecars is an operator error,
        # not a cutover.
        print("FAILURES: no migrated sessions survived verification; refusing to publish")
        return 1

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

    _test_mutate("HERMES_MIGRATE_TEST_MUTATE", session_dir)

    # Pre-publish source-identity CAS: every migrated sidecar must still be
    # the exact preimage that was copied. Any drift proves a live writer is
    # serving this session dir mid-cutover — refuse the whole run rather
    # than publish a mixed snapshot (the lost-write hole). The staged file
    # is retained (the next run rebuilds it), sessions.db is NOT published,
    # and no sidecar is moved. A mid-run deletion of a copied sidecar is
    # also drift: publishing its copied row would resurrect a deleted
    # session.
    drifted = []
    for sid, path, size, sha in survivors:
        if not _matches_preimage(path, size, sha):
            drifted.append(sid)
            print(f"DRIFT {sid}: source changed after copy")
    if drifted:
        print(f"FAILURES: {len(drifted)}")
        return 1

    # Publish BEFORE moving sidecars: a crash after publication leaves a
    # fully-populated active database with the JSON files still in place.
    _crash_check("publish")
    os.replace(staged_path, session_dir / db_name)
    _fsync_dir(session_dir)

    _test_mutate("HERMES_MIGRATE_TEST_MUTATE_AFTER_PUBLISH", session_dir)

    # Post-publish retirement CAS: move only exact preimages. A drifted
    # sidecar stays in place (the newer bytes are preserved on disk as
    # operator-visible evidence) and is NOT auto-re-imported over the
    # fenced, now-authoritative SQL row.
    backup_dir = session_dir / "json-backup"
    backup_dir.mkdir(exist_ok=True)
    moved = 0
    retired_drift = []
    for sid, path, size, sha in survivors:
        if _matches_preimage(path, size, sha):
            dest = backup_dir / path.name
            shutil.move(str(path), str(dest))
            moved += 1
        else:
            retired_drift.append(sid)
            print(f"DRIFTED {sid}: not retired (source changed after publication)")
    print(f"Moved {moved} JSON files to {backup_dir}")
    # Exit 3 (distinct from 1): the database IS published and active; the
    # run's retirement contract is incomplete.
    return 3 if retired_drift else 0


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

    # Serialize concurrent migration processes for the WHOLE cutover:
    # enumeration, copy, verify, mark, checkpoint/fsync, identity CAS,
    # os.replace, and sidecar retirement all happen under the lock. The
    # live WebUI does not take this lock (zero server changes) — the
    # source-identity CAS in _publish is the live-writer defense.
    with _migration_lock(session_dir):
        # *.json ignores dotfiles, so .migrating.lock is never mistaken for
        # a sidecar.
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
                migrated, skipped = _run_copy(db, json_files)
                elapsed = time.time() - t0
                print(f"\nMigrated {len(migrated)}/{len(json_files)} sessions in {elapsed:.2f}s")
                if skipped:
                    print(f"SKIPPED: {skipped}")
                _crash_check("verify")
                print("Dry-run complete. Pass --commit to move JSON files to json-backup/.")
                return 0 if migrated else 1

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
        migrated, skipped = _run_copy(db, json_files)
        elapsed = time.time() - t0
        print(f"\nMigrated {len(migrated)}/{len(json_files)} sessions in {elapsed:.2f}s")
        if skipped:
            print(f"SKIPPED: {skipped}")
        if not migrated:
            print("FAILURES: no sessions migrated; refusing to publish an empty store")
            return 1
        return _publish(db, migrated, session_dir, db_name=args.db_name, staged_name=staged_name)


if __name__ == "__main__":
    raise SystemExit(main())
