"""Staged-migration crash-point tests (gate finding: no unintended authority).

Drives scripts/migrate_sessions_to_sqlite.py as a subprocess with
HERMES_MIGRATE_CRASH_AFTER set, then inspects the session directory and
store activation from this process.

Commit mode builds sessions.db STAGED as sessions.db.migrating and publishes
by atomic os.replace only after the staged file verifies through a fresh
connection, is marked migration_complete=1, checkpointed, and fsynced. No
crash point may leave a selector-visible sessions.db with authority it did
not earn, and a long-lived selector must observe a later publish.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

import api.models as models
from api.webui_session_sqlite import WebUISqliteSessionDB

REPO = Path(__file__).resolve().parents[1]

STAGED_NAME = "sessions.db.migrating"


def _seed_sessions(d: Path, n: int = 3) -> None:
    for i in range(n):
        payload = {
            "session_id": f"mig{i}",
            "title": f"T{i}",
            "workspace": "/w",
            "created_at": 1.0,
            "updated_at": 2.0,
            "personality": None,  # explicit-None presence must survive
            "messages": [{"role": "user", "content": "hello"}],
            "tool_calls": [],
            "context_messages": [],
        }
        (d / f"mig{i}.json").write_text(json.dumps(payload), encoding="utf-8")


def _run_script(d: Path, crash: str | None, commit: bool) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("HERMES_MIGRATE_CRASH_AFTER", None)
    if crash:
        env["HERMES_MIGRATE_CRASH_AFTER"] = crash
    cmd = [sys.executable, str(REPO / "scripts" / "migrate_sessions_to_sqlite.py"), str(d)]
    if commit:
        cmd.append("--commit")
    return subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO))


def _store_active(d: Path, monkeypatch) -> bool:
    monkeypatch.setattr(models, "SESSION_DIR", d)
    monkeypatch.setattr(models, "_sqlite_session_store_instance", None)
    return bool(models._get_sqlite_session_store())


def _meta(d: Path, db_name: str = "sessions.db") -> dict:
    conn = sqlite3.connect(str(d / db_name))
    try:
        return dict(conn.execute("SELECT key, value FROM meta").fetchall())
    finally:
        conn.close()


def test_dry_run_does_not_mutate_session_dir(tmp_path, monkeypatch):
    _seed_sessions(tmp_path)
    result = _run_script(tmp_path, None, commit=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / "sessions.db").exists()
    assert not (tmp_path / STAGED_NAME).exists()
    assert sorted(p.name for p in tmp_path.glob("*.json")) == [f"mig{i}.json" for i in range(3)]


def test_crash_after_pre_schema_leaves_no_db(tmp_path, monkeypatch):
    """pre_schema fires before the store constructor: nothing is created."""
    _seed_sessions(tmp_path)
    result = _run_script(tmp_path, "pre_schema", commit=True)
    assert result.returncode == 42
    assert not (tmp_path / "sessions.db").exists()
    assert not (tmp_path / STAGED_NAME).exists()
    assert _store_active(tmp_path, monkeypatch) is False


def test_crash_after_schema_leaves_unmarked_staged_db(tmp_path, monkeypatch):
    """Pre-marker constructor crash: the staged file has tables but NO
    markers, so nothing can activate it — and a later app open must not
    stamp created_by=app on the leftover."""
    _seed_sessions(tmp_path)
    result = _run_script(tmp_path, "schema", commit=True)
    assert result.returncode == 42
    # The crash window is on the STAGED name: the final path is absent.
    assert not (tmp_path / "sessions.db").exists()
    staged = tmp_path / STAGED_NAME
    assert staged.exists()
    # Unmarked: Commit A (tables) landed, Commit B (markers) never did.
    assert "created_by" not in _meta(tmp_path, STAGED_NAME)
    assert _store_active(tmp_path, monkeypatch) is False

    # Opening the leftover through the store (created_by=app default) must
    # NOT stamp created_by=app: it is a pre-existing, row-less database of
    # unknown origin and stays unmarked (fail closed).
    store = WebUISqliteSessionDB(session_dir=tmp_path, db_name=STAGED_NAME)
    assert store.get_meta("created_by") is None
    assert store.is_active() is False
    store.close()
    assert "created_by" not in _meta(tmp_path, STAGED_NAME)

    # The next migration run deletes the staged leftover and completes.
    result = _run_script(tmp_path, None, commit=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _store_active(tmp_path, monkeypatch) is True


def test_crash_after_markers_leaves_marked_incomplete_staged_db(tmp_path, monkeypatch):
    """Post-marker constructor crash: markers exist but migration_complete=0."""
    _seed_sessions(tmp_path)
    result = _run_script(tmp_path, "markers", commit=True)
    assert result.returncode == 42
    assert not (tmp_path / "sessions.db").exists()
    meta = _meta(tmp_path, STAGED_NAME)
    assert meta.get("created_by") == "migration"
    assert meta.get("migration_complete") == "0"
    assert _store_active(tmp_path, monkeypatch) is False


def test_crash_after_create_leaves_inactive_staged_db(tmp_path, monkeypatch):
    _seed_sessions(tmp_path)
    result = _run_script(tmp_path, "create", commit=True)
    assert result.returncode == 42
    # Staged-only: the final path is never created in place.
    assert not (tmp_path / "sessions.db").exists()
    assert (tmp_path / STAGED_NAME).exists()
    assert _store_active(tmp_path, monkeypatch) is False


def test_crash_after_copy_leaves_partial_inactive_db(tmp_path, monkeypatch):
    _seed_sessions(tmp_path)
    result = _run_script(tmp_path, "copy", commit=True)
    assert result.returncode == 42
    assert not (tmp_path / "sessions.db").exists()
    assert (tmp_path / STAGED_NAME).exists()
    assert _store_active(tmp_path, monkeypatch) is False


def test_crash_after_verify_leaves_complete_inactive_db(tmp_path, monkeypatch):
    _seed_sessions(tmp_path)
    result = _run_script(tmp_path, "verify", commit=True)
    assert result.returncode == 42
    assert not (tmp_path / "sessions.db").exists()
    assert (tmp_path / STAGED_NAME).exists()
    assert _store_active(tmp_path, monkeypatch) is False
    # JSON files untouched — the session dir still serves from sidecars.
    assert sorted(p.name for p in tmp_path.glob("*.json")) == [f"mig{i}.json" for i in range(3)]


def test_crash_after_publish_leaves_unpublished_staged_db(tmp_path, monkeypatch):
    """The publish hook fires BEFORE the rename: a crash there leaves the
    staged file fully verified and marked complete but never visible as
    sessions.db. Re-running the migration rebuilds and publishes cleanly."""
    _seed_sessions(tmp_path)
    result = _run_script(tmp_path, "publish", commit=True)
    assert result.returncode == 42
    assert not (tmp_path / "sessions.db").exists()
    # The staged file was verified, marked complete, checkpointed, fsynced.
    meta = _meta(tmp_path, STAGED_NAME)
    assert meta.get("created_by") == "migration"
    assert meta.get("migration_complete") == "1"
    assert _store_active(tmp_path, monkeypatch) is False
    # Publication happens before sidecar moves, so no session is lost.
    assert sorted(p.name for p in tmp_path.glob("*.json")) == [f"mig{i}.json" for i in range(3)]

    # Recovery: a clean re-run publishes and activates.
    result = _run_script(tmp_path, None, commit=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _store_active(tmp_path, monkeypatch) is True
    store = models._get_sqlite_session_store()
    for i in range(3):
        row = store.read_session(f"mig{i}")
        assert row is not None and len(row["messages"]) == 1
        assert "personality" in row and row["personality"] is None


def test_commit_completes_and_activates(tmp_path, monkeypatch):
    _seed_sessions(tmp_path)
    result = _run_script(tmp_path, None, commit=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / STAGED_NAME).exists()
    assert _store_active(tmp_path, monkeypatch) is True
    assert (tmp_path / "json-backup").is_dir()
    assert sorted(p.name for p in (tmp_path / "json-backup").glob("*.json")) == [
        f"mig{i}.json" for i in range(3)
    ]


def test_commit_refuses_to_overwrite_active_db(tmp_path, monkeypatch):
    """A final sessions.db with authority is never replaced: refuse first."""
    _seed_sessions(tmp_path)
    store = WebUISqliteSessionDB(session_dir=tmp_path)  # created_by=app -> active
    store.close()
    result = _run_script(tmp_path, None, commit=True)
    assert result.returncode == 1
    assert "refusing" in (result.stdout + result.stderr).lower()
    # Untouched: still the app db, still no backup moves.
    assert _meta(tmp_path).get("created_by") == "app"
    assert not (tmp_path / "json-backup").exists()
    assert sorted(p.name for p in tmp_path.glob("*.json")) == [f"mig{i}.json" for i in range(3)]
    assert _store_active(tmp_path, monkeypatch) is True


def test_store_constructor_crash_hooks_without_script(tmp_path, monkeypatch):
    """The store exposes the migration crash hooks so unit tests can hit the
    in-constructor stages directly (env HERMES_MIGRATE_CRASH_AFTER)."""
    monkeypatch.setenv("HERMES_MIGRATE_CRASH_AFTER", "schema")
    with pytest.raises(SystemExit):
        WebUISqliteSessionDB(session_dir=tmp_path, created_by="migration")
    assert "created_by" not in _meta(tmp_path)

    # markers: tables + cutover markers committed, then the hook fires.
    for p in tmp_path.iterdir():
        p.unlink()
    monkeypatch.setenv("HERMES_MIGRATE_CRASH_AFTER", "markers")
    with pytest.raises(SystemExit):
        WebUISqliteSessionDB(session_dir=tmp_path, created_by="migration")
    monkeypatch.delenv("HERMES_MIGRATE_CRASH_AFTER")
    meta = _meta(tmp_path)
    assert meta.get("created_by") == "migration"
    assert meta.get("migration_complete") == "0"

    # The marked-incomplete leftover has no authority, and reopening it as a
    # migration db does not re-run the hooks once the env is cleared.
    store = WebUISqliteSessionDB(session_dir=tmp_path, created_by="migration")
    assert store.is_active() is False
    store.close()


def test_app_open_of_preexisting_empty_db_does_not_stamp(tmp_path, monkeypatch):
    """Fail-closed stamp rule: a pre-existing, row-less, unmarked database
    (the exact leftover of a pre-marker constructor crash on the final path)
    must NOT be stamped created_by=app when the app opens it — that would
    activate an empty store over live JSON sidecars."""
    monkeypatch.setenv("HERMES_MIGRATE_CRASH_AFTER", "schema")
    with pytest.raises(SystemExit):
        WebUISqliteSessionDB(session_dir=tmp_path, created_by="migration")
    monkeypatch.delenv("HERMES_MIGRATE_CRASH_AFTER")
    assert (tmp_path / "sessions.db").exists()

    # The selector denies it and the app open leaves it unmarked.
    assert _store_active(tmp_path, monkeypatch) is False
    store = WebUISqliteSessionDB(session_dir=tmp_path)  # created_by=app default
    assert store.get_meta("created_by") is None
    assert store.is_active() is False
    store.close()
    assert "created_by" not in _meta(tmp_path)


def test_long_lived_selector_sees_publish(tmp_path, monkeypatch):
    """One long-lived selector across publication: an armed negative cache
    must NOT stick after a marked-complete staged file is published over
    sessions.db via os.replace."""
    monkeypatch.setattr(models, "SESSION_DIR", tmp_path)
    monkeypatch.setattr(models, "_sqlite_session_store_instance", None)

    # Arm the negative cache: no sessions.db exists yet.
    assert models._get_sqlite_session_store() is False

    # Build and publish a marked-complete staged file exactly the way the
    # migration script does (mark -> checkpoint -> fsync -> os.replace).
    staged = WebUISqliteSessionDB(session_dir=tmp_path, db_name=STAGED_NAME, created_by="migration")
    staged.write_session(
        {
            "session_id": "pub1",
            "title": "published",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_calls": [],
            "context_messages": [],
        }
    )
    staged.set_meta("migration_complete", "1")
    staged._conn().execute("PRAGMA wal_checkpoint(TRUNCATE)")
    staged._conn().commit()
    staged.close()
    os.replace(tmp_path / STAGED_NAME, tmp_path / "sessions.db")

    # No singleton reset: the same selector must observe the publication.
    store = models._get_sqlite_session_store()
    assert store
    assert store.read_session("pub1") is not None


def test_unrecognized_markers_fail_closed(tmp_path, monkeypatch):
    _seed_sessions(tmp_path)
    assert _run_script(tmp_path, None, commit=True).returncode == 0
    # Unrecognizable markers must deny authority rather than assume it.
    conn = sqlite3.connect(str(tmp_path / "sessions.db"))
    conn.execute("UPDATE meta SET value = 'unknown-origin' WHERE key = 'created_by'")
    conn.commit()
    conn.close()
    assert _store_active(tmp_path, monkeypatch) is False
