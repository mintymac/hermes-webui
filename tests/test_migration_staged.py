"""Staged-migration crash-point tests (gate finding: no unintended authority).

Drives scripts/migrate_sessions_to_sqlite.py as a subprocess with
HERMES_MIGRATE_CRASH_AFTER set, then inspects the session directory and
store activation from this process.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import api.models as models

REPO = Path(__file__).resolve().parents[1]


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


def test_dry_run_does_not_mutate_session_dir(tmp_path, monkeypatch):
    _seed_sessions(tmp_path)
    result = _run_script(tmp_path, None, commit=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / "sessions.db").exists()
    assert sorted(p.name for p in tmp_path.glob("*.json")) == [f"mig{i}.json" for i in range(3)]


def test_crash_after_create_leaves_inactive_empty_db(tmp_path, monkeypatch):
    _seed_sessions(tmp_path)
    result = _run_script(tmp_path, "create", commit=True)
    assert result.returncode == 42
    assert (tmp_path / "sessions.db").exists()
    assert _store_active(tmp_path, monkeypatch) is False


def test_crash_after_copy_leaves_partial_inactive_db(tmp_path, monkeypatch):
    _seed_sessions(tmp_path)
    result = _run_script(tmp_path, "copy", commit=True)
    assert result.returncode == 42
    assert _store_active(tmp_path, monkeypatch) is False


def test_crash_after_verify_leaves_complete_inactive_db(tmp_path, monkeypatch):
    _seed_sessions(tmp_path)
    result = _run_script(tmp_path, "verify", commit=True)
    assert result.returncode == 42
    assert _store_active(tmp_path, monkeypatch) is False
    # JSON files untouched — the session dir still serves from sidecars.
    assert sorted(p.name for p in tmp_path.glob("*.json")) == [f"mig{i}.json" for i in range(3)]


def test_crash_after_publish_leaves_active_complete_db(tmp_path, monkeypatch):
    _seed_sessions(tmp_path)
    result = _run_script(tmp_path, "publish", commit=True)
    assert result.returncode == 42
    assert _store_active(tmp_path, monkeypatch) is True
    # Publication happens before sidecar moves, so no session is lost.
    assert sorted(p.name for p in tmp_path.glob("*.json")) == [f"mig{i}.json" for i in range(3)]
    # The published rows are complete (verified before publication).
    store = models._get_sqlite_session_store()
    for i in range(3):
        row = store.read_session(f"mig{i}")
        assert row is not None and len(row["messages"]) == 1
        assert "personality" in row and row["personality"] is None


def test_commit_completes_and_activates(tmp_path, monkeypatch):
    _seed_sessions(tmp_path)
    result = _run_script(tmp_path, None, commit=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _store_active(tmp_path, monkeypatch) is True
    assert (tmp_path / "json-backup").is_dir()
    assert sorted(p.name for p in (tmp_path / "json-backup").glob("*.json")) == [
        f"mig{i}.json" for i in range(3)
    ]


def test_unrecognized_markers_fail_closed(tmp_path, monkeypatch):
    import sqlite3

    _seed_sessions(tmp_path)
    assert _run_script(tmp_path, None, commit=True).returncode == 0
    # Unrecognizable markers must deny authority rather than assume it.
    conn = sqlite3.connect(str(tmp_path / "sessions.db"))
    conn.execute("UPDATE meta SET value = 'unknown-origin' WHERE key = 'created_by'")
    conn.commit()
    conn.close()
    assert _store_active(tmp_path, monkeypatch) is False
