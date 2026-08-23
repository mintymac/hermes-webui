"""Backend-agnostic SessionStore contract tests.

Every backend that implements api.session_store.SessionStore must pass this
same suite — that is what proves the abstraction holds. Currently exercised:

- JSON sidecar adapter (api.webui_session_db.WebUIJsonSessionDB)
- SQLite store (api.webui_session_sqlite.WebUISqliteSessionDB)
- Postgres store (api.webui_session_postgres.WebUIPostgresSessionDB), when
  HERMES_TEST_PG_DSN is set and reachable; skipped otherwise.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from api.webui_session_db import WebUIJsonSessionDB
from api.webui_session_sqlite import WebUISqliteSessionDB, StaleSessionWriteError


def _tmp_dir():
    return Path(tempfile.mkdtemp())


def _sample(sid: str) -> dict:
    return {
        "session_id": sid,
        "title": "Contract Session",
        "workspace": "/workspace",
        "model": "gpt-4",
        "created_at": 1000.0,
        "updated_at": 1001.0,
        "messages": [
            {"role": "user", "content": "hello", "timestamp": 1000.0},
            {"role": "assistant", "content": "hi", "timestamp": 1001.0},
        ],
        "tool_calls": [{"name": "fs.read", "args": {"path": "/x"}}],
        "context_messages": [{"role": "system", "content": "ctx"}],
        "anchor_activity_scenes": {},
    }


def _pg_store():
    dsn = os.environ.get("HERMES_TEST_PG_DSN")
    if not dsn:
        pytest.skip("HERMES_TEST_PG_DSN not set")
    try:
        from api.webui_session_postgres import WebUIPostgresSessionDB

        store = WebUIPostgresSessionDB(dsn)
        # The container database persists across tests: start clean.
        conn = store._conn()
        with conn:
            conn.execute(
                "TRUNCATE sessions, messages, tool_calls, context_messages, anchor_scenes"
            )
        return store
    except Exception as e:
        pytest.skip(f"Postgres unavailable: {e}")


@pytest.fixture(params=["json", "sqlite", "postgres"])
def store(request):
    if request.param == "json":
        yield WebUIJsonSessionDB(session_dir=_tmp_dir())
    elif request.param == "sqlite":
        yield WebUISqliteSessionDB(session_dir=_tmp_dir())
    else:
        s = _pg_store()
        yield s
        s.close()


# SQL backends only (the JSON adapter predates these capabilities).
@pytest.fixture(params=["sqlite", "postgres"])
def sql_store(request):
    if request.param == "sqlite":
        yield WebUISqliteSessionDB(session_dir=_tmp_dir())
    else:
        s = _pg_store()
        yield s
        s.close()


def test_write_read_round_trip(store):
    store.write_session(_sample("c1"))
    loaded = store.read_session("c1")
    assert loaded is not None
    assert loaded["session_id"] == "c1"
    assert loaded["title"] == "Contract Session"
    assert [m["role"] for m in loaded["messages"]] == ["user", "assistant"]
    assert loaded["tool_calls"][0]["name"] == "fs.read"
    assert loaded["context_messages"][0]["content"] == "ctx"


def test_session_exists(store):
    store.write_session(_sample("c2"))
    assert store.session_exists("c2") is True
    assert store.session_exists("nope") is False
    assert store.session_exists("../escape") is False


def test_update_metadata_preserves_transcript(store):
    store.write_session(_sample("c3"))
    store.update_metadata("c3", {"composer_draft": {"text": "draft", "files": []}})
    loaded = store.read_session("c3")
    assert loaded["composer_draft"]["text"] == "draft"
    assert len(loaded["messages"]) == 2


def test_update_metadata_rejects_unsafe_fields(store):
    store.write_session(_sample("c4"))
    with pytest.raises(ValueError):
        store.update_metadata("c4", {"messages": []})


def test_delete_session_removes_everything(store):
    store.write_session(_sample("c5"))
    assert store.delete_session("c5") is True
    assert store.session_exists("c5") is False
    assert store.read_session("c5") is None
    assert store.delete_session("c5") is False


def test_list_sessions_ordering(store):
    a = _sample("c-a")
    a["updated_at"] = 100.0
    b = _sample("c-b")
    b["updated_at"] = 200.0
    store.write_session(a)
    store.write_session(b)
    rows = store.list_sessions()
    ids = [r["session_id"] for r in rows]
    assert ids[0] == "c-b"  # newer updated_at first


def test_unknown_top_level_keys_round_trip(store):
    payload = _sample("c6")
    payload["future_field"] = {"nested": [1, 2, 3]}
    store.write_session(payload)
    loaded = store.read_session("c6")
    assert loaded["future_field"] == {"nested": [1, 2, 3]}


def test_is_active(store):
    assert store.is_active() is True


def test_revision_bumps_on_writes_and_deletes(sql_store):
    r0 = sql_store.get_revision()
    sql_store.write_session(_sample("c7"))
    r1 = sql_store.get_revision()
    assert r1 > r0
    sql_store.update_metadata("c7", {"title": "renamed"})
    r2 = sql_store.get_revision()
    assert r2 > r1
    sql_store.delete_session("c7")
    assert sql_store.get_revision() > r2


def test_metadata_only_read_has_count_without_transcript(sql_store):
    sql_store.write_session(_sample("c8"))
    meta = sql_store.read_metadata_only("c8")
    assert meta is not None
    assert meta["message_count"] == 2
    assert "messages" not in meta


def test_generation_cas(sql_store):
    """Durable per-session generation CAS on every SQL backend."""
    sql_store.write_session(_sample("c9"))  # generation 1
    with pytest.raises(StaleSessionWriteError):
        sql_store.write_session(_sample("c9"))  # no lineage
    sql_store.write_session(_sample("c9"), expected_generation=1)  # bump -> 2
    assert sql_store.read_session("c9")["generation"] == 2
    with pytest.raises(StaleSessionWriteError):
        sql_store.write_session(_sample("c9"), expected_generation=1)  # stale
    sql_store.write_session(_sample("c9"), force=True)  # deliberate heal
    assert sql_store.read_session("c9")["generation"] == 3


def test_null_key_presence(sql_store):
    """Explicitly-None fields read back as present None (JSON parity)."""
    payload = _sample("c10")
    payload["personality"] = None
    payload["project_id"] = None
    sql_store.write_session(payload)
    loaded = sql_store.read_session("c10")
    assert "personality" in loaded and loaded["personality"] is None
    assert "project_id" in loaded and loaded["project_id"] is None
    sql_store.update_metadata("c10", {"personality": "set", "threshold_tokens": None})
    reloaded = sql_store.read_session("c10")
    assert reloaded["personality"] == "set"
    assert "threshold_tokens" in reloaded and reloaded["threshold_tokens"] is None


def test_cutover_marker_blocks_incomplete_migration(sql_store):
    sql_store.set_meta("created_by", "migration")
    sql_store.set_meta("migration_complete", "0")
    assert sql_store.is_active() is False
    sql_store.set_meta("migration_complete", "1")
    assert sql_store.is_active() is True
