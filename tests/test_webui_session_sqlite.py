"""SQLite-backed session store regression tests."""
from __future__ import annotations

import tempfile
from pathlib import Path

import api.models as models
import api.webui_session_sqlite as sqlite_db


def _tmp_session_dir():
    return Path(tempfile.mkdtemp())


def _sample_session_dict(sid: str = "test-session") -> dict:
    return {
        "session_id": sid,
        "title": "Test Session",
        "workspace": "/workspace",
        "model": "gpt-4",
        "created_at": 1000.0,
        "updated_at": 1001.0,
        "messages": [
            {"role": "user", "content": "hello", "timestamp": 1000.0},
            {"role": "assistant", "content": "hi", "timestamp": 1001.0},
        ],
        "tool_calls": [],
        "context_messages": [],
        "anchor_activity_scenes": {},
    }


def test_sqlite_store_round_trip():
    d = _tmp_session_dir()
    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    payload = _sample_session_dict("sid-1")
    store.write_session(payload)

    loaded = store.read_session("sid-1")
    assert loaded is not None
    assert loaded["session_id"] == "sid-1"
    assert len(loaded["messages"]) == 2
    assert loaded["messages"][0]["role"] == "user"


def test_sqlite_store_update_metadata():
    d = _tmp_session_dir()
    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    store.write_session(_sample_session_dict("sid-2"))

    store.update_metadata("sid-2", {"composer_draft": {"text": "draft", "files": []}})
    loaded = store.read_session("sid-2")
    assert loaded["composer_draft"]["text"] == "draft"
    assert len(loaded["messages"]) == 2


def test_session_load_uses_sqlite_when_db_exists(monkeypatch):
    d = _tmp_session_dir()
    # Patch global session dir for this test.
    monkeypatch.setattr(models, "SESSION_DIR", d)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", d / "_index.json")
    monkeypatch.setattr(models, "_sqlite_session_store_instance", None)

    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    store.write_session(_sample_session_dict("sid-3"))

    s = models.Session.load("sid-3")
    assert s is not None
    assert s.session_id == "sid-3"
    assert len(s.messages) == 2


def test_persisted_session_ids_includes_sqlite(monkeypatch):
    d = _tmp_session_dir()
    monkeypatch.setattr(models, "SESSION_DIR", d)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", d / "_index.json")
    monkeypatch.setattr(models, "_PERSISTED_SESSION_IDS_CACHE", (None, None, frozenset()))
    monkeypatch.setattr(models, "_sqlite_session_store_instance", None)

    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    store.write_session(_sample_session_dict("sid-4"))

    ids = models._persisted_session_ids_snapshot()
    assert "sid-4" in ids


def test_index_entry_exists_for_sqlite_only(monkeypatch):
    d = _tmp_session_dir()
    monkeypatch.setattr(models, "SESSION_DIR", d)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", d / "_index.json")
    monkeypatch.setattr(models, "_sqlite_session_store_instance", None)

    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    store.write_session(_sample_session_dict("sid-5"))

    assert models._index_entry_exists("sid-5", in_memory_ids=set()) is True
    assert models._index_entry_exists("missing", in_memory_ids=set()) is False


def test_session_save_metadata_sqlite_only(monkeypatch):
    d = _tmp_session_dir()
    monkeypatch.setattr(models, "SESSION_DIR", d)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", d / "_index.json")
    monkeypatch.setattr(models, "_sqlite_session_store_instance", None)

    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    store.write_session(_sample_session_dict("sid-6"))

    s = models.Session.load("sid-6")
    s.save_metadata({"composer_draft": {"text": "quick draft", "files": []}})

    reloaded = models.Session.load("sid-6")
    assert reloaded.composer_draft["text"] == "quick draft"
    assert len(reloaded.messages) == 2
