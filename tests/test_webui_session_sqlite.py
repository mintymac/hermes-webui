"""SQLite-backed session store regression tests."""
from __future__ import annotations

import json
import tempfile
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

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


def test_session_load_falls_back_to_json_when_sqlite_misses_row(monkeypatch):
    """If sessions.db exists but a session was not migrated, JSON sidecar is used."""
    d = _tmp_session_dir()
    monkeypatch.setattr(models, "SESSION_DIR", d)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", d / "_index.json")
    monkeypatch.setattr(models, "_sqlite_session_store_instance", None)

    # Create a JSON-only session.
    sid = "sid-json-only"
    payload = _sample_session_dict(sid)
    (d / f"{sid}.json").write_text(json.dumps(payload), encoding="utf-8")

    # Activating SQLite must not hide the JSON-only session.
    _ = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    s = models.Session.load(sid)
    assert s is not None
    assert s.session_id == sid
    assert len(s.messages) == 2


def test_session_load_falls_back_to_json_on_sqlite_read_error(monkeypatch):
    """A corrupt SQLite row must not block the JSON sidecar fallback.

    Greptile P1: read_session() raising (unreadable message_json, DB read
    error) propagated out of Session.load(), failing mutation requests for
    sessions whose sidecar is still valid. load_metadata_only() already
    degrades on store errors; Session.load() now does the same.
    """
    import sqlite3 as _sq

    d = _tmp_session_dir()
    monkeypatch.setattr(models, "SESSION_DIR", d)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", d / "_index.json")
    monkeypatch.setattr(models, "_sqlite_session_store_instance", None)

    sid = "sid-corrupt"
    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    store.write_session(_sample_session_dict(sid))
    # Sidecar holds the intact copy (e.g. pre-migration backup).
    (d / f"{sid}.json").write_text(
        json.dumps(_sample_session_dict(sid)), encoding="utf-8"
    )

    # Corrupt the SQLite message payload.
    conn = _sq.connect(str(d / "sessions.db"))
    conn.execute(
        "UPDATE messages SET message_json = 'not-json' WHERE session_id = ?",
        (sid,),
    )
    conn.commit()
    conn.close()

    s = models.Session.load(sid)
    assert s is not None
    assert s.session_id == sid
    assert len(s.messages) == 2


def test_save_metadata_falls_back_to_json_for_unmigrated_session(monkeypatch):
    """sessions.db exists but the session is JSON-only: draft autosave must
    persist to the sidecar, not raise KeyError from a zero-row SQLite UPDATE."""
    d = _tmp_session_dir()
    monkeypatch.setattr(models, "SESSION_DIR", d)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", d / "_index.json")
    monkeypatch.setattr(models, "_sqlite_session_store_instance", None)

    sid = "sid-json-draft"
    payload = _sample_session_dict(sid)
    (d / f"{sid}.json").write_text(json.dumps(payload), encoding="utf-8")

    # Activate the SQLite store without migrating this session.
    _ = sqlite_db.WebUISqliteSessionDB(session_dir=d)

    s = models.Session.load(sid)
    assert s is not None
    s.save_metadata({"composer_draft": {"text": "json draft", "files": []}})

    on_disk = json.loads((d / f"{sid}.json").read_text(encoding="utf-8"))
    assert on_disk["composer_draft"]["text"] == "json draft"

    reloaded = models.Session.load(sid)
    assert reloaded.composer_draft["text"] == "json draft"
    assert len(reloaded.messages) == 2


def test_save_metadata_json_fallback_updates_in_memory_session(monkeypatch):
    """Greptile P1 (r3745331728): the JSON-fallback branch of save_metadata()
    must keep the in-memory Session consistent with the sidecar it just wrote.

    The draft route happens to pre-set ``s.composer_draft`` before calling
    save_metadata(), which masks the asymmetry; this test exercises the method
    directly without that pre-set, so it fails if the JSON branch forgets to
    setattr the object (the original bug left ``s.composer_draft`` as ``{}``).
    """
    d = _tmp_session_dir()
    monkeypatch.setattr(models, "SESSION_DIR", d)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", d / "_index.json")
    monkeypatch.setattr(models, "_sqlite_session_store_instance", None)

    sid = "sid-json-mem"
    payload = _sample_session_dict(sid)
    (d / f"{sid}.json").write_text(json.dumps(payload), encoding="utf-8")

    # sessions.db exists but this session was not migrated into it.
    _ = sqlite_db.WebUISqliteSessionDB(session_dir=d)

    s = models.Session.load(sid)
    assert s is not None
    assert s.composer_draft == {}

    # Deliberately do NOT pre-set s.composer_draft: save_metadata() must own
    # keeping the in-memory object consistent with the persisted sidecar.
    s.save_metadata({"composer_draft": {"text": "mem draft", "files": []}})

    # In-memory object must reflect the persisted draft (staleness would leave
    # this as the original empty dict).
    assert s.composer_draft == {"text": "mem draft", "files": []}

    # And the sidecar on disk must match.
    on_disk = json.loads((d / f"{sid}.json").read_text(encoding="utf-8"))
    assert on_disk["composer_draft"]["text"] == "mem draft"


def test_session_exists():
    d = _tmp_session_dir()
    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    store.write_session(_sample_session_dict("sid-exists"))

    assert store.session_exists("sid-exists") is True
    assert store.session_exists("sid-missing") is False
    assert store.session_exists("../escape") is False


def test_update_metadata_does_not_advance_updated_at_for_drafts():
    d = _tmp_session_dir()
    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    store.write_session(_sample_session_dict("sid-draft"))

    loaded = store.read_session("sid-draft")
    original_updated_at = loaded["updated_at"]

    store.update_metadata("sid-draft", {"composer_draft": {"text": "draft", "files": []}})

    reloaded = store.read_session("sid-draft")
    assert reloaded["composer_draft"]["text"] == "draft"
    assert reloaded["updated_at"] == original_updated_at


def test_update_metadata_advances_updated_at_when_requested():
    d = _tmp_session_dir()
    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    store.write_session(_sample_session_dict("sid-touch"))

    new_time = 9999.0
    store.update_metadata("sid-touch", {"updated_at": new_time})

    reloaded = store.read_session("sid-touch")
    assert reloaded["updated_at"] == new_time


def test_save_refuses_metadata_only_sqlite_session(monkeypatch):
    """#1558 P0 guard must also protect the SQLite fast path.

    Session.save() on a session loaded with metadata_only=True would
    otherwise write messages=[] through write_session(), replacing the
    message tables and wiping the transcript.
    """
    d = _tmp_session_dir()
    monkeypatch.setattr(models, "SESSION_DIR", d)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", d / "_index.json")
    monkeypatch.setattr(models, "_sqlite_session_store_instance", None)

    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    store.write_session(_sample_session_dict("sid-1558"))

    s = models.Session.load_metadata_only("sid-1558")
    assert s is not None
    assert getattr(s, "_loaded_metadata_only", False) is True

    try:
        s.save()
    except RuntimeError:
        pass
    else:
        raise AssertionError("save() must refuse metadata-only sessions on the SQLite path")

    reloaded = store.read_session("sid-1558")
    assert reloaded is not None
    assert len(reloaded["messages"]) == 2


def test_sqlite_store_delete_session_removes_all_rows():
    import sqlite3 as _sq

    d = _tmp_session_dir()
    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    payload = _sample_session_dict("sid-del")
    payload["anchor_activity_scenes"] = {"h1": {"scene": "data"}}
    store.write_session(payload)
    assert store.session_exists("sid-del")

    assert store.delete_session("sid-del") is True
    assert store.session_exists("sid-del") is False
    assert store.read_session("sid-del") is None

    # No orphan rows left in any child table.
    conn = _sq.connect(str(d / "sessions.db"))
    try:
        for table in ("sessions", "messages", "tool_calls", "context_messages", "anchor_scenes"):
            col = "scene_hash" if table == "anchor_scenes" else "session_id"
            (count,) = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {col} = ?",
                ("sid-del" if table != "anchor_scenes" else "h1",),
            ).fetchone()
            assert count == 0, f"{table} still has rows for sid-del"
    finally:
        conn.close()

    # Deleting a missing session is a no-op.
    assert store.delete_session("sid-del") is False


def test_delete_route_removes_sqlite_rows_and_index_does_not_resurrect(monkeypatch):
    """POST /api/session/delete must remove SQLite-backed sessions too.

    Migrated sessions have no sidecar; previously the route only unlinked
    the sidecar, leaving the sessions.db row — and a full index rebuild then
    resurrected the deleted session in the sidebar.
    """
    d = _tmp_session_dir()
    _patch_route_state(monkeypatch, d)

    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    store.write_session(_sample_session_dict("sid-del"))

    import api.routes as routes

    # Keep the CLI state.db out of the test; the WebUI store is under test.
    monkeypatch.setattr(models, "delete_cli_session", lambda sid: True)

    captured = _drive_delete_post(monkeypatch, {"session_id": "sid-del"})
    assert captured.get("payload", {}).get("ok") is True
    assert store.session_exists("sid-del") is False

    # A full index rebuild must not resurrect the deleted session.
    index_file = d / "_index.json"
    if index_file.exists():
        index_file.unlink()
    models._write_session_index(updates=None, session_dir=d, session_index_file=index_file)
    entries = json.loads(index_file.read_text())
    assert all(e.get("session_id") != "sid-del" for e in entries)


def test_delete_route_handles_unmigrated_json_session_with_store_active(monkeypatch):
    """Mixed-store delete: sessions.db exists but the session is JSON-only."""
    d = _tmp_session_dir()
    _patch_route_state(monkeypatch, d)

    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)  # activates sessions.db
    sidecar = d / "sid-json.json"
    sidecar.write_text(json.dumps(_sample_session_dict("sid-json")), encoding="utf-8")

    import api.routes as routes

    monkeypatch.setattr(models, "delete_cli_session", lambda sid: True)

    captured = _drive_delete_post(monkeypatch, {"session_id": "sid-json"})
    assert captured.get("payload", {}).get("ok") is True
    assert not sidecar.exists()
    assert store.session_exists("sid-json") is False


def test_pre_compression_snapshot_check_reads_sqlite_store(monkeypatch):
    """_is_pre_compression_snapshot_id must work for migrated sessions.

    The sidebar lineage grouping reads the sidecar directly; a migrated
    (SQLite-only) snapshot parent would look like a non-snapshot and its
    continuation rows would lose their grouping.
    """
    d = _tmp_session_dir()
    _patch_route_state(monkeypatch, d)

    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    snap = _sample_session_dict("snap_parent")
    snap["pre_compression_snapshot"] = True
    store.write_session(snap)
    store.write_session(_sample_session_dict("plain_parent"))

    import api.routes as routes

    assert routes._is_pre_compression_snapshot_id("snap_parent") is True
    assert routes._is_pre_compression_snapshot_id("plain_parent") is False
    assert routes._is_pre_compression_snapshot_id("missing_parent") is False


def _drive_delete_post(monkeypatch, body):
    """Run POST /api/session/delete through routes.handle_post (CSRF bypassed,
    JSON responders captured) and return the captured response."""
    import api.routes as routes

    captured = {}
    monkeypatch.setattr(routes, "_check_csrf", lambda handler: True)
    monkeypatch.setattr(routes, "read_body", lambda handler: body)
    monkeypatch.setattr(
        routes,
        "bad",
        lambda handler, msg, status=400, extra_headers=None: captured.update(
            error=msg, status=status
        )
        or True,
    )
    monkeypatch.setattr(
        routes,
        "j",
        lambda handler, payload, status=200, extra_headers=None, pretty=True: captured.update(
            payload=payload, status=status
        )
        or True,
    )
    handler = SimpleNamespace(command="POST", _safe_webui_print=lambda *_a, **_k: None)
    assert routes.handle_post(handler, SimpleNamespace(path="/api/session/delete")) is True
    return captured


# ── Route-level regressions: POST /api/session/draft through the real
# routes.handle_post dispatcher, the way the composer's 400ms debounced
# auto-save actually reaches this code in production. The unit tests above
# call save_metadata() directly; these prove the store ordering holds
# end-to-end (dispatch → get_session → save_metadata → persistence). ──────


def _patch_route_state(monkeypatch, d):
    """Point models+routes session state at an isolated tmpdir."""
    import api.routes as routes

    sessions = OrderedDict()
    monkeypatch.setattr(models, "SESSION_DIR", d)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", d / "_index.json")
    monkeypatch.setattr(models, "SESSIONS", sessions)
    monkeypatch.setattr(models, "_sqlite_session_store_instance", None)
    monkeypatch.setattr(routes, "SESSION_DIR", d)
    monkeypatch.setattr(routes, "SESSIONS", sessions)


def _drive_draft_post(monkeypatch, body):
    """Run POST /api/session/draft through routes.handle_post (CSRF bypassed,
    JSON responders captured) and return the captured response."""
    import api.routes as routes

    captured = {}
    monkeypatch.setattr(routes, "_check_csrf", lambda handler: True)
    monkeypatch.setattr(routes, "read_body", lambda handler: body)
    monkeypatch.setattr(
        routes,
        "bad",
        lambda handler, msg, status=400, extra_headers=None: captured.update(
            error=msg, status=status
        )
        or True,
    )
    monkeypatch.setattr(
        routes,
        "j",
        lambda handler, payload, status=200, extra_headers=None, pretty=True: captured.update(
            payload=payload, status=status
        )
        or True,
    )
    handler = SimpleNamespace(command="POST", _safe_webui_print=lambda *_a, **_k: None)
    assert routes.handle_post(handler, SimpleNamespace(path="/api/session/draft")) is True
    return captured


def test_draft_route_sqlite_ordering_persists_to_migrated_row(monkeypatch):
    """sessions.db active + session migrated: the draft autosave must update
    only the SQLite sessions row — no JSON sidecar is created, the transcript
    is untouched, and updated_at does not move (a keystroke is not activity)."""
    d = _tmp_session_dir()
    _patch_route_state(monkeypatch, d)

    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    store.write_session(_sample_session_dict("sid-route-sqlite"))

    captured = _drive_draft_post(
        monkeypatch,
        {"session_id": "sid-route-sqlite", "text": "sqlite route draft", "files": []},
    )

    assert captured.get("status") == 200
    assert captured["payload"]["ok"] is True
    assert captured["payload"]["draft"]["text"] == "sqlite route draft"

    row = store.read_session("sid-route-sqlite")
    assert row["composer_draft"]["text"] == "sqlite route draft"
    assert len(row["messages"]) == 2
    assert row["updated_at"] == 1001.0
    assert not (d / "sid-route-sqlite.json").exists()


def test_draft_route_json_fallback_ordering_for_unmigrated_session(monkeypatch):
    """Mixed-store ordering (the 3026ecb production failure): sessions.db
    exists but this session was never migrated. Before the session_exists()
    gate, save_metadata() routed the draft to SQLite, the UPDATE matched zero
    rows, the follow-up lookup raised KeyError, and the draft was persisted
    nowhere. The route must return ok and write the JSON sidecar."""
    d = _tmp_session_dir()
    _patch_route_state(monkeypatch, d)

    sid = "sid-route-json-only"
    (d / f"{sid}.json").write_text(json.dumps(_sample_session_dict(sid)), encoding="utf-8")
    # Activate sessions.db WITHOUT migrating this session into it.
    _ = sqlite_db.WebUISqliteSessionDB(session_dir=d)

    captured = _drive_draft_post(
        monkeypatch, {"session_id": sid, "text": "json route draft", "files": []}
    )

    assert captured.get("status") == 200
    assert captured["payload"]["ok"] is True
    assert captured["payload"]["draft"]["text"] == "json route draft"

    on_disk = json.loads((d / f"{sid}.json").read_text(encoding="utf-8"))
    assert on_disk["composer_draft"]["text"] == "json route draft"
    assert on_disk["updated_at"] == 1001.0
    assert len(on_disk["messages"]) == 2

    # A fresh load (cold cache, e.g. after restart) reads the draft back.
    reloaded = models.Session.load(sid)
    assert reloaded.composer_draft["text"] == "json route draft"
    assert len(reloaded.messages) == 2
