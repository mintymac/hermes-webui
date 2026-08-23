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


def test_sqlite_round_trip_preserves_all_session_fields():
    """Gate finding: SQLite round-trips must be lossless."""
    d = _tmp_session_dir()
    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    payload = _sample_session_dict("sid-full")
    payload.update(
        {
            "created_workspace": "/ws/created",
            "intentional_shrink_generation": "gen-7",
            "user_id": "user-1",
            "chat_id": "chat-9",
            "session_key": "key-abc",
            "platform": "discord",
            "enabled_toolsets": ["fs", "shell"],
            "process_wakeup_pause": {"paused": True},
        }
    )
    store.write_session(payload)
    loaded = store.read_session("sid-full")
    for k, v in payload.items():
        assert loaded.get(k) == v, f"field {k!r} did not round-trip: {loaded.get(k)!r} != {v!r}"


def test_sqlite_round_trip_preserves_unknown_top_level_keys():
    """extra_json: fields this schema version does not know survive."""
    d = _tmp_session_dir()
    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    payload = _sample_session_dict("sid-unknown")
    payload["future_feature_flag"] = {"enabled": True, "tier": 3}
    payload["brand_new_scalar"] = "hello-future"
    store.write_session(payload)
    loaded = store.read_session("sid-unknown")
    assert loaded["future_feature_flag"] == {"enabled": True, "tier": 3}
    assert loaded["brand_new_scalar"] == "hello-future"
    # And via update_metadata merge.
    store.update_metadata("sid-unknown", {"another_future_key": [1, 2]})
    assert store.read_session("sid-unknown")["another_future_key"] == [1, 2]
    assert store.read_session("sid-unknown")["brand_new_scalar"] == "hello-future"


def test_session_level_unknown_keys_round_trip(monkeypatch):
    """Session(**data) with unknown keys -> save() -> store keeps them."""
    d = _tmp_session_dir()
    monkeypatch.setattr(models, "SESSION_DIR", d)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", d / "_index.json")
    monkeypatch.setattr(models, "_sqlite_session_store_instance", None)

    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    store.write_session(_sample_session_dict("sid-bag"))

    data = store.read_session("sid-bag")
    data["custom_messaging_field"] = {"channel": "tg", "thread": 42}
    s = models.Session(**data)
    s.save()

    loaded = store.read_session("sid-bag")
    assert loaded["custom_messaging_field"] == {"channel": "tg", "thread": 42}


def test_schema_v2_migration_adds_columns_to_legacy_db():
    """A db created by the pre-v2 schema gains the new columns on open."""
    import re as _re
    import sqlite3 as _sq

    d = _tmp_session_dir()
    db_path = d / "sessions.db"
    # Simulate a v1 database: the current schema minus the v2 columns.
    v2_cols = set(sqlite_db._SESSIONS_V2_COLUMNS)
    v1_lines = []
    for line in sqlite_db.SCHEMA_SQL.splitlines():
        stripped = line.strip()
        if stripped and stripped.split()[0].rstrip(",") in v2_cols:
            continue
        v1_lines.append(line)
    v1_schema = _re.sub(r",(\s*\);)", r"\1", "\n".join(v1_lines))
    conn = _sq.connect(str(db_path))
    conn.executescript(v1_schema)
    conn.execute(
        "INSERT INTO sessions (session_id, title, updated_at) VALUES ('s1', 'legacy', 1.0)"
    )
    conn.commit()
    conn.close()

    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    loaded = store.read_session("s1")
    assert loaded is not None
    assert loaded["title"] == "legacy"

    store.update_metadata("s1", {"user_id": "u1", "new_stuff": {"x": 1}})
    reloaded = store.read_session("s1")
    assert reloaded["user_id"] == "u1"
    assert reloaded["new_stuff"] == {"x": 1}

    conn = _sq.connect(str(db_path))
    (version,) = conn.execute("PRAGMA user_version").fetchone()
    conn.close()
    assert version == sqlite_db._SCHEMA_VERSION


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
    monkeypatch.setattr(models, "_PERSISTED_SESSION_IDS_CACHE", (None, None, None, frozenset()))
    monkeypatch.setattr(models, "_sqlite_session_store_instance", None)

    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    store.write_session(_sample_session_dict("sid-4"))

    ids = models._persisted_session_ids_snapshot()
    assert "sid-4" in ids


def test_generation_cas_refuses_stale_writers():
    """Gate finding: the fence is a durable per-session generation CAS.

    updated_at cannot fence real stale Session objects (save() stamps
    time.time() before writing); only the generation the object was loaded
    with can. Direct writes without lineage to an existing row are refused;
    matching lineage bumps; stale lineage is refused; force heals.
    """
    d = _tmp_session_dir()
    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)

    store.write_session(_sample_session_dict("sid-fence"))  # generation 1

    # No lineage (expected_generation=None) onto an existing row: refused.
    try:
        store.write_session(_sample_session_dict("sid-fence"))
    except sqlite_db.StaleSessionWriteError:
        pass
    else:
        raise AssertionError("lineage-less overwrite must be refused")

    # Matching lineage: compare-and-bump.
    store.write_session(_sample_session_dict("sid-fence"), expected_generation=1)
    assert store.read_session("sid-fence")["generation"] == 2

    # Stale lineage: refused.
    try:
        store.write_session(_sample_session_dict("sid-fence"), expected_generation=1)
    except sqlite_db.StaleSessionWriteError:
        pass
    else:
        raise AssertionError("stale generation must be refused")

    # force bypasses for deliberate heals and bumps anyway.
    store.write_session(_sample_session_dict("sid-fence"), force=True)
    assert store.read_session("sid-fence")["generation"] == 3


def test_two_readers_stale_writer_is_rejected(monkeypatch):
    """Gate finding: load A and B, save A, then stale B must be rejected."""
    d = _tmp_session_dir()
    monkeypatch.setattr(models, "SESSION_DIR", d)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", d / "_index.json")
    monkeypatch.setattr(models, "_sqlite_session_store_instance", None)

    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    store.write_session(_sample_session_dict("sid-2readers"))
    monkeypatch.setattr(models, "_sqlite_session_store_instance", store)

    a = models.Session.load("sid-2readers")
    b = models.Session.load("sid-2readers")
    assert a is not None and b is not None
    assert a._persisted_generation == 1
    assert b._persisted_generation == 1

    a.title = "saved by A"
    a.save()
    assert a._persisted_generation == 2
    assert store.read_session("sid-2readers")["title"] == "saved by A"

    b.title = "stale B write"
    try:
        b.save()
    except sqlite_db.StaleSessionWriteError:
        pass
    else:
        raise AssertionError("stale reader B must be rejected")

    assert store.read_session("sid-2readers")["title"] == "saved by A"

    # A remains writeable after its successful save (generation updated).
    a.title = "saved again by A"
    a.save()
    assert store.read_session("sid-2readers")["title"] == "saved again by A"


def test_null_valued_keys_round_trip_with_presence(monkeypatch):
    """Key-presence contract: an explicitly-None field reads back as a
    present None, matching the JSON backend's native behavior."""
    d = _tmp_session_dir()
    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)

    payload = _sample_session_dict("sid-nulls")
    payload["personality"] = None
    payload["project_id"] = None
    payload["user_id"] = None
    store.write_session(payload)

    loaded = store.read_session("sid-nulls")
    assert "personality" in loaded and loaded["personality"] is None
    assert "project_id" in loaded and loaded["project_id"] is None
    assert "user_id" in loaded and loaded["user_id"] is None

    # update_metadata maintains presence in both directions.
    store.update_metadata("sid-nulls", {"personality": "now-set", "threshold_tokens": None})
    reloaded = store.read_session("sid-nulls")
    assert reloaded["personality"] == "now-set"
    assert "threshold_tokens" in reloaded and reloaded["threshold_tokens"] is None


def test_save_heal_bypasses_fence_for_marked_session(monkeypatch):
    """A marked (unreadable) session may have a row newer than its sidecar;
    the heal save must bypass the fence."""
    import sqlite3 as _sq

    d = _tmp_session_dir()
    monkeypatch.setattr(models, "SESSION_DIR", d)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", d / "_index.json")

    sid = "sid-healfence"
    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    payload = _sample_session_dict(sid)
    payload["updated_at"] = 9000.0  # row newer than the sidecar below
    store.write_session(payload)
    sidecar = _sample_session_dict(sid)
    sidecar["updated_at"] = 1000.0
    (d / f"{sid}.json").write_text(json.dumps(sidecar), encoding="utf-8")
    monkeypatch.setattr(models, "_sqlite_session_store_instance", store)

    conn = _sq.connect(str(d / "sessions.db"))
    conn.execute("UPDATE messages SET message_json = 'bad' WHERE session_id = ?", (sid,))
    conn.commit()
    conn.close()

    s = models.Session.load(sid)
    assert s is not None
    assert sid in store.unreadable_sids

    s.save()  # must not raise StaleSessionWriteError despite the older sidecar state
    assert sid not in store.unreadable_sids
    assert models.Session.load(sid) is not None


def test_incomplete_migration_does_not_activate_store(monkeypatch):
    """Gate finding: sessions.db existence alone must not grant authority."""
    d = _tmp_session_dir()
    monkeypatch.setattr(models, "SESSION_DIR", d)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", d / "_index.json")
    monkeypatch.setattr(models, "_sqlite_session_store_instance", None)

    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    store.set_meta("created_by", "migration")
    store.set_meta("migration_complete", "0")
    store.write_session(_sample_session_dict("sid-migrated-partial"))
    store.close()

    # Interrupted migration: the store must NOT take authority.
    assert models._get_sqlite_session_store() is False

    # Stamping completion activates it.
    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    store.set_meta("migration_complete", "1")
    store.close()
    monkeypatch.setattr(models, "_sqlite_session_store_instance", None)
    assert models._get_sqlite_session_store()


def test_app_created_store_is_active_by_default(monkeypatch):
    d = _tmp_session_dir()
    monkeypatch.setattr(models, "SESSION_DIR", d)
    monkeypatch.setattr(models, "_sqlite_session_store_instance", None)
    sqlite_db.WebUISqliteSessionDB(session_dir=d)  # creates sessions.db (created_by=app)
    monkeypatch.setattr(models, "_sqlite_session_store_instance", None)
    assert models._get_sqlite_session_store()


def test_persisted_session_ids_snapshot_tracks_sqlite_revision(monkeypatch):
    """Gate finding: WAL commits do not move the directory mtime, so the
    persisted-ids cache must key on the store revision instead."""
    d = _tmp_session_dir()
    monkeypatch.setattr(models, "SESSION_DIR", d)
    monkeypatch.setattr(models, "_PERSISTED_SESSION_IDS_CACHE", (None, None, None, frozenset()))
    monkeypatch.setattr(models, "_sqlite_session_store_instance", None)

    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    store.write_session(_sample_session_dict("sid-r1"))
    first = models._persisted_session_ids_snapshot()
    assert "sid-r1" in first

    # Second write: the directory mtime does not move for a WAL commit to an
    # existing sessions.db, but the revision bump must refresh the snapshot.
    store.write_session(_sample_session_dict("sid-r2"))
    second = models._persisted_session_ids_snapshot()
    assert "sid-r2" in second
    assert second is not first

    # Deletes bump the revision too.
    store.delete_session("sid-r2")
    third = models._persisted_session_ids_snapshot()
    assert "sid-r2" not in third
    assert "sid-r1" in third


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


def test_save_metadata_routes_to_sidecar_after_sqlite_read_error(monkeypatch):
    """Greptile P1: an unreadable row must not black-hole draft autosaves.

    Session.load() falls back to the sidecar on a corrupt row; without the
    unreadable marker, save_metadata() kept routing drafts to SQLite (the
    row exists) where the next load can never read them — the autosave
    reported success while the draft stayed invisible.
    """
    import sqlite3 as _sq

    d = _tmp_session_dir()
    monkeypatch.setattr(models, "SESSION_DIR", d)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", d / "_index.json")
    monkeypatch.setattr(models, "_sqlite_session_store_instance", None)

    sid = "sid-corrupt-draft"
    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    store.write_session(_sample_session_dict(sid))
    (d / f"{sid}.json").write_text(
        json.dumps(_sample_session_dict(sid)), encoding="utf-8"
    )

    conn = _sq.connect(str(d / "sessions.db"))
    conn.execute(
        "UPDATE messages SET message_json = 'not-json' WHERE session_id = ?",
        (sid,),
    )
    conn.commit()
    conn.close()

    # Load detects the corrupt row and falls back to the sidecar.
    s = models.Session.load(sid)
    assert s is not None

    # The draft autosave must go to the sidecar — the store the next load reads.
    s.save_metadata({"composer_draft": {"text": "draft after corruption", "files": []}})

    reloaded = models.Session.load(sid)
    assert reloaded is not None
    assert reloaded.composer_draft["text"] == "draft after corruption"


def test_transient_read_error_without_sidecar_does_not_poison_routing(monkeypatch):
    """Greptile P1: a transient read error must not stick a migrated
    (sidecar-less) session to the JSON write path.

    The unreadable mark means "the sidecar just saved us" — with no sidecar
    there is nothing to route to, and a stuck mark would make save_metadata()
    read a missing sidecar and 500 every autosave after the DB recovers.
    """
    import sqlite3 as _sq

    d = _tmp_session_dir()
    monkeypatch.setattr(models, "SESSION_DIR", d)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", d / "_index.json")

    sid = "sid-transient"
    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    store.write_session(_sample_session_dict(sid))
    monkeypatch.setattr(models, "_sqlite_session_store_instance", store)

    real_read = store.read_session
    calls = {"n": 0}

    def fail_once(sid_):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _sq.OperationalError("database is locked")
        return real_read(sid_)

    monkeypatch.setattr(store, "read_session", fail_once)

    # Transient failure, no sidecar: the session is unavailable this request,
    # but must NOT be marked unreadable.
    assert models.Session.load(sid) is None
    assert sid not in store.unreadable_sids

    # DB recovered: loads and draft autosaves use SQLite normally.
    s = models.Session.load(sid)
    assert s is not None
    s.save_metadata({"composer_draft": {"text": "ok after recovery", "files": []}})
    assert store.read_metadata_only(sid)["composer_draft"]["text"] == "ok after recovery"


def test_recovery_demotes_mark_and_carries_draft(monkeypatch):
    """Greptile P1 lifecycle: after the row recovers, the next load demotes
    the mark and carries marked-window drafts into the row; stale
    sidecar-loaded writers are refused by the generation CAS."""
    import sqlite3 as _sq

    d = _tmp_session_dir()
    monkeypatch.setattr(models, "SESSION_DIR", d)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", d / "_index.json")

    sid = "sid-recover2"
    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    store.write_session(_sample_session_dict(sid))
    (d / f"{sid}.json").write_text(
        json.dumps(_sample_session_dict(sid)), encoding="utf-8"
    )
    monkeypatch.setattr(models, "_sqlite_session_store_instance", store)

    real_read = store.read_session
    calls = {"n": 0}

    def fail_once(sid_):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _sq.OperationalError("database is locked")
        return real_read(sid_)

    monkeypatch.setattr(store, "read_session", fail_once)

    # Transient failure with a sidecar: fallback load succeeds, sid marked.
    s = models.Session.load(sid)
    assert s is not None
    assert sid in store.unreadable_sids

    # While marked, the draft autosave routes to the sidecar.
    s.save_metadata({"composer_draft": {"text": "typed during outage", "files": []}})

    # Recovery: the next load demotes the mark and carries the draft.
    s2 = models.Session.load(sid)
    assert s2 is not None
    assert sid not in store.unreadable_sids
    assert s2.composer_draft["text"] == "typed during outage"
    assert store.read_metadata_only(sid)["composer_draft"]["text"] == "typed during outage"
    assert s2._persisted_generation == 1

    # The stale sidecar-loaded object is now a stale reader: refused loudly
    # instead of rolling the recovered row back.
    try:
        s.title = "stale write"
        s.save()
    except sqlite_db.StaleSessionWriteError:
        pass
    else:
        raise AssertionError("stale sidecar-loaded writer must be refused")

    assert store.read_session(sid)["title"] == "Test Session"


def test_recovered_row_keeps_its_own_draft_on_demote(monkeypatch):
    """Demote-on-recovery never reconciles an unchanged sidecar draft, so
    the row's own newer draft survives."""
    import sqlite3 as _sq

    d = _tmp_session_dir()
    monkeypatch.setattr(models, "SESSION_DIR", d)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", d / "_index.json")

    sid = "sid-noclobber"
    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    store.write_session(_sample_session_dict(sid))
    store.update_metadata(sid, {"composer_draft": {"text": "newer in sqlite", "files": []}})
    sidecar_payload = _sample_session_dict(sid)
    sidecar_payload["composer_draft"] = {"text": "older in sidecar", "files": []}
    (d / f"{sid}.json").write_text(json.dumps(sidecar_payload), encoding="utf-8")
    monkeypatch.setattr(models, "_sqlite_session_store_instance", store)

    real_read = store.read_session
    calls = {"n": 0}

    def fail_once(sid_):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _sq.OperationalError("database is locked")
        return real_read(sid_)

    monkeypatch.setattr(store, "read_session", fail_once)

    assert models.Session.load(sid) is not None
    assert sid in store.unreadable_sids

    # DB recovered: the next load demotes the mark. The sidecar draft never
    # moved while marked, so nothing is carried — the row keeps its draft.
    assert models.Session.load(sid) is not None
    assert sid not in store.unreadable_sids
    assert store.read_metadata_only(sid)["composer_draft"]["text"] == "newer in sqlite"


def test_marked_sid_without_sidecar_falls_back_to_healthy_row(monkeypatch):
    """If the sidecar disappears while marked and the row reads healthy,
    the mark clears — there is nothing left to protect."""
    import sqlite3 as _sq

    d = _tmp_session_dir()
    monkeypatch.setattr(models, "SESSION_DIR", d)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", d / "_index.json")

    sid = "sid-gone-sidecar"
    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    store.write_session(_sample_session_dict(sid))
    (d / f"{sid}.json").write_text(
        json.dumps(_sample_session_dict(sid)), encoding="utf-8"
    )
    monkeypatch.setattr(models, "_sqlite_session_store_instance", store)

    real_read = store.read_session
    calls = {"n": 0}

    def fail_once(sid_):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _sq.OperationalError("database is locked")
        return real_read(sid_)

    monkeypatch.setattr(store, "read_session", fail_once)

    assert models.Session.load(sid) is not None
    assert sid in store.unreadable_sids

    (d / f"{sid}.json").unlink()
    s = models.Session.load(sid)
    assert s is not None
    assert sid not in store.unreadable_sids


def test_full_save_heals_corrupt_sqlite_row(monkeypatch):
    """A successful full save() rewrites the row with healthy data and
    clears the unreadable mark, so draft routing returns to SQLite."""
    import sqlite3 as _sq

    d = _tmp_session_dir()
    monkeypatch.setattr(models, "SESSION_DIR", d)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", d / "_index.json")
    monkeypatch.setattr(models, "_sqlite_session_store_instance", None)

    sid = "sid-heal"
    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    monkeypatch.setattr(models, "_sqlite_session_store_instance", store)
    store.write_session(_sample_session_dict(sid))
    (d / f"{sid}.json").write_text(
        json.dumps(_sample_session_dict(sid)), encoding="utf-8"
    )

    conn = _sq.connect(str(d / "sessions.db"))
    conn.execute(
        "UPDATE messages SET message_json = 'not-json' WHERE session_id = ?",
        (sid,),
    )
    conn.commit()
    conn.close()

    s = models.Session.load(sid)
    assert sid in store.unreadable_sids

    s.save()
    assert sid not in store.unreadable_sids

    # The row is healthy again: loads come from SQLite, drafts route there.
    (d / f"{sid}.json").unlink()
    reloaded = models.Session.load(sid)
    assert reloaded is not None
    assert len(reloaded.messages) == 2


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


def test_save_metadata_sqlite_write_failure_leaves_in_memory_untouched(monkeypatch):
    """A failed SQLite metadata write must not poison the cached Session."""
    import sqlite3 as _sq

    d = _tmp_session_dir()
    monkeypatch.setattr(models, "SESSION_DIR", d)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", d / "_index.json")

    sid = "sid-writefail"
    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    store.write_session(_sample_session_dict(sid))
    monkeypatch.setattr(models, "_sqlite_session_store_instance", store)

    s = models.Session.load(sid)
    old_draft = dict(getattr(s, "composer_draft", {}) or {})

    def _boom(sid_, fields):
        raise _sq.OperationalError("database is locked")

    monkeypatch.setattr(store, "update_metadata", _boom)

    try:
        s.save_metadata({"composer_draft": {"text": "lost?", "files": []}})
    except _sq.OperationalError:
        pass
    else:
        raise AssertionError("write failure must propagate")

    assert (getattr(s, "composer_draft", {}) or {}) == old_draft


def test_save_metadata_json_write_failure_leaves_in_memory_untouched(monkeypatch):
    """A failed sidecar write must not poison the cached Session either."""
    d = _tmp_session_dir()
    monkeypatch.setattr(models, "SESSION_DIR", d)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", d / "_index.json")
    monkeypatch.setattr(models, "_sqlite_session_store_instance", None)

    sid = "sid-jsonfail"
    sidecar = d / f"{sid}.json"
    sidecar.write_text(json.dumps(_sample_session_dict(sid)), encoding="utf-8")

    s = models.Session.load(sid)
    old_draft = dict(getattr(s, "composer_draft", {}) or {})

    monkeypatch.setattr(
        models,
        "_safe_replace",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
    )

    try:
        s.save_metadata({"composer_draft": {"text": "lost?", "files": []}})
    except OSError:
        pass
    else:
        raise AssertionError("write failure must propagate")

    assert (getattr(s, "composer_draft", {}) or {}) == old_draft
    assert "composer_draft" not in json.loads(sidecar.read_text(encoding="utf-8"))


def test_draft_route_retry_after_sqlite_write_failure_persists(monkeypatch):
    """Greptile P1: a failed write must not poison the draft cache.

    save_metadata() applied the in-memory update before the SQLite write,
    so a failed write left s.composer_draft ahead of disk; the route's
    unchanged fast path then skipped the retry and the draft vanished on
    reload. The route drives POST /api/session/draft through the real
    dispatcher, fail-once the write, and the identical retry must persist.
    """
    import sqlite3 as _sq

    d = _tmp_session_dir()
    _patch_route_state(monkeypatch, d)

    sid = "sid-poison"
    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    store.write_session(_sample_session_dict(sid))
    monkeypatch.setattr(models, "_sqlite_session_store_instance", store)

    real_update = store.update_metadata
    calls = {"n": 0}

    def fail_once(sid_, fields):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _sq.OperationalError("database is locked")
        return real_update(sid_, fields)

    monkeypatch.setattr(store, "update_metadata", fail_once)

    # First attempt: the write error propagates (production turns it into a 500).
    try:
        _drive_draft_post(monkeypatch, {"session_id": sid, "text": "retry me"})
    except Exception:
        pass
    else:
        raise AssertionError("the failed write must surface")

    # The cached draft must NOT have advanced — the unchanged fast path on
    # the retry compares against it.
    s = models.get_session(sid)
    assert (getattr(s, "composer_draft", {}) or {}).get("text", "") != "retry me"

    # Identical retry: must persist this time (not hit the unchanged path).
    captured = _drive_draft_post(monkeypatch, {"session_id": sid, "text": "retry me"})
    assert captured.get("payload", {}).get("ok") is True
    assert "unchanged" not in captured.get("payload", {})
    assert store.read_metadata_only(sid)["composer_draft"]["text"] == "retry me"


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


def test_delete_route_fails_closed_when_sqlite_delete_fails(monkeypatch):
    """Gate finding: deletion must fail closed at the store boundary.

    When the SQLite delete raises, the route must NOT unlink the sidecar,
    prune the index, or report success — the transcript stays visible and
    the delete stays retryable.
    """
    import sqlite3 as _sq

    d = _tmp_session_dir()
    _patch_route_state(monkeypatch, d)

    sid = "sid-failclosed"
    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    store.write_session(_sample_session_dict(sid))
    monkeypatch.setattr(models, "_sqlite_session_store_instance", store)
    monkeypatch.setattr(models, "delete_cli_session", lambda sid_: True)

    real_delete = store.delete_session

    def _boom(sid_):
        raise _sq.OperationalError("database is locked")

    monkeypatch.setattr(store, "delete_session", _boom)

    captured = _drive_delete_post(monkeypatch, {"session_id": sid})
    assert captured.get("status") == 500
    assert "error" in captured
    # Nothing was torn down: the row survives and the session stays listed.
    assert store.session_exists(sid) is True

    # Recovery: the delete succeeds once the store works again.
    monkeypatch.setattr(store, "delete_session", real_delete)
    captured = _drive_delete_post(monkeypatch, {"session_id": sid})
    assert captured.get("payload", {}).get("ok") is True
    assert store.session_exists(sid) is False


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


def test_preserve_pre_compression_snapshot_marks_sqlite_only_session(tmp_path, monkeypatch):
    """Gate finding: compression snapshot preservation must not return early
    just because the old session has no JSON sidecar."""
    import api.streaming as streaming

    monkeypatch.setattr(streaming, "SESSION_DIR", tmp_path)
    monkeypatch.setattr(models, "SESSION_DIR", tmp_path)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", tmp_path / "_index.json")

    store = sqlite_db.WebUISqliteSessionDB(session_dir=tmp_path)
    old = _sample_session_dict("old_sqlite")
    old["messages"] = [{"role": "user", "content": f"m{i}"} for i in range(5)]
    store.write_session(old)
    monkeypatch.setattr(models, "_sqlite_session_store_instance", store)

    # Continuation session holds fewer messages than the stored old session,
    # so the load-and-mark branch runs and must stamp the marker in the store.
    s = models.Session(session_id="new_cont")
    s.messages = [{"role": "user", "content": "x"}]
    s.parent_session_id = "original_parent"

    streaming._preserve_pre_compression_snapshot(s, "old_sqlite")

    meta = store.read_metadata_only("old_sqlite")
    assert meta is not None
    assert bool(meta["pre_compression_snapshot"]) is True
    # The fuller stored transcript is untouched.
    assert len(store.read_session("old_sqlite")["messages"]) == 5


def test_preserve_pre_compression_snapshot_rewrites_sqlite_snapshot_from_memory(
    tmp_path, monkeypatch
):
    """Rewrite branch: in-memory messages are newer than the stored row."""
    import api.streaming as streaming

    monkeypatch.setattr(streaming, "SESSION_DIR", tmp_path)
    monkeypatch.setattr(models, "SESSION_DIR", tmp_path)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", tmp_path / "_index.json")

    store = sqlite_db.WebUISqliteSessionDB(session_dir=tmp_path)
    old = _sample_session_dict("old_sqlite2")
    old["messages"] = [{"role": "user", "content": "m0"}]
    store.write_session(old)
    monkeypatch.setattr(models, "_sqlite_session_store_instance", store)

    s = models.Session(session_id="new_cont2")
    s.messages = [{"role": "user", "content": f"m{i}"} for i in range(5)]
    s.parent_session_id = "original_parent"

    streaming._preserve_pre_compression_snapshot(s, "old_sqlite2")

    full = store.read_session("old_sqlite2")
    assert full is not None
    assert len(full["messages"]) == 5
    assert full["pre_compression_snapshot"] is True
    # Continuation state restored.
    assert s.session_id == "new_cont2"
    assert s.pre_compression_snapshot is False


def _drive_clear_post(monkeypatch, body):
    """Run POST /api/session/clear through routes.handle_post."""
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
    assert routes.handle_post(handler, SimpleNamespace(path="/api/session/clear")) is True
    return captured


def test_clear_route_verifies_persisted_state_via_store(monkeypatch):
    """Clear verification reads the active store, not a missing sidecar."""
    d = _tmp_session_dir()
    _patch_route_state(monkeypatch, d)

    sid = "sid-clear"
    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    store.write_session(_sample_session_dict(sid))
    monkeypatch.setattr(models, "_sqlite_session_store_instance", store)

    captured = _drive_clear_post(monkeypatch, {"session_id": sid})
    assert captured.get("payload", {}).get("ok") is True, captured

    persisted = store.read_session(sid)
    assert persisted is not None
    assert persisted["messages"] == []
    assert persisted.get("active_stream_id") is None
    assert persisted.get("pending_user_message") is None


def test_metadata_only_demote_carries_marked_draft(monkeypatch):
    """Greptile P1: the metadata-only recovery path must carry sidecar-saved
    drafts too, or demoting strands them."""
    import sqlite3 as _sq

    d = _tmp_session_dir()
    monkeypatch.setattr(models, "SESSION_DIR", d)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", d / "_index.json")

    sid = "sid-meta-carry"
    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    store.write_session(_sample_session_dict(sid))
    (d / f"{sid}.json").write_text(
        json.dumps(_sample_session_dict(sid)), encoding="utf-8"
    )
    monkeypatch.setattr(models, "_sqlite_session_store_instance", store)

    real_read = store.read_session
    calls = {"n": 0}

    def fail_once(sid_):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _sq.OperationalError("database is locked")
        return real_read(sid_)

    monkeypatch.setattr(store, "read_session", fail_once)

    s = models.Session.load(sid)  # transient fail -> sidecar fallback, marked
    assert s is not None
    assert sid in store.unreadable_sids

    s.save_metadata({"composer_draft": {"text": "meta-carried", "files": []}})

    # Row recovered: the metadata-only read demotes and carries the draft.
    meta = models.Session.load_metadata_only(sid)
    assert meta is not None
    assert sid not in store.unreadable_sids
    assert store.read_metadata_only(sid)["composer_draft"]["text"] == "meta-carried"


def test_ephemeral_cancel_cleanup_deletes_store_row(monkeypatch):
    """DB-only ephemeral cancel cleanup must delete the row (gate finding)."""
    import api.streaming as streaming

    d = _tmp_session_dir()
    _patch_route_state(monkeypatch, d)
    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    store.write_session(_sample_session_dict("sid-ephem"))
    monkeypatch.setattr(models, "_sqlite_session_store_instance", store)
    monkeypatch.setattr(streaming, "SESSION_DIR", d)

    class FakeEphemeralSession:
        session_id = "sid-ephem"
        path = d / "sid-ephem.json"
        active_stream_id = "live"
        pending_user_message = "x"
        pending_attachments = []
        pending_started_at = 1.0
        pending_user_source = None

    streaming._cleanup_ephemeral_cancelled_turn(FakeEphemeralSession())
    assert store.session_exists("sid-ephem") is False


def test_ephemeral_cancel_cleanup_retains_row_on_store_failure(monkeypatch):
    """Failed authoritative deletion retains the record for retry."""
    import sqlite3 as _sq

    import api.streaming as streaming

    d = _tmp_session_dir()
    _patch_route_state(monkeypatch, d)
    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    store.write_session(_sample_session_dict("sid-ephem2"))
    monkeypatch.setattr(models, "_sqlite_session_store_instance", store)
    monkeypatch.setattr(streaming, "SESSION_DIR", d)

    def _boom(sid_):
        raise _sq.OperationalError("database is locked")

    monkeypatch.setattr(store, "delete_session", _boom)

    class FakeEphemeralSession:
        session_id = "sid-ephem2"
        path = d / "sid-ephem2.json"
        active_stream_id = "live"
        pending_user_message = "x"
        pending_attachments = []
        pending_started_at = 1.0
        pending_user_source = None

    streaming._cleanup_ephemeral_cancelled_turn(FakeEphemeralSession())
    assert store.session_exists("sid-ephem2") is True


def test_recovered_workspace_binding_patches_store_row(monkeypatch):
    """DB-only workspace recovery must patch the row, not demand a sidecar."""
    d = _tmp_session_dir()
    monkeypatch.setattr(models, "SESSION_DIR", d)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", d / "_index.json")
    monkeypatch.setattr(models, "_sqlite_session_store_instance", None)

    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    store.write_session(_sample_session_dict("sid-ws"))
    monkeypatch.setattr(models, "_sqlite_session_store_instance", store)

    s = models.Session.load("sid-ws")
    assert s is not None
    models.persist_recovered_workspace_binding(
        s, "/recovered/ws", expected_workspace=s.workspace
    )
    assert store.read_metadata_only("sid-ws")["workspace"] == "/recovered/ws"

    # A mismatched expected_workspace fails closed, like the sidecar path.
    try:
        models.persist_recovered_workspace_binding(
            s, "/another/ws", expected_workspace="/different"
        )
    except models.WorkspaceBindingPersistenceError:
        pass
    else:
        raise AssertionError("workspace-changed mismatch must fail closed")


def test_sessions_cleanup_ghost_sweep_keeps_sqlite_rows(monkeypatch):
    """DB-only index rows are live sessions, not ghosts (gate finding)."""
    import api.routes as routes

    d = _tmp_session_dir()
    _patch_route_state(monkeypatch, d)
    monkeypatch.setattr(routes, "SESSION_INDEX_FILE", d / "_index.json")
    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    payload = _sample_session_dict("sid-live-db")
    payload["title"] = "Real DB Session"
    store.write_session(payload)
    monkeypatch.setattr(models, "_sqlite_session_store_instance", store)

    index_file = d / "_index.json"
    index_file.write_text(
        json.dumps(
            [
                {"session_id": "sid-live-db", "title": "Real DB Session"},
                {"session_id": "sid-ghost", "title": "Ghost"},
            ]
        ),
        encoding="utf-8",
    )

    captured = {}
    monkeypatch.setattr(
        routes, "j", lambda handler, payload, **kwargs: captured.update(payload) or True
    )
    handler = SimpleNamespace(_safe_webui_print=lambda *_a, **_k: None)
    routes._handle_sessions_cleanup(handler, {}, zero_only=False)

    survivors = json.loads(index_file.read_text(encoding="utf-8"))
    survivor_ids = [e["session_id"] for e in survivors]
    assert "sid-live-db" in survivor_ids  # DB-only row is live
    assert "sid-ghost" not in survivor_ids  # real ghost removed
    assert store.session_exists("sid-live-db") is True


def test_sessions_cleanup_removes_db_only_zero_message_untitled(monkeypatch):
    """Phase 1b: DB-only zero-message Untitled sessions are cleaned."""
    import api.routes as routes

    d = _tmp_session_dir()
    _patch_route_state(monkeypatch, d)
    store = sqlite_db.WebUISqliteSessionDB(session_dir=d)
    payload = _sample_session_dict("sid-zero")
    payload["title"] = "Untitled"
    payload["messages"] = []
    store.write_session(payload)
    monkeypatch.setattr(models, "_sqlite_session_store_instance", store)

    captured = {}
    monkeypatch.setattr(
        routes, "j", lambda handler, payload, **kwargs: captured.update(payload) or True
    )
    handler = SimpleNamespace(_safe_webui_print=lambda *_a, **_k: None)
    routes._handle_sessions_cleanup(handler, {}, zero_only=False)

    assert store.session_exists("sid-zero") is False


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
