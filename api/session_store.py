"""Canonical session-store abstraction.

Both session backends — the JSON sidecar store (``api.webui_session_db``)
and the SQLite store (``api.webui_session_sqlite``) — implement this
contract. ``api.models.get_session_store()`` returns the active backend;
callers route session persistence through it instead of branching on the
backend themselves.

Capabilities
------------
Backends declare capabilities instead of classes: production code branches
on ``supports_generation`` (durable per-session generation CAS + cutover
markers + unreadable-row recovery state), ``supports_revision_counter``
(meta-table revision for cache invalidation), and
``persists_without_sidecar`` (sessions survive with no JSON sidecar on
disk — SQLite/Postgres = True, JSON = False). The last is a topology axis,
distinct from CAS (``supports_generation``) and cache
(``supports_revision_counter``): it tells lifecycle call sites (existence
checks, index rebuilds, lineage scans) that the store is the authoritative
place to look when no sidecar exists. The JSON sidecar backend has none of
the three: its writes are atomic-replace (last writer wins by file
semantics) and its freshness signal is the directory mtime. The ``backend``
tag is informational (logging/tests) only — never a dispatch condition.

Deliberate boundary: ``Session.save()`` keeps its legacy inline JSON writer
for the sidecar backend (field-ordered metadata prefix, legacy-facts
caching, self-heal hooks). The JSON adapter's ``write_session`` is the
canonical CRUD surface used by routing/tests; they are not two competing
writers — the adapter writes whole sidecars, the legacy path writes
field-ordered ones for metadata-only readers.

Backend notes
-------------
- JSON sidecars: one ``<sid>.json`` per session. ``get_revision()`` is the
  session-directory mtime (sidecar create/delete bumps it); ``is_active()``
  is always True (JSON is the fallback authority).
- SQLite: single ``sessions.db``. ``get_revision()`` reads a meta-table
  counter bumped by every write/delete transaction; ``is_active()`` checks
  the durable cutover markers (``created_by=app`` or ``migration_complete``).

A future backend (e.g. Postgres) implements the same surface: ``extra_json``
maps to JSONB, ``get_revision()`` to a sequence, and the cutover markers to
rows in a meta table.
"""
from __future__ import annotations

from typing import Any, Protocol


class SessionStore(Protocol):
    """Canonical WebUI session persistence contract."""

    def list_sessions(self) -> list[dict[str, Any]]:
        """All session metadata rows, sidebar ordering."""
        ...

    def read_session(self, sid: str) -> dict[str, Any] | None:
        """Full session dict (metadata + messages/tool_calls/context), or None."""
        ...

    def read_metadata_only(self, sid: str) -> dict[str, Any] | None:
        """Metadata fields only — no transcript tables. None if absent."""
        ...

    def write_session(
        self,
        session: dict[str, Any],
        *,
        expected_generation: int | None = None,
        force: bool = False,
        fresh_incarnation: bool = False,
    ) -> dict[str, Any]:
        """Persist a full session dict; returns the persisted representation.

        SQL backends enforce a durable per-session generation CAS, backed by
        a per-SID incarnation authority (``session_incarnations``) that
        survives ``delete_session``:

        - Live row: ``expected_generation`` must match the row's generation
          (compare-and-bump); ``force=True`` may overwrite a live row for
          deliberate heals/imports. Anything else raises
          ``StaleSessionWriteError``.
        - Absent row, no authority, no lineage: first create at generation 1.
        - Absent row with a non-None ``expected_generation``: the writer
          carries lineage for a session that is gone —
          ``DeletedSessionWriteError``.
        - Absent row, retired authority, no lineage, no lease: the SID was
          deleted; recreating it requires an explicit lease —
          ``RetiredSessionWriteError``. ``force=True`` never inserts an
          absent row.
        - Absent row, retired authority, ``fresh_incarnation=True``: a
          leased recreate — the incarnation counter advances and the new
          row starts at generation 1. Only an explicit recreate API may
          pass this; ``Session.save()``, delete/migration, and heal paths
          never do. A leftover sidecar of the old incarnation is stale and
          is unlinked only by cleanup/delete, never by recreate.

        The returned dict carries the new ``generation``.
        JSON backends accept only ``expected_generation``/``force`` (both
        ignored — last-writer-wins atomic replace); ``fresh_incarnation``
        is SQL-only and must not be passed to a JSON store.
        """
        ...

    def update_metadata(self, sid: str, fields: dict[str, Any]) -> dict[str, Any]:
        """Persist a subset of metadata fields without touching the transcript."""
        ...

    def delete_session(self, sid: str) -> bool:
        """Remove the session and all its state. True if a row/file existed."""
        ...

    def lifecycle_delete(self, sid: str, *, owner: str) -> dict[str, Any]:
        """Delete the store-owned representation with durable retry ownership.

        Returns ``{"ok": True, "existed": bool}`` on success, or
        ``{"ok": False, "error": str, "existed": bool}`` on failure.

        On failure the store persists a retry lease keyed by ``sid`` (in
        ``cleanup_leases``) recording ``owner`` and the error, so a later
        sweep retains an authoritative retry owner instead of silently
        leaking the row. On success any lease for ``sid`` is cleared. The
        store never touches a representation it does not own (e.g. the
        SQL stores do not unlink JSON sidecars — callers do that only
        after an ``ok`` result).

        The JSON sidecar backend has no crash-safe lease (last-writer-wins
        file semantics); its failure result is process-local only.
        """
        ...

    def archive(self, sid: str, archived: bool = True) -> dict[str, Any]:
        ...

    def session_exists(self, sid: str) -> bool:
        ...

    def get_revision(self) -> int:
        """Monotonic change counter; every write/delete bumps it.

        List/index caches key their freshness on this instead of filesystem
        mtimes, which SQLite WAL commits do not move.
        """
        ...

    def is_active(self) -> bool:
        """Whether this backend currently holds persistence authority.

        The SQLite backend answers from durable cutover markers so a partial
        migration cannot silently activate it.
        """
        ...

    def close(self) -> None:
        ...
