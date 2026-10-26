"""claims-tracker MCP server.

Stores cases, notes, appeal deadlines and the human's approval decision in SQLite,
and writes an audit row for every call.

Narrow by design:

* There is no tool to delete anything, send anything or pay anything.
* Every case belongs to an ``owner_id``. A call that names someone else's case gets
  ``not_found`` (the same answer as for a case that does not exist, so ids cannot be probed)
  and the attempt is recorded in the audit log as ``denied``.
* ``record_decision`` accepts only ``approved`` or ``rejected`` and can be applied once.
* Case status can only reach ``appeal_approved`` / ``appeal_rejected`` through ``record_decision``.
* The audit log stores a SHA-256 of each call's arguments, never the arguments themselves,
  so letters and notes are not copied into it.

Configuration: ``INSURITHM_TRACKER_DB`` (path to the SQLite file; default ``./insurithm_tracker.sqlite``).

Run it on its own:  python -m insurithm.mcp_servers.claims_tracker
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
import threading
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Callable, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

OWNER_ID = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.@+\-]{1,64}$", description="Who owns the case")]
CASE_ID = Annotated[str, Field(pattern=r"^[A-Za-z0-9_\-]{3,64}$")]
POLICY_ID = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.\-]{1,64}$")]
SHORT_TEXT = Annotated[str, Field(min_length=1, max_length=64)]
NOTE_TEXT = Annotated[str, Field(min_length=1, max_length=2000)]
LETTER_TEXT = Annotated[str, Field(min_length=20, max_length=20000)]
DUE_DATE = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="ISO date, YYYY-MM-DD")]
COMMENT = Annotated[str, Field(max_length=1000)]
QUESTION = Annotated[str, Field(max_length=1000, description="What the policyholder wants to know (optional)")]

OPEN_STATUSES = ("open", "awaiting_review", "needs_more_evidence")
FINAL_STATUSES = {"approved": "appeal_approved", "rejected": "appeal_rejected"}

_OWNER_RE = re.compile(r"^[A-Za-z0-9_.@+\-]{1,64}$")
_CASE_RE = re.compile(r"^[A-Za-z0-9_\-]{3,64}$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, policy_id TEXT NOT NULL,
    letter_text TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
    question TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL REFERENCES cases(case_id),
    author TEXT NOT NULL, text TEXT NOT NULL, dedupe_key TEXT, created_at TEXT NOT NULL,
    UNIQUE (case_id, dedupe_key));
CREATE TABLE IF NOT EXISTS deadlines (
    id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL REFERENCES cases(case_id),
    label TEXT NOT NULL, due_date TEXT NOT NULL, created_at TEXT NOT NULL,
    UNIQUE (case_id, label));
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL UNIQUE REFERENCES cases(case_id),
    decision TEXT NOT NULL CHECK (decision IN ('approved', 'rejected')),
    reviewer TEXT NOT NULL, comment TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, tool TEXT NOT NULL,
    owner_id TEXT, case_id TEXT, args_sha256 TEXT NOT NULL, status TEXT NOT NULL, error_code TEXT);
"""


class ToolFault(Exception):
    """An expected, reportable failure (bad input, missing case, access refused)."""

    def __init__(self, code: str, message: str, denied: bool = False):
        super().__init__(message)
        self.code, self.message, self.denied = code, message, denied


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash_args(args: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(args, sort_keys=True, default=str).encode()).hexdigest()


class TrackerStore:
    """All tracker logic. Plain Python so it can be tested without MCP."""

    def __init__(self, db_path: str | Path):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(SCHEMA)
        # A database made before the question field existed gets the column added.
        if "question" not in {r["name"] for r in self._db.execute("PRAGMA table_info(cases)")}:
            self._db.execute("ALTER TABLE cases ADD COLUMN question TEXT NOT NULL DEFAULT ''")
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    # -- plumbing ---------------------------------------------------------
    def _call(
        self, tool: str, owner_id: Any, case_id: Any, args: dict[str, Any], fn: Callable[[], dict[str, Any]]
    ) -> dict[str, Any]:
        status, code = "ok", None
        try:
            with self._lock:
                self._validate_ids(owner_id, case_id)
                result = {"ok": True, **fn()}
                self._db.commit()
        except ToolFault as fault:
            with self._lock:
                self._db.rollback()
            status, code = ("denied" if fault.denied else "error"), fault.code
            result = {"ok": False, "error": {"code": fault.code, "message": fault.message}}
        except Exception as exc:  # never let a bug leak a traceback to the model
            with self._lock:
                self._db.rollback()
            print(f"claims-tracker internal error in {tool}: {exc!r}", file=sys.stderr)
            status, code = "error", "internal_error"
            result = {"ok": False, "error": {"code": code, "message": "Unexpected server error."}}
        self._audit(tool, owner_id, case_id, args, status, code)
        return result

    @staticmethod
    def _validate_ids(owner_id: Any, case_id: Any) -> None:
        if not isinstance(owner_id, str) or not _OWNER_RE.match(owner_id):
            raise ToolFault("invalid_owner", "owner_id must be 1-64 characters: letters, digits, _ . @ + -")
        if case_id is not None and (not isinstance(case_id, str) or not _CASE_RE.match(case_id)):
            raise ToolFault("invalid_case_id", "case_id must be 3-64 characters: letters, digits, _ -")

    def _audit(self, tool: str, owner_id: Any, case_id: Any, args: dict[str, Any], status: str, code: str | None) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO audit_log (ts, tool, owner_id, case_id, args_sha256, status, error_code) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    _now(),
                    tool,
                    owner_id if isinstance(owner_id, str) else None,
                    case_id if isinstance(case_id, str) else None,
                    _hash_args(args),
                    status,
                    code,
                ),
            )
            self._db.commit()

    def _owned_case(self, owner_id: str, case_id: str) -> sqlite3.Row:
        row = self._db.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        if row is None:
            raise ToolFault("not_found", f"No case {case_id!r}.")
        if row["owner_id"] != owner_id:
            raise ToolFault("not_found", f"No case {case_id!r}.", denied=True)
        return row

    # -- tools --------------------------------------------------------------
    def create_case(
        self, owner_id: str, policy_id: str, letter_text: str, case_id: str | None = None, question: str = ""
    ) -> dict[str, Any]:
        args = {"owner_id": owner_id, "policy_id": policy_id, "letter_text": letter_text, "case_id": case_id, "question": question}

        def run() -> dict[str, Any]:
            if not isinstance(letter_text, str) or len(letter_text.strip()) < 20:
                raise ToolFault("invalid_letter", "letter_text must be at least 20 characters.")
            if not isinstance(question, str) or len(question) > 1000:
                raise ToolFault("invalid_question", "question must be at most 1000 characters.")
            new_id = case_id or uuid.uuid4().hex[:12]
            if self._db.execute("SELECT 1 FROM cases WHERE case_id = ?", (new_id,)).fetchone():
                raise ToolFault("already_exists", f"Case {new_id!r} already exists.")
            self._db.execute(
                "INSERT INTO cases (case_id, owner_id, policy_id, letter_text, status, created_at, question) VALUES (?,?,?,?,?,?,?)",
                (new_id, owner_id, policy_id, letter_text, "open", _now(), question.strip()),
            )
            return {"case_id": new_id, "status": "open"}

        return self._call("create_case", owner_id, case_id, args, run)

    def get_case(self, owner_id: str, case_id: str) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            row = self._owned_case(owner_id, case_id)
            notes = self._db.execute(
                "SELECT author, text, created_at FROM notes WHERE case_id = ? ORDER BY id", (case_id,)
            ).fetchall()
            deadlines = self._db.execute(
                "SELECT label, due_date FROM deadlines WHERE case_id = ? ORDER BY due_date, label", (case_id,)
            ).fetchall()
            decision = self._db.execute(
                "SELECT decision, reviewer, comment, created_at FROM decisions WHERE case_id = ?", (case_id,)
            ).fetchone()
            return {
                "case": {k: row[k] for k in ("case_id", "owner_id", "policy_id", "status", "created_at", "letter_text", "question")},
                "notes": [dict(n) for n in notes],
                "deadlines": [dict(d) for d in deadlines],
                "decision": dict(decision) if decision else None,
            }

        return self._call("get_case", owner_id, case_id, {"owner_id": owner_id, "case_id": case_id}, run)

    def add_note(
        self, owner_id: str, case_id: str, text: str, author: str = "system", dedupe_key: str | None = None
    ) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            self._owned_case(owner_id, case_id)
            cur = self._db.execute(
                "INSERT OR IGNORE INTO notes (case_id, author, text, dedupe_key, created_at) VALUES (?,?,?,?,?)",
                (case_id, author, text, dedupe_key, _now()),
            )
            return {"added": cur.rowcount == 1}

        args = {"owner_id": owner_id, "case_id": case_id, "text": text, "author": author, "dedupe_key": dedupe_key}
        return self._call("add_note", owner_id, case_id, args, run)

    def set_case_status(self, owner_id: str, case_id: str, status: str) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            if status not in OPEN_STATUSES:
                raise ToolFault(
                    "invalid_status", f"status must be one of {', '.join(OPEN_STATUSES)}. Final statuses are set by record_decision."
                )
            row = self._owned_case(owner_id, case_id)
            if row["status"] in FINAL_STATUSES.values():
                raise ToolFault("case_closed", "A decision has been recorded; the status can no longer change.")
            self._db.execute("UPDATE cases SET status = ? WHERE case_id = ?", (status, case_id))
            return {"status": status}

        args = {"owner_id": owner_id, "case_id": case_id, "status": status}
        return self._call("set_case_status", owner_id, case_id, args, run)

    def set_deadline(self, owner_id: str, case_id: str, label: str, due_date: str) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            self._owned_case(owner_id, case_id)
            if not isinstance(due_date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", due_date):
                raise ToolFault("invalid_date", "due_date must look like 2026-10-12.")
            try:
                date.fromisoformat(due_date)
            except ValueError:
                raise ToolFault("invalid_date", f"{due_date!r} is not a real calendar date.") from None
            self._db.execute(
                "INSERT INTO deadlines (case_id, label, due_date, created_at) VALUES (?,?,?,?) "
                "ON CONFLICT(case_id, label) DO UPDATE SET due_date = excluded.due_date",
                (case_id, label, due_date, _now()),
            )
            return {"label": label, "due_date": due_date}

        args = {"owner_id": owner_id, "case_id": case_id, "label": label, "due_date": due_date}
        return self._call("set_deadline", owner_id, case_id, args, run)

    def list_deadlines(self, owner_id: str, case_id: str) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            self._owned_case(owner_id, case_id)
            rows = self._db.execute(
                "SELECT label, due_date FROM deadlines WHERE case_id = ? ORDER BY due_date, label", (case_id,)
            ).fetchall()
            return {"deadlines": [dict(r) for r in rows]}

        return self._call("list_deadlines", owner_id, case_id, {"owner_id": owner_id, "case_id": case_id}, run)

    def record_decision(
        self, owner_id: str, case_id: str, decision: str, reviewer: str, comment: str = ""
    ) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            if decision not in FINAL_STATUSES:
                raise ToolFault("invalid_decision", "decision must be 'approved' or 'rejected'.")
            self._owned_case(owner_id, case_id)
            existing = self._db.execute(
                "SELECT decision, reviewer FROM decisions WHERE case_id = ?", (case_id,)
            ).fetchone()
            if existing is not None:
                if existing["decision"] == decision and existing["reviewer"] == reviewer:
                    return {"decision": decision, "status": FINAL_STATUSES[decision], "already_recorded": True}
                raise ToolFault("conflict", f"A different decision ({existing['decision']}) is already recorded.")
            self._db.execute(
                "INSERT INTO decisions (case_id, decision, reviewer, comment, created_at) VALUES (?,?,?,?,?)",
                (case_id, decision, reviewer, comment, _now()),
            )
            self._db.execute("UPDATE cases SET status = ? WHERE case_id = ?", (FINAL_STATUSES[decision], case_id))
            return {"decision": decision, "status": FINAL_STATUSES[decision], "already_recorded": False}

        args = {"owner_id": owner_id, "case_id": case_id, "decision": decision, "reviewer": reviewer, "comment": comment}
        return self._call("record_decision", owner_id, case_id, args, run)

    def get_audit_log(self, owner_id: str, case_id: str, limit: int = 100) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            self._owned_case(owner_id, case_id)
            rows = self._db.execute(
                "SELECT ts, tool, owner_id, status, error_code, args_sha256 FROM audit_log "
                "WHERE case_id = ? ORDER BY id DESC LIMIT ?",
                (case_id, max(1, min(int(limit), 500))),
            ).fetchall()
            return {"entries": [dict(r) for r in reversed(rows)]}

        args = {"owner_id": owner_id, "case_id": case_id, "limit": limit}
        return self._call("get_audit_log", owner_id, case_id, args, run)


def build_server(store: TrackerStore) -> FastMCP:
    mcp = FastMCP("claims-tracker", log_level="WARNING")

    @mcp.tool()
    def create_case(
        owner_id: OWNER_ID, policy_id: POLICY_ID, letter_text: LETTER_TEXT, case_id: CASE_ID | None = None, question: QUESTION = ""
    ) -> dict:
        """Open a new case from the text of a rejection or settlement letter, with the policyholder's question if they gave one."""
        return store.create_case(owner_id, policy_id, letter_text, case_id, question)

    @mcp.tool()
    def get_case(owner_id: OWNER_ID, case_id: CASE_ID) -> dict:
        """Return a case with its letter, notes, deadlines and recorded decision."""
        return store.get_case(owner_id, case_id)

    @mcp.tool()
    def add_note(owner_id: OWNER_ID, case_id: CASE_ID, text: NOTE_TEXT, author: SHORT_TEXT = "system", dedupe_key: SHORT_TEXT | None = None) -> dict:
        """Attach a note to a case. A repeated dedupe_key is ignored, so retries do not duplicate notes."""
        return store.add_note(owner_id, case_id, text, author, dedupe_key)

    @mcp.tool()
    def set_case_status(owner_id: OWNER_ID, case_id: CASE_ID, status: Literal["open", "awaiting_review", "needs_more_evidence"]) -> dict:
        """Set a working status. Final statuses can only be set by record_decision."""
        return store.set_case_status(owner_id, case_id, status)

    @mcp.tool()
    def set_deadline(owner_id: OWNER_ID, case_id: CASE_ID, label: SHORT_TEXT, due_date: DUE_DATE) -> dict:
        """Create or update a named deadline (ISO date). Repeating a label updates it."""
        return store.set_deadline(owner_id, case_id, label, due_date)

    @mcp.tool()
    def list_deadlines(owner_id: OWNER_ID, case_id: CASE_ID) -> dict:
        """List a case's deadlines, earliest first."""
        return store.list_deadlines(owner_id, case_id)

    @mcp.tool()
    def record_decision(owner_id: OWNER_ID, case_id: CASE_ID, decision: Literal["approved", "rejected"], reviewer: SHORT_TEXT, comment: COMMENT = "") -> dict:
        """Record the human reviewer's decision on the drafted appeal. Allowed once per case."""
        return store.record_decision(owner_id, case_id, decision, reviewer, comment)

    @mcp.tool()
    def get_audit_log(owner_id: OWNER_ID, case_id: CASE_ID, limit: Annotated[int, Field(ge=1, le=500)] = 100) -> dict:
        """Return the audit trail for one case: which tool ran, when, and with what outcome."""
        return store.get_audit_log(owner_id, case_id, limit)

    return mcp


def main() -> None:
    store = TrackerStore(os.environ.get("INSURITHM_TRACKER_DB", "insurithm_tracker.sqlite"))
    build_server(store).run(transport="stdio")


if __name__ == "__main__":
    main()
