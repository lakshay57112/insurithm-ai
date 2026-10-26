import sqlite3

import pytest

from insurithm.mcp_servers.claims_tracker import TrackerStore

LETTER = "Your claim is rejected under clause 4.2 of the policy. Date: 12 September 2026."


@pytest.fixture()
def store(tmp_path):
    s = TrackerStore(tmp_path / "t.sqlite")
    yield s
    s.close()


@pytest.fixture()
def case(store):
    out = store.create_case("asha", "policy-1", LETTER, "case-001")
    assert out == {"ok": True, "case_id": "case-001", "status": "open"}
    return "case-001"


def code(result):
    assert result["ok"] is False
    return result["error"]["code"]


def test_create_and_get_case(store, case):
    got = store.get_case("asha", case)
    assert got["ok"] and got["case"]["letter_text"] == LETTER
    assert got["case"]["status"] == "open" and got["notes"] == [] and got["decision"] is None


def test_generated_case_ids_are_unique(store):
    ids = {store.create_case("asha", "p", LETTER)["case_id"] for _ in range(5)}
    assert len(ids) == 5


def test_duplicate_and_invalid_input_is_rejected(store, case):
    assert code(store.create_case("asha", "p", LETTER, case)) == "already_exists"
    assert code(store.create_case("asha", "p", "too short")) == "invalid_letter"
    assert code(store.create_case("bad owner!", "p", LETTER)) == "invalid_owner"
    assert code(store.get_case("asha", "x")) == "invalid_case_id"


def test_other_owners_cannot_see_or_change_a_case(store, case):
    for result in (
        store.get_case("mallory", case),
        store.add_note("mallory", case, "hi"),
        store.set_deadline("mallory", case, "x", "2026-10-01"),
        store.list_deadlines("mallory", case),
        store.record_decision("mallory", case, "approved", "mallory"),
        store.set_case_status("mallory", case, "open"),
        store.get_audit_log("mallory", case),
    ):
        assert code(result) == "not_found"  # same answer as a case that does not exist
    assert code(store.get_case("asha", "does-not-exist")) == "not_found"
    # Nothing changed, and the owner can see the refused attempts.
    assert store.get_case("asha", case)["decision"] is None
    denied = [e for e in store.get_audit_log("asha", case)["entries"] if e["status"] == "denied"]
    assert len(denied) == 7 and {e["owner_id"] for e in denied} == {"mallory"}


def test_decision_is_enumerated_recorded_once_and_closes_the_case(store, case):
    assert code(store.record_decision("asha", case, "maybe", "asha")) == "invalid_decision"
    ok = store.record_decision("asha", case, "approved", "asha", "looks right")
    assert ok["status"] == "appeal_approved" and ok["already_recorded"] is False
    assert store.get_case("asha", case)["case"]["status"] == "appeal_approved"
    # A retry of the same decision is harmless; a different one is a conflict.
    assert store.record_decision("asha", case, "approved", "asha")["already_recorded"] is True
    assert code(store.record_decision("asha", case, "rejected", "asha")) == "conflict"
    assert code(store.set_case_status("asha", case, "open")) == "case_closed"


def test_final_statuses_cannot_be_set_directly(store, case):
    assert code(store.set_case_status("asha", case, "appeal_approved")) == "invalid_status"
    assert store.set_case_status("asha", case, "awaiting_review")["status"] == "awaiting_review"


def test_deadlines_validate_dates_and_update_by_label(store, case):
    assert code(store.set_deadline("asha", case, "appeal", "2026-02-31")) == "invalid_date"
    assert code(store.set_deadline("asha", case, "appeal", "12/10/2026")) == "invalid_date"
    store.set_deadline("asha", case, "appeal", "2026-10-12")
    store.set_deadline("asha", case, "appeal", "2026-10-15")  # same label: updated, not duplicated
    store.set_deadline("asha", case, "documents", "2026-09-30")
    assert store.list_deadlines("asha", case)["deadlines"] == [
        {"label": "documents", "due_date": "2026-09-30"},
        {"label": "appeal", "due_date": "2026-10-15"},
    ]


def test_notes_with_a_dedupe_key_are_not_repeated(store, case):
    assert store.add_note("asha", case, "first", "system", "k1")["added"] is True
    assert store.add_note("asha", case, "first again", "system", "k1")["added"] is False
    assert store.add_note("asha", case, "no key", "asha")["added"] is True
    assert store.add_note("asha", case, "no key", "asha")["added"] is True
    assert [n["text"] for n in store.get_case("asha", case)["notes"]] == ["first", "no key", "no key"]


def test_audit_log_records_every_call_without_storing_content(store, tmp_path, case):
    store.add_note("asha", case, "a private medical detail", "asha")
    store.record_decision("asha", case, "bogus", "asha")
    entries = store.get_audit_log("asha", case)["entries"]
    assert [(e["tool"], e["status"]) for e in entries] == [
        ("create_case", "ok"),  # from the fixture
        ("add_note", "ok"),
        ("record_decision", "error"),
    ]
    assert all(len(e["args_sha256"]) == 64 for e in entries)
    # The raw table never contains the letter or the note text.
    db = sqlite3.connect(tmp_path / "t.sqlite")
    dump = " ".join(str(v) for row in db.execute("SELECT * FROM audit_log") for v in row)
    assert "private medical detail" not in dump and "rejected under clause" not in dump


def test_state_survives_reopening_the_database(tmp_path):
    first = TrackerStore(tmp_path / "t.sqlite")
    first.create_case("asha", "p", LETTER, "case-keep")
    first.record_decision("asha", "case-keep", "rejected", "asha", "not now")
    first.close()
    second = TrackerStore(tmp_path / "t.sqlite")
    got = second.get_case("asha", "case-keep")
    assert got["case"]["status"] == "appeal_rejected" and got["decision"]["comment"] == "not now"


def test_the_question_is_stored_trimmed_and_limited(store):
    store.create_case("asha", "p", LETTER, "case-q", "  Why was my claim cut?  ")
    assert store.get_case("asha", "case-q")["case"]["question"] == "Why was my claim cut?"
    store.create_case("asha", "p", LETTER, "case-nq")
    assert store.get_case("asha", "case-nq")["case"]["question"] == ""
    assert code(store.create_case("asha", "p", LETTER, "case-long", "x" * 1001)) == "invalid_question"


def test_a_database_from_before_the_question_field_still_opens(tmp_path):
    old = sqlite3.connect(tmp_path / "old.sqlite")
    old.executescript(
        "CREATE TABLE cases (case_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, policy_id TEXT NOT NULL,"
        " letter_text TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL);"
        "INSERT INTO cases VALUES ('old-case', 'asha', 'p', 'an old letter that is long enough', 'open', '2026-01-01T00:00:00+00:00');"
    )
    old.commit()
    old.close()
    migrated = TrackerStore(tmp_path / "old.sqlite")
    assert migrated.get_case("asha", "old-case")["case"]["question"] == ""
    assert migrated.create_case("asha", "p", LETTER, "new-case", "a question")["ok"]
    migrated.close()
