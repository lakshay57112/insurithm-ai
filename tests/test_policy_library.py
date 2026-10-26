import shutil

import pytest

from insurithm.config import DEFAULT_POLICY_PDF
from insurithm.mcp_servers.policy_library import PolicyLibrary

PID = DEFAULT_POLICY_PDF.stem


@pytest.fixture(scope="module")
def lib():
    return PolicyLibrary([DEFAULT_POLICY_PDF])


def test_list_policies(lib):
    out = lib.list_policies()
    assert out == {"ok": True, "policies": [{"policy_id": PID, "pages": 8, "chunks": 54}]}


def test_search_returns_page_numbered_ranked_clauses(lib):
    out = lib.search("emergency admission notice to the claims desk", k=3)
    assert out["ok"] and out["policy_id"] == PID
    top = out["results"][0]
    assert (top["clause_id"], top["page"], top["rank"]) == ("6.2", 6, 1)
    assert {"chunk_id", "text", "start_char", "end_char", "score"} <= set(top)
    assert [r["rank"] for r in out["results"]] == [1, 2, 3]


def test_get_clause_and_chunk_round_trip(lib):
    clause = lib.get_clause(PID, "4.2")
    assert clause["ok"] and clause["results"][0]["page"] == 4
    chunk_id = clause["results"][0]["chunk_id"]
    assert lib.get_chunk(chunk_id)["chunk"]["text"] == clause["results"][0]["text"]


def test_missing_things_are_reported_not_raised(lib):
    assert lib.get_clause(PID, "99.9")["error"]["code"] == "not_found"
    assert lib.get_chunk("nope")["error"]["code"] == "not_found"
    assert lib.search("waiting period", policy_id="ghost")["error"]["code"] == "unknown_policy"


def test_allow_list_blocks_every_path_to_a_policy():
    locked = PolicyLibrary([DEFAULT_POLICY_PDF], allowed={"some_other_policy"})
    assert locked.list_policies()["policies"] == []
    assert locked.search("waiting period", policy_id=PID)["error"]["code"] == "access_denied"
    assert locked.get_clause(PID, "3.2")["error"]["code"] == "access_denied"
    chunk_id = f"{PID}-p3-c1"
    assert locked.get_chunk(chunk_id)["error"]["code"] == "access_denied"
    with pytest.raises(ValueError, match="not allowed"):
        locked.read_page(PID, 3)


def test_policy_id_is_required_when_several_policies_are_served(tmp_path):
    copy = tmp_path / "second_policy.pdf"
    shutil.copy(DEFAULT_POLICY_PDF, copy)
    two = PolicyLibrary([DEFAULT_POLICY_PDF, copy])
    assert two.search("waiting period")["error"]["code"] == "policy_id_required"
    assert two.search("waiting period", policy_id="second_policy")["ok"]
    assert {p["policy_id"] for p in two.list_policies()["policies"]} == {PID, "second_policy"}


def test_read_page_bounds(lib):
    assert "6.2 Emergency Admission" in lib.read_page(PID, 6)
    for bad in (0, 9):
        with pytest.raises(ValueError, match="pages 1 to 8"):
            lib.read_page(PID, bad)


def test_server_has_no_write_capability(lib):
    import asyncio

    from insurithm.mcp_servers.policy_library import build_server

    names = {t.name for t in asyncio.run(build_server(lib).list_tools())}
    assert names == {"list_policies", "search_policy_clauses", "get_clause", "get_chunk"}


# ---- uploaded policies (a folder that is re-read on every call) ----------------------------------
def test_uploaded_policies_appear_and_disappear(tmp_path):
    from pdfs import policy_pdf

    uploads = tmp_path / "uploads"
    uploads.mkdir()
    library = PolicyLibrary([DEFAULT_POLICY_PDF], upload_dir=uploads)
    assert [p["policy_id"] for p in library.list_policies()["policies"]] == [PID]

    (uploads / "my-plan.pdf").write_bytes(policy_pdf())
    listed = {p["policy_id"]: p for p in library.list_policies()["policies"]}
    assert set(listed) == {PID, "my-plan"} and listed["my-plan"]["pages"] == 1
    found = library.search("waiting period pre-existing diseases", policy_id="my-plan")
    assert found["ok"] and found["results"][0]["clause_id"] == "2.1"
    assert library.get_clause("my-plan", "1.2")["ok"]

    (uploads / "my-plan.pdf").unlink()
    assert [p["policy_id"] for p in library.list_policies()["policies"]] == [PID]
    assert library.search("waiting", policy_id="my-plan")["error"]["code"] == "unknown_policy"


def test_a_bad_upload_is_skipped_and_does_not_break_the_others(tmp_path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "broken.pdf").write_bytes(b"%PDF-1.4 this is not a pdf")
    (uploads / ".half-written.tmp").write_bytes(b"x")
    library = PolicyLibrary([DEFAULT_POLICY_PDF], upload_dir=uploads)
    assert [p["policy_id"] for p in library.list_policies()["policies"]] == [PID]
    assert library.search("emergency admission", policy_id=PID)["ok"]


def test_uploads_are_served_even_when_the_built_in_allow_list_is_narrow(tmp_path):
    from pdfs import policy_pdf

    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "mine.pdf").write_bytes(policy_pdf())
    library = PolicyLibrary([DEFAULT_POLICY_PDF], allowed=set(), upload_dir=uploads)  # no built-in policy allowed
    assert [p["policy_id"] for p in library.list_policies()["policies"]] == ["mine"]
    assert library.search("waiting", policy_id=PID)["error"]["code"] == "access_denied"
