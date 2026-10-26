"""File upload through the API: letters (PDF or text) and the uploader's own policy PDFs."""

import re

import pytest
from fastapi.testclient import TestClient

from insurithm import api
from insurithm.agent.llm import FunctionLLM, ScriptedLLM
from insurithm.api import create_app
from pdfs import blank_pdf, make_pdf, policy_pdf

ASHA = {"X-Owner-Id": "asha"}
MALLORY = {"X-Owner-Id": "mallory"}
LETTER_TEXT = "Your claim for hospitalisation is rejected because of the waiting period. Date: 12 September 2026."


@pytest.fixture()
def client(tmp_path):
    with TestClient(create_app(ScriptedLLM([]), tmp_path)) as c:
        yield c


# ---- letters ---------------------------------------------------------------------------------------
def test_a_text_letter_is_read(client):
    r = client.post("/extract-letter", content=LETTER_TEXT.encode(), headers=ASHA)
    assert r.status_code == 200
    assert r.json() == {"text": LETTER_TEXT, "kind": "text", "pages": None, "truncated": False}


def test_windows_text_files_are_read_too(client):
    letter = "Café Assurance regrets that your claim is rejected. Date: 12 September 2026."
    r = client.post("/extract-letter", content=letter.encode("cp1252"), headers=ASHA)
    assert r.status_code == 200 and r.json()["text"] == letter


def test_a_pdf_letter_is_read(client):
    pdf = make_pdf([["Dear Policyholder,", "Your claim is rejected under clause 2.1 of the policy."], ["Appeal within 30 days of this letter."]])
    r = client.post("/extract-letter", content=pdf, headers=ASHA).json()
    assert r["kind"] == "pdf" and r["pages"] == 2
    assert "rejected under clause 2.1" in r["text"] and "Appeal within 30 days" in r["text"]


def test_bad_letter_files_get_a_plain_explanation(client):
    assert client.post("/extract-letter", content=b"", headers=ASHA).status_code == 422
    assert client.post("/extract-letter", content=b"\x89PNG\r\n\x1a\n\x00\x00binary", headers=ASHA).status_code == 415
    scan = client.post("/extract-letter", content=blank_pdf(), headers=ASHA)
    assert scan.status_code == 422 and "paste the text" in scan.json()["detail"]
    assert client.post("/extract-letter", content=b"too short", headers=ASHA).status_code == 422
    assert client.post("/extract-letter", content=b"%PDF-1.4 not a pdf", headers=ASHA).status_code == 422


def test_letter_upload_needs_a_name_and_has_a_size_limit(client, monkeypatch):
    assert client.post("/extract-letter", content=LETTER_TEXT.encode()).status_code == 401
    monkeypatch.setattr(api, "MAX_LETTER_BYTES", 50)
    r = client.post("/extract-letter", content=LETTER_TEXT.encode(), headers=ASHA)
    assert r.status_code == 413


def test_a_very_long_letter_is_cut_and_flagged(client):
    r = client.post("/extract-letter", content=("word " * 6000).encode(), headers=ASHA).json()
    assert r["truncated"] is True and len(r["text"]) == 20000


# ---- policies --------------------------------------------------------------------------------------
def upload(client, headers=ASHA, name="My Health Plan.pdf", data=None):
    return client.post("/policies", params={"name": name}, content=policy_pdf() if data is None else data, headers=headers)


def test_an_uploaded_policy_is_listed_for_its_owner_only(client):
    made = upload(client)
    assert made.status_code == 201, made.text
    body = made.json()
    assert body["title"] == "my-health-plan" and body["mine"] is True and body["pages"] == 1
    assert re.fullmatch(r"u[0-9a-f]{16}-my-health-plan", body["policy_id"])

    mine = client.get("/policies", headers=ASHA).json()
    assert [(p["title"], p["mine"]) for p in mine] == [("synthetic_health_policy", False), ("my-health-plan", True)]  # shared first, then yours
    for headers in (MALLORY, {}):
        others = client.get("/policies", headers=headers).json()
        assert [p["policy_id"] for p in others] == ["synthetic_health_policy"]


def test_nobody_else_can_open_a_case_against_your_policy(client):
    policy_id = upload(client).json()["policy_id"]
    letter = {"letter_text": LETTER_TEXT, "policy_id": policy_id}
    assert client.post("/cases", json=letter, headers=MALLORY).status_code == 404
    assert client.post("/cases", json={**letter, "case_id": "own-case"}, headers=ASHA).status_code == 201


def test_a_full_review_runs_against_an_uploaded_policy(tmp_path):
    seen = {}

    def model(stage, system, prompt):
        if stage == "investigate":
            return {"action": "search", "query": "waiting period pre-existing diseases"} if "Actions left: 4" in prompt else {"action": "finish"}
        seen["draft_prompt"] = prompt
        chunk = re.search(r"\[(u[0-9a-f]{16}-[^\]]+-p1-c\d+)\] \(policy ([^,]+), page 1, clause 2\.1\)", prompt)
        assert chunk, prompt
        seen["label"] = chunk.group(2)
        return {"findings": [{
            "claim": "Pre-existing diseases are payable after 24 months of continuous coverage.",
            "citations": [{"chunk_id": chunk.group(1), "quote": "payable only after 24 months of"}],
        }], "gaps": [], "documents_to_gather": []}

    with TestClient(create_app(FunctionLLM(model), tmp_path)) as client:
        policy_id = upload(client).json()["policy_id"]
        client.post("/cases", json={"letter_text": LETTER_TEXT, "policy_id": policy_id, "case_id": "up-001", "question": "Is the waiting period 36 months?"}, headers=ASHA)
        run = client.post("/cases/up-001/run", headers=ASHA).json()
        assert run["status"] == "awaiting_review" and run["outcome"] == "draft_ready"
        finding = run["review"]["findings"][0]
        assert finding["citations"][0]["valid"] and finding["citations"][0]["clause_id"] == "2.1"
        # The private owner prefix never reaches the text a person would send to an insurer.
        assert seen["label"] == "my-health-plan"
        assert "policy my-health-plan, page 1, clause 2.1" in run["review"]["appeal_text"]
        assert policy_id not in run["review"]["appeal_text"]
        assert "Is the waiting period 36 months?" in seen["draft_prompt"]


def test_bad_policy_uploads_are_refused_with_a_reason(client, monkeypatch):
    assert upload(client, data=b"just some words").status_code == 415
    scan = upload(client, data=blank_pdf())
    assert scan.status_code == 422 and "almost no readable text" in scan.json()["detail"]
    assert upload(client, data=b"%PDF-1.4 broken").status_code == 422
    assert client.post("/policies", content=policy_pdf()).status_code == 401
    monkeypatch.setattr(api, "MAX_POLICY_BYTES", 100)
    assert upload(client).status_code == 413
    assert [p["policy_id"] for p in client.get("/policies", headers=ASHA).json()] == ["synthetic_health_policy"]


def test_uploading_the_same_name_twice_keeps_both_and_the_count_is_limited(client, monkeypatch):
    first, second = upload(client).json(), upload(client).json()
    assert first["policy_id"] != second["policy_id"] and second["title"] == "my-health-plan-2"
    monkeypatch.setattr(api, "MAX_POLICIES_PER_OWNER", 2)
    refused = upload(client)
    assert refused.status_code == 409 and "up to 2" in refused.json()["detail"]
    assert upload(client, headers=MALLORY).status_code == 201  # the limit is per person
