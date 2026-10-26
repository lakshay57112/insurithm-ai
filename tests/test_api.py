import pytest
from fastapi.testclient import TestClient

import scenarios as S
from insurithm.agent.llm import FunctionLLM, LLMError, ScriptedLLM
from insurithm.api import create_app

H = {"X-Owner-Id": "asha"}


def new_case(client, case_id, headers=H, letter=S.LETTER):
    r = client.post("/cases", json={"letter_text": letter, "case_id": case_id}, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()


def test_full_flow_over_http(tmp_path):
    llm = ScriptedLLM(S.full_script("api-001"))
    with TestClient(create_app(llm, tmp_path)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/policies").json()[0]["policy_id"] == S.POLICY_ID

        assert new_case(client, "api-001") == {"case_id": "api-001", "status": "open", "policy_id": S.POLICY_ID}

        run = client.post("/cases/api-001/run", headers=H)
        assert run.status_code == 200
        body = run.json()
        assert body["status"] == "awaiting_review" and body["outcome"] == "draft_ready"
        assert "Evidence (policy synthetic_health_policy, page 4, clause 4.2)" in body["review"]["appeal_text"]

        again = client.post("/cases/api-001/run", headers=H)  # running again changes nothing
        assert again.json()["status"] == "awaiting_review" and len(llm.calls) == 3

        shown = client.get("/cases/api-001", headers=H).json()
        assert shown["case"]["status"] == "awaiting_review" and "owner_id" not in shown["case"]
        assert shown["workflow"]["status"] == "awaiting_review" and shown["decision"] is None
        assert {"label": "Appeal window (earliest)", "due_date": "2026-10-12"} in shown["deadlines"]

        done = client.post("/cases/api-001/decision", headers=H, json={"decision": "approve", "reviewer": "asha", "comment": "ok"})
        assert done.status_code == 200 and done.json()["final_status"] == "appeal_approved"

        final = client.get("/cases/api-001", headers=H).json()
        assert final["case"]["status"] == "appeal_approved" and final["decision"]["comment"] == "ok"
        tools = [e["tool"] for e in client.get("/cases/api-001/audit", headers=H).json()["entries"]]
        assert "record_decision" in tools and "create_case" in tools


def test_owner_header_is_required_and_cases_are_private(tmp_path):
    with TestClient(create_app(ScriptedLLM([]), tmp_path)) as client:
        assert client.post("/cases", json={"letter_text": S.LETTER}).status_code == 401
        assert client.post("/cases", json={"letter_text": S.LETTER}, headers={"X-Owner-Id": "bad owner!"}).status_code == 401
        new_case(client, "private-1")
        other = {"X-Owner-Id": "mallory"}
        for method, url, kwargs in [
            ("get", "/cases/private-1", {}),
            ("post", "/cases/private-1/run", {}),
            ("post", "/cases/private-1/decision", {"json": {"decision": "approve", "reviewer": "m"}}),
            ("get", "/cases/private-1/audit", {}),
        ]:
            assert getattr(client, method)(url, headers=other, **kwargs).status_code == 404, url
        assert client.get("/cases/nope-nope", headers=H).status_code == 404  # same answer as for someone else's case


def test_input_validation(tmp_path):
    with TestClient(create_app(ScriptedLLM([]), tmp_path)) as client:
        assert client.post("/cases", json={"letter_text": "short"}, headers=H).status_code == 422
        assert client.post("/cases", json={"letter_text": S.LETTER, "case_id": "x"}, headers=H).status_code == 422
        assert client.post("/cases", json={"letter_text": S.LETTER, "policy_id": "ghost"}, headers=H).status_code == 404
        new_case(client, "dup-001")
        assert client.post("/cases", json={"letter_text": S.LETTER, "case_id": "dup-001"}, headers=H).status_code == 409
        bad = client.post("/cases/dup-001/decision", headers=H, json={"decision": "maybe", "reviewer": "a"})
        assert bad.status_code == 422


def test_a_decision_needs_a_pending_review(tmp_path):
    with TestClient(create_app(ScriptedLLM([]), tmp_path)) as client:
        new_case(client, "early-01")
        r = client.post("/cases/early-01/decision", headers=H, json={"decision": "approve", "reviewer": "asha"})
        assert r.status_code == 409 and "not_started" in r.json()["detail"]


def test_insufficient_evidence_ends_without_an_approval_step(tmp_path):
    script = [{"action": "finish"}, {"findings": [], "gaps": ["Nothing supports an appeal."], "documents_to_gather": []}]
    with TestClient(create_app(ScriptedLLM(script), tmp_path)) as client:
        new_case(client, "thin-001")
        body = client.post("/cases/thin-001/run", headers=H).json()
        assert body["status"] == "completed" and body["final_status"] == "needs_more_evidence" and body["appeal_text"] is None
        r = client.post("/cases/thin-001/decision", headers=H, json={"decision": "approve", "reviewer": "asha"})
        assert r.status_code == 409


def test_a_model_failure_is_a_502_and_run_can_be_retried(tmp_path):
    state = {"fail": True}
    replies = iter(S.investigate_replies())

    def fn(stage, system, prompt):
        if stage == "investigate":
            return next(replies)
        if state["fail"]:
            state["fail"] = False
            raise LLMError("Gemini API error 503: overloaded")
        return S.draft_reply("retry-01")

    with TestClient(create_app(FunctionLLM(fn), tmp_path)) as client:
        new_case(client, "retry-01")
        failed = client.post("/cases/retry-01/run", headers=H)
        assert failed.status_code == 502 and "Call run again" in failed.json()["detail"]
        assert client.get("/cases/retry-01", headers=H).json()["workflow"]["status"] == "incomplete"
        ok = client.post("/cases/retry-01/run", headers=H)
        assert ok.status_code == 200 and ok.json()["status"] == "awaiting_review"


def test_the_server_can_be_restarted_while_a_draft_waits_for_review(tmp_path):
    with TestClient(create_app(ScriptedLLM(S.full_script("rest-001")), tmp_path)) as first:
        new_case(first, "rest-001")
        text = first.post("/cases/rest-001/run", headers=H).json()["review"]["appeal_text"]

    never = ScriptedLLM([])
    with TestClient(create_app(never, tmp_path)) as second:   # new app, new tool servers, same data folder
        waiting = second.get("/cases/rest-001", headers=H).json()
        assert waiting["workflow"]["status"] == "awaiting_review" and waiting["workflow"]["review"]["appeal_text"] == text
        done = second.post("/cases/rest-001/decision", headers=H, json={"decision": "approve", "reviewer": "asha"})
        assert done.json()["final_status"] == "appeal_approved"
    assert never.calls == []
