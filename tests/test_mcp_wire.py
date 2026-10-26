"""These tests start the two MCP servers as real child processes and talk to them over stdio."""

import os

import pytest
import pytest_asyncio

from insurithm.agent.tools import MCPConnection, Tools, ToolError, server_env

pytestmark = pytest.mark.asyncio(loop_scope="module")

PID = "synthetic_health_policy"
LETTER = "Your claim is rejected under clause 4.2 of the policy. Date: 12 September 2026."


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def tools(tmp_path_factory):
    db = tmp_path_factory.mktemp("tracker") / "tracker.sqlite"
    async with Tools.local(db) as t:
        yield t


async def test_servers_advertise_only_narrow_tools(tools):
    policy = {t["name"] for t in await tools.policies.list_tools()}
    tracker = {t["name"] for t in await tools.tracker.list_tools()}
    assert policy == {"list_policies", "search_policy_clauses", "get_clause", "get_chunk"}
    assert tracker == {
        "create_case", "get_case", "add_note", "set_case_status",
        "set_deadline", "list_deadlines", "record_decision", "get_audit_log",
    }
    forbidden = ("delete", "remove", "drop", "send", "pay", "email", "submit")
    assert not [n for n in policy | tracker if any(word in n for word in forbidden)]


async def test_every_tool_has_a_schema_with_constraints(tools):
    listing = {t["name"]: t["input_schema"] for t in await tools.policies.list_tools()}
    props = listing["search_policy_clauses"]["properties"]
    assert props["k"]["maximum"] == 10 and props["query"]["maxLength"] == 500


async def test_search_over_the_wire(tools):
    out = await tools.policies.call("search_policy_clauses", query="waiting period for pre-existing disease", k=3)
    assert out["results"][0]["clause_id"] == "3.2" and out["results"][0]["page"] == 3
    clause = await tools.policies.call("get_clause", policy_id=PID, clause_id="4.2")
    assert "Proportionate Deduction" in clause["results"][0]["text"]


async def test_schema_violations_are_rejected_before_any_code_runs(tools):
    with pytest.raises(ToolError) as err:
        await tools.policies.call("search_policy_clauses", query="waiting period", k=50)
    assert err.value.code == "invalid_call"
    with pytest.raises(ToolError):
        await tools.policies.call("search_policy_clauses", query="x")
    with pytest.raises(ToolError):
        await tools.policies.call("get_clause", policy_id=PID, clause_id="4.2; DROP TABLE")
    with pytest.raises(ToolError):
        await tools.tracker.call("record_decision", owner_id="asha", case_id="abc", decision="maybe", reviewer="asha")


async def test_tool_errors_carry_the_tools_own_code(tools):
    with pytest.raises(ToolError) as err:
        await tools.policies.call("get_clause", policy_id=PID, clause_id="99.9")
    assert err.value.code == "not_found"
    with pytest.raises(ToolError) as err:
        await tools.policies.call("get_clause", policy_id="ghost", clause_id="1.1")
    assert err.value.code == "unknown_policy"


async def test_resource_and_prompt(tools):
    page = await tools.policies.read_resource(f"policy://{PID}/page/6")
    assert "6.2 Emergency Admission" in page
    assert "exactly" in (await tools.policies.get_prompt("appeal_structure"))
    with pytest.raises(Exception):
        await tools.policies.read_resource(f"policy://{PID}/page/99")


async def test_tracker_access_control_over_the_wire(tools):
    made = await tools.tracker.call("create_case", owner_id="asha", policy_id=PID, letter_text=LETTER, case_id="wire-001")
    assert made["status"] == "open"
    with pytest.raises(ToolError) as err:
        await tools.tracker.call("get_case", owner_id="mallory", case_id="wire-001")
    assert err.value.code == "not_found"
    await tools.tracker.call("set_deadline", owner_id="asha", case_id="wire-001", label="appeal", due_date="2026-10-12")
    log = await tools.tracker.call("get_audit_log", owner_id="asha", case_id="wire-001")
    assert [(e["tool"], e["status"]) for e in log["entries"]] == [
        ("create_case", "ok"), ("get_case", "denied"), ("set_deadline", "ok"),
    ]


async def test_connection_restarts_if_stopped(tmp_path):
    conn = MCPConnection("policy-library", "insurithm.mcp_servers.policy_library")
    await conn.start()
    assert (await conn.call("list_policies"))["policies"][0]["policy_id"] == PID
    await conn.stop()
    assert (await conn.call("list_policies"))["ok"]  # started again on demand
    await conn.stop()


async def test_a_server_that_cannot_start_fails_cleanly():
    conn = MCPConnection("broken", "insurithm.does_not_exist")
    with pytest.raises(ToolError) as err:
        await conn.start()
    assert err.value.code == "server_unavailable"


async def test_child_processes_do_not_inherit_the_model_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret")
    monkeypatch.setenv("INSURITHM_TRACKER_DB", "/x/y.sqlite")
    env = server_env()
    assert "GEMINI_API_KEY" not in env and "super-secret" not in env.values()
    assert env["INSURITHM_TRACKER_DB"] == "/x/y.sqlite"
    assert "PYTHONPATH" in env
