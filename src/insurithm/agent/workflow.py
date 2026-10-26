"""The appeal workflow, as a LangGraph graph with a human approval step.

    intake -> investigate (loops) -> draft -> verify -> human_review -> finalize
                                                  \\-> END   (insufficient evidence)

* ``intake``       loads the case from the claims tracker, reads dates, amounts and clause
                   numbers from the letter with regular expressions, sets the appeal deadline,
                   and fetches the clauses the letter cites.
* ``investigate``  one model-chosen action per pass (search / get_clause / finish), executed
                   through the policy-library MCP server. Bounded by ``max_steps``.
* ``draft``        the model proposes findings, each with citations {chunk_id, quote}.
* ``verify``       deterministic. Citations that do not check out are dropped, and if nothing
                   survives the outcome is ``insufficient_evidence`` and no appeal is produced.
* ``human_review`` calls ``interrupt()``. The graph stops, its state is saved by the checkpointer,
                   and it continues only when someone resumes it with a decision.
* ``finalize``     records that decision in the claims tracker.

The workflow has no tool that sends, pays or deletes anything. "Approve" means the human
accepts the draft; getting it to the insurer is still the human's job.

All state is plain JSON (dicts, lists, strings, numbers) so the SQLite checkpointer can store it.
"""

from __future__ import annotations

import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, TypedDict

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from ..facts import extract_facts
from ..ingest import chunk_page
from ..models import Chunk
from ..verify import DEFAULT_MIN_OVERLAP, verify_findings
from .llm import LLM, LLMOutputError
from .prompts import draft_prompt, investigate_prompt, DRAFT_SYSTEM, INVESTIGATE_SYSTEM, label
from .tools import ToolError, Tools

MAX_FINDINGS = 8
MAX_EVIDENCE = 40
_CLAUSE_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){0,3}$")


class AppealState(TypedDict, total=False):
    case_id: str
    owner_id: str
    policy_id: str
    letter_text: str
    question: str
    facts: dict[str, Any]
    evidence: dict[str, dict[str, Any]]
    missing_clauses: list[str]
    trace: list[dict[str, Any]]
    steps_taken: int
    investigation_done: bool
    draft: dict[str, Any]
    verification: dict[str, Any]
    outcome: str
    appeal_text: str
    review: dict[str, Any]
    final_status: str


# --------------------------------------------------------------------------- helpers
def chunk_from_tool(policy_chunk: dict[str, Any]) -> dict[str, Any]:
    """Convert a policy-library result into the Chunk shape used in state."""
    return Chunk(
        chunk_id=policy_chunk["chunk_id"],
        doc_id=policy_chunk["policy_id"],
        page=policy_chunk["page"],
        section=policy_chunk.get("section"),
        clause_id=policy_chunk.get("clause_id"),
        text=policy_chunk["text"],
        start_char=policy_chunk["start_char"],
        end_char=policy_chunk["end_char"],
    ).model_dump()


def letter_chunks(case_id: str, letter_text: str) -> dict[str, dict[str, Any]]:
    chunks, _ = chunk_page(f"letter-{case_id}", 1, letter_text)
    return {c.chunk_id: c.model_dump() for c in chunks}


def pool_from(evidence: dict[str, dict[str, Any]]) -> dict[str, Chunk]:
    return {cid: Chunk(**c) for cid, c in evidence.items()}


def normalise_review(value: Any) -> dict[str, str]:
    """Accept {"decision": "approve"|"reject", "reviewer": ..., "comment": ...}."""
    if not isinstance(value, dict):
        raise ValueError("review decision must be an object")
    raw = str(value.get("decision", "")).strip().lower()
    decision = {"approve": "approved", "approved": "approved", "reject": "rejected", "rejected": "rejected"}.get(raw)
    if decision is None:
        raise ValueError("decision must be 'approve' or 'reject'")
    reviewer = str(value.get("reviewer") or "reviewer").strip()[:64] or "reviewer"
    return {"decision": decision, "reviewer": reviewer, "comment": str(value.get("comment") or "")[:1000]}


def render_appeal(state: AppealState) -> str:
    """Build the appeal text from verified findings only. Quotes come from the source, not the model."""
    facts = state.get("facts", {})
    by_kind: dict[str, list[str]] = {}
    for f in facts.get("facts", []):
        by_kind.setdefault(f["kind"], []).append(f["value"])
    verification = state["verification"]
    draft = state.get("draft", {})

    lines = ["APPEAL DRAFT - for your review. Nothing has been sent to anyone.", ""]
    refs = []
    if by_kind.get("claim_number"):
        refs.append(f"Claim {by_kind['claim_number'][0]}")
    if by_kind.get("policy_number"):
        refs.append(f"Policy {by_kind['policy_number'][0]}")
    if refs:
        lines.append("Re: " + ", ".join(refs))
    names = {"rejected": "claim rejected", "delayed": "claim delayed", "short_paid": "claim paid short"}
    decision = ", ".join(names.get(d, d) for d in facts.get("decision_types") or []) or "decision on the claim"
    dated = f" (letter dated {facts['letter_date']})" if facts.get("letter_date") else ""
    lines += [f"Decision challenged: {decision}{dated}", "", "I ask you to reconsider this decision for the following reasons.", ""]

    for n, finding in enumerate(verification["supported"], start=1):
        lines.append(f"{n}. {finding['claim']}")
        for cit in (c for c in finding["citations"] if c["valid"]):
            where = label({"doc_id": cit["doc_id"], "page": cit["page"], "clause_id": cit["clause_id"]})
            quote = " ".join(cit["quote"].split())
            lines.append(f'   Evidence ({where}): "{quote}"')
        lines.append("")

    gaps = [g for g in draft.get("gaps", []) if isinstance(g, str) and g.strip()]
    if gaps:
        lines.append("Open points - no supporting passage was found, so check these before relying on them:")
        lines += [f"  - {' '.join(g.split())}" for g in gaps[:8]]
        lines.append("")
    docs = [d for d in draft.get("documents_to_gather", []) if isinstance(d, str) and d.strip()]
    if docs:
        lines.append("Documents to gather:")
        lines += [f"  - {' '.join(d.split())}" for d in docs[:8]]
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- graph
def build_graph(llm: LLM, tools: Tools, *, checkpointer: Any, max_steps: int = 4, min_overlap: float = DEFAULT_MIN_OVERLAP):
    """Compile the workflow. `checkpointer` decides where state is saved (SQLite in production)."""

    async def intake(state: AppealState) -> dict[str, Any]:
        owner, case_id = state["owner_id"], state["case_id"]
        case = (await tools.tracker.call("get_case", owner_id=owner, case_id=case_id))["case"]
        letter = case["letter_text"]
        facts = extract_facts(letter)

        deadline = facts.appeal_deadline
        if deadline:
            await tools.tracker.call(
                "set_deadline", owner_id=owner, case_id=case_id, label="Appeal window (earliest)", due_date=deadline[0]
            )

        evidence = letter_chunks(case_id, letter)
        trace: list[dict[str, Any]] = []
        missing: list[str] = []
        for ref in facts.clause_refs[:6]:
            if not _CLAUSE_RE.match(ref):
                continue
            try:
                found = await tools.policies.call("get_clause", policy_id=case["policy_id"], clause_id=ref)
            except ToolError as err:
                if err.code == "not_found":
                    missing.append(ref)
                trace.append({"action": "get_clause", "clause_id": ref, "source": "letter", "error": err.code})
                continue
            for chunk in found["results"]:
                evidence[chunk["chunk_id"]] = chunk_from_tool(chunk)
            trace.append({"action": "get_clause", "clause_id": ref, "source": "letter", "found": len(found["results"])})

        summary = (
            f"Intake: clauses cited {facts.clause_refs or 'none'}; decision {facts.decision_types or 'unclear'}; "
            f"missing from policy {missing or 'none'}."
        )
        await tools.tracker.call("add_note", owner_id=owner, case_id=case_id, text=summary, author="system", dedupe_key="intake")
        return {
            "policy_id": case["policy_id"],
            "letter_text": letter,
            "question": case.get("question", ""),
            "facts": facts.to_dict(),
            "evidence": evidence,
            "missing_clauses": missing,
            "trace": trace,
            "steps_taken": 0,
            "investigation_done": False,
        }

    async def investigate(state: AppealState) -> dict[str, Any]:
        steps = state.get("steps_taken", 0)
        evidence = dict(state["evidence"])
        trace = list(state.get("trace", []))
        prompt = investigate_prompt(
            state["letter_text"], state["facts"], evidence, state.get("missing_clauses", []), max_steps - steps,
            state.get("question", ""),
        )
        done = False
        try:
            reply = await llm.complete_json("investigate", INVESTIGATE_SYSTEM, prompt)
        except LLMOutputError as err:
            reply = {"action": "invalid", "error": str(err)}

        action = reply.get("action")
        entry: dict[str, Any] = {"step": steps + 1, "action": action}
        previous = {(t.get("action"), t.get("query") or t.get("clause_id")) for t in trace}

        try:
            if action == "finish":
                done = True
            elif action == "search":
                query = reply.get("query")
                if not isinstance(query, str) or not 2 <= len(query.strip()) <= 300:
                    entry["error"] = "invalid_query"
                elif ("search", query) in previous:
                    entry.update(query=query, error="duplicate_action")
                else:
                    entry["query"] = query
                    found = await tools.policies.call(
                        "search_policy_clauses", query=query.strip(), k=5, policy_id=state["policy_id"]
                    )
                    for chunk in found["results"]:
                        evidence[chunk["chunk_id"]] = chunk_from_tool(chunk)
                    entry["returned"] = [c["chunk_id"] for c in found["results"]]
            elif action == "get_clause":
                clause_id = reply.get("clause_id")
                if not isinstance(clause_id, str) or not _CLAUSE_RE.match(clause_id.strip()):
                    entry["error"] = "invalid_clause_id"
                elif ("get_clause", clause_id) in previous:
                    entry.update(clause_id=clause_id, error="duplicate_action")
                else:
                    entry["clause_id"] = clause_id
                    found = await tools.policies.call("get_clause", policy_id=state["policy_id"], clause_id=clause_id.strip())
                    for chunk in found["results"]:
                        evidence[chunk["chunk_id"]] = chunk_from_tool(chunk)
                    entry["returned"] = [c["chunk_id"] for c in found["results"]]
            else:
                entry["error"] = reply.get("error") or "unknown_action"
        except ToolError as err:
            entry["error"] = err.code

        if len(evidence) > MAX_EVIDENCE:
            done = True
        trace.append(entry)
        steps += 1
        return {
            "evidence": evidence,
            "trace": trace,
            "steps_taken": steps,
            "investigation_done": done or steps >= max_steps,
        }

    def after_investigate(state: AppealState) -> str:
        return "draft" if state.get("investigation_done") else "investigate"

    async def draft(state: AppealState) -> dict[str, Any]:
        prompt = draft_prompt(
            state["letter_text"], state["facts"], state["evidence"], state.get("missing_clauses", []), state.get("question", "")
        )
        reply = await llm.complete_json("draft", DRAFT_SYSTEM, prompt)
        findings = reply.get("findings")
        return {
            "draft": {
                "findings": findings[:MAX_FINDINGS] if isinstance(findings, list) else [],
                "gaps": reply.get("gaps") if isinstance(reply.get("gaps"), list) else [],
                "documents_to_gather": reply.get("documents_to_gather")
                if isinstance(reply.get("documents_to_gather"), list)
                else [],
            }
        }

    async def verify(state: AppealState) -> dict[str, Any]:
        owner, case_id = state["owner_id"], state["case_id"]
        result = verify_findings(state["draft"]["findings"], pool_from(state["evidence"]), min_overlap)
        update: dict[str, Any] = {"verification": result.to_dict(), "outcome": result.outcome}
        if result.outcome == "draft_ready":
            update["appeal_text"] = render_appeal({**state, **update})  # type: ignore[typeddict-item]
            await tools.tracker.call("set_case_status", owner_id=owner, case_id=case_id, status="awaiting_review")
        else:
            gaps = [g for g in state["draft"].get("gaps", []) if isinstance(g, str)][:5]
            note = "Insufficient evidence: the policy text provided does not support an appeal point."
            if gaps:
                note += " Unsupported: " + "; ".join(gaps)
            await tools.tracker.call("set_case_status", owner_id=owner, case_id=case_id, status="needs_more_evidence")
            await tools.tracker.call("add_note", owner_id=owner, case_id=case_id, text=note[:1900], author="system", dedupe_key="insufficient")
            update["final_status"] = "needs_more_evidence"
        return update

    def after_verify(state: AppealState) -> str:
        return "human_review" if state["outcome"] == "draft_ready" else END

    async def human_review(state: AppealState) -> dict[str, Any]:
        # Nothing before interrupt() may have side effects: on resume this node runs again from the top.
        decision = interrupt(
            {
                "kind": "appeal_review",
                "case_id": state["case_id"],
                "appeal_text": state["appeal_text"],
                "findings": state["verification"]["supported"],
                "dropped_findings": state["verification"]["unsupported"],
                "options": ["approve", "reject"],
            }
        )
        return {"review": normalise_review(decision)}

    async def finalize(state: AppealState) -> dict[str, Any]:
        review = state["review"]
        recorded = await tools.tracker.call(
            "record_decision",
            owner_id=state["owner_id"],
            case_id=state["case_id"],
            decision=review["decision"],
            reviewer=review["reviewer"],
            comment=review["comment"],
        )
        await tools.tracker.call(
            "add_note",
            owner_id=state["owner_id"],
            case_id=state["case_id"],
            text=f"{review['reviewer']} {review['decision']} the appeal draft. {review['comment']}".strip(),
            author=review["reviewer"],
            dedupe_key="decision",
        )
        return {"final_status": recorded["status"]}

    graph = StateGraph(AppealState)
    graph.add_node("intake", intake)
    graph.add_node("investigate", investigate)
    graph.add_node("draft", draft)
    graph.add_node("verify", verify)
    graph.add_node("human_review", human_review)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "intake")
    graph.add_edge("intake", "investigate")
    graph.add_conditional_edges("investigate", after_investigate, {"investigate": "investigate", "draft": "draft"})
    graph.add_edge("draft", "verify")
    graph.add_conditional_edges("verify", after_verify, {"human_review": "human_review", END: END})
    graph.add_edge("human_review", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


# --------------------------------------------------------------------------- running it
class AppealRunner:
    """Starts, resumes and inspects workflow runs. One run per case; the thread id is the case id."""

    def __init__(self, graph: Any):
        self.graph = graph

    @staticmethod
    def _config(case_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": case_id}}

    async def state_values(self, case_id: str) -> dict[str, Any]:
        """The full saved workflow state (evidence, raw draft, verification). For evaluation and debugging."""
        return dict((await self.graph.aget_state(self._config(case_id))).values)

    async def status(self, case_id: str) -> dict[str, Any]:
        """Where the run is, read from the saved checkpoint (so it works after a restart)."""
        snap = await self.graph.aget_state(self._config(case_id))
        if not snap.values:
            return {"status": "not_started"}
        interrupts = [i for task in snap.tasks for i in task.interrupts]
        values = snap.values
        base = {
            "outcome": values.get("outcome"),
            "steps_taken": values.get("steps_taken"),
            "trace": values.get("trace", []),
            "facts": values.get("facts"),
        }
        if interrupts:
            return {"status": "awaiting_review", "review": interrupts[0].value, **base}
        if snap.next:
            return {"status": "incomplete", "next": list(snap.next), **base}
        return {
            "status": "completed",
            "final_status": values.get("final_status"),
            "appeal_text": values.get("appeal_text"),
            "verification": values.get("verification"),
            "review": values.get("review"),
            **base,
        }

    async def start(self, case_id: str, owner_id: str) -> dict[str, Any]:
        """Run from the beginning until the human review step (or the end)."""
        await self.graph.ainvoke({"case_id": case_id, "owner_id": owner_id}, self._config(case_id))
        return await self.status(case_id)

    async def resume_after_failure(self, case_id: str) -> dict[str, Any]:
        """Continue a run that stopped part-way (for example after a crash or a model error)."""
        await self.graph.ainvoke(None, self._config(case_id))
        return await self.status(case_id)

    async def decide(self, case_id: str, decision: dict[str, Any]) -> dict[str, Any]:
        """Resume a paused run with the human's decision."""
        await self.graph.ainvoke(Command(resume=normalise_review(decision)), self._config(case_id))
        return await self.status(case_id)


@asynccontextmanager
async def open_runner(
    llm: LLM, tools: Tools, checkpoint_db: str | Path, **graph_options: Any
) -> AsyncIterator[AppealRunner]:
    """Open the SQLite checkpoint file and yield a runner. Leaving the block closes the file."""
    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_db)) as saver:
        yield AppealRunner(build_graph(llm, tools, checkpointer=saver, **graph_options))
