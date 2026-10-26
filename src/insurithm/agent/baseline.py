"""The non-agentic baseline: retrieve once, ask the model once, ship whatever it says.

It uses the same drafting instructions and the same output format as the agent's draft step,
so the comparison isolates what the agent adds: choosing its own searches, fetching the clauses
the letter cites, checking every citation, and abstaining when nothing checks out.
"""

from __future__ import annotations

from typing import Any

from ..facts import extract_facts
from ..retrieval import Retriever
from .llm import LLM
from .prompts import DRAFT_SYSTEM, draft_prompt
from .workflow import MAX_FINDINGS, letter_chunks

QUERY_CHARS = 1000


async def run_baseline(case_id: str, letter_text: str, retriever: Retriever, llm: LLM, k: int = 5) -> dict[str, Any]:
    evidence = letter_chunks(case_id, letter_text)
    hits = retriever.search(letter_text[:QUERY_CHARS], k=k)
    for hit in hits:
        evidence[hit.chunk.chunk_id] = hit.chunk.model_dump()

    facts = extract_facts(letter_text).to_dict()
    reply = await llm.complete_json("baseline_draft", DRAFT_SYSTEM, draft_prompt(letter_text, facts, evidence, []))
    findings = reply.get("findings")
    return {
        "findings": findings[:MAX_FINDINGS] if isinstance(findings, list) else [],
        "evidence": evidence,
        "retrieved": [h.chunk.chunk_id for h in hits],
    }
