"""Scripted 'models' for testing the evaluation harness. They are not claims about any real model."""

from __future__ import annotations

import re

from insurithm.evals.compare import normalize

BLOCK = re.compile(r"\[([^\]\n]+)\] \(([^)\n]*)\)\n(.*?)(?=\n\n\[|\n\nReply with)", re.DOTALL)


def find_case(cases, prompt):
    for c in cases:
        if c["letter"][:150] in prompt:
            return c
    raise AssertionError("prompt does not contain any known letter")


def case_from_draft_prompt(cases, prompt):
    for c in cases:
        if f"letter-eval-{c['id'].lower()}-p1" in prompt:
            return c
    raise AssertionError("draft prompt has no letter chunk")


def chunks_in(prompt):
    return [{"chunk_id": m.group(1), "label": m.group(2), "text": m.group(3)} for m in BLOCK.finditer(prompt)]


def oracle_findings(case, prompt):
    findings = []
    for exp in case["expected"]:
        for ch in chunks_in(prompt):
            page = re.search(r"page (\d+)", ch["label"])
            if page and int(page.group(1)) == exp["page"] and normalize(exp["passage"]) in normalize(ch["text"]):
                findings.append({
                    "claim": f"The policy states: {exp['passage']}.",
                    "citations": [{"chunk_id": ch["chunk_id"], "quote": exp["passage"]}],
                })
                break
    return findings


class OracleLLM:
    """Knows the expected passages. Retrieves and cites them whenever they are available to it."""

    def __init__(self, cases, extra_bad_finding=False):
        self.cases, self.extra_bad = cases, extra_bad_finding

    async def complete_json(self, stage, system, prompt):
        if stage == "investigate":
            case = find_case(self.cases, prompt)
            have = prompt.split("Policy evidence collected so far:")[1].split("Actions left")[0]
            collected_clauses = set(re.findall(r", clause ([\d.]+) \|", have))
            collected_pages = set(re.findall(r"page (\d+)(?:, clause [\d.]+)? \|", have))
            for exp in case["expected"]:
                if exp["clause"] and exp["clause"] not in collected_clauses:
                    return {"action": "get_clause", "clause_id": exp["clause"]}
                if not exp["clause"] and str(exp["page"]) not in collected_pages:
                    return {"action": "search", "query": "Haemodialysis chemotherapy radiotherapy cataract day care procedures"}
            return {"action": "finish"}
        case = case_from_draft_prompt(self.cases, prompt)
        findings = oracle_findings(case, prompt)
        if self.extra_bad:
            some = chunks_in(prompt)[0]["chunk_id"]
            findings.append({"claim": "The insurer owes a penalty of 25 percent.", "citations": [{"chunk_id": some, "quote": "a penalty of 25 percent is payable"}]})
        return {"findings": findings, "gaps": [], "documents_to_gather": []}


class HallucinatingLLM:
    """Always asserts things with made-up quotes, and never retrieves anything extra."""

    async def complete_json(self, stage, system, prompt):
        if stage == "investigate":
            return {"action": "finish"}
        some = chunks_in(prompt)[0]["chunk_id"]
        return {
            "findings": [
                {"claim": "The insurer breached clause 12.4 of the policy.", "citations": [{"chunk_id": some, "quote": "clause 12.4 forbids this deduction entirely"}]},
                {"claim": "The policy guarantees payment within seven days.", "citations": []},
            ],
            "gaps": [],
            "documents_to_gather": [],
        }
