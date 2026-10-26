"""Tests for the evaluation harness itself, using scripted models. They check the maths and the plumbing,
not how good any real model is."""

import tempfile
from pathlib import Path

import pytest
from fakes import HallucinatingLLM, OracleLLM

from insurithm.agent.llm import LLMError, ScriptedLLM
from insurithm.config import DEFAULT_POLICY_PDF
from insurithm.evals.compare import (
    CaseRun, compare, load_cases, render_markdown, run_agent_system, run_baseline_system, score,
)
from insurithm.ingest import ingest_pdf

CASES = load_cases()


def finding(supported, citations):
    return {"claim": "x", "supported": supported, "citations": citations}


def cit(chunk_id, valid=True):
    return {"chunk_id": chunk_id, "quote": "q", "valid": valid}


def test_score_arithmetic_on_hand_built_runs():
    cases = [
        {"id": "A", "expected_outcome": "appeal", "expected": [{"page": 4, "clause": "4.2", "passage": "room above the limit"}]},
        {"id": "B", "expected_outcome": "appeal", "expected": [{"page": 6, "clause": "6.6", "passage": "within 30 days"}]},
        {"id": "N", "expected_outcome": "insufficient_evidence", "expected": []},
        {"id": "M", "expected_outcome": "insufficient_evidence", "expected": []},
    ]
    pool_a = {"c42": {"chunk_id": "c42", "doc_id": "pol", "page": 4, "text": "occupies a room above the limit in 4.1"}}
    good = finding(True, [cit("c42")])
    bad = finding(False, [cit("zzz", valid=False)])
    runs = [
        CaseRun("A", "s", raw=[good, bad], shipped=[good], pool=pool_a, model_calls=4),             # correct, ships 1 good
        CaseRun("B", "s", raw=[bad], shipped=[bad], pool={}, model_calls=2),                        # wrong: ships an unsupported claim
        CaseRun("N", "s", raw=[bad], shipped=[], pool={}, model_calls=3),                           # correct abstention
        CaseRun("M", "s", raw=[good], shipped=[good], pool=pool_a, model_calls=3),                  # should have abstained
    ]
    s = score(cases, runs)
    assert s["answer_correctness_appeal"] == 0.5
    assert s["evidence_coverage_appeal"] == 0.5
    assert s["pool_recall"] == 0.5
    assert s["abstention_correct"] == 0.5
    assert s["false_abstention_rate"] == 0.0
    assert s["citation_correctness"] == round(2 / 5, 4)           # valid: 2 of 5 proposed
    assert s["claims_proposed"] == 5 and s["unsupported_claim_rate_proposed"] == round(3 / 5, 4)
    assert s["claims_shipped"] == 3 and s["unsupported_claim_rate_shipped"] == round(1 / 3, 4)
    assert s["mean_model_calls"] == 3.0


def test_an_error_is_never_counted_as_a_correct_abstention_or_answer():
    cases = [
        {"id": "A", "expected_outcome": "appeal", "expected": [{"page": 1, "clause": None, "passage": "p"}]},
        {"id": "N", "expected_outcome": "insufficient_evidence", "expected": []},
    ]
    runs = [CaseRun("A", "s", error="boom"), CaseRun("N", "s", error="boom")]
    s = score(cases, runs)
    assert s["errors"] == 2 and s["answer_correctness_appeal"] == 0.0 and s["abstention_correct"] == 0.0
    assert s["false_abstention_rate"] == 0.0  # failed runs are errors, not abstentions


async def test_a_perfect_oracle_scores_perfectly_through_the_agent():
    with tempfile.TemporaryDirectory() as tmp:
        runs = await run_agent_system(CASES, OracleLLM(CASES), Path(tmp))
    s = score(CASES, runs)
    assert s["errors"] == 0
    assert s["pool_recall"] == 1.0 and s["citation_correctness"] == 1.0
    assert s["answer_correctness_appeal"] == 1.0 and s["evidence_coverage_appeal"] == 1.0
    assert s["abstention_correct"] == 1.0 and s["false_abstention_rate"] == 0.0
    assert s["unsupported_claim_rate_proposed"] == 0.0 and s["unsupported_claim_rate_shipped"] == 0.0


async def test_the_baseline_only_sees_what_one_retrieval_returns():
    runs = await run_baseline_system(CASES, OracleLLM(CASES), ingest_pdf(DEFAULT_POLICY_PDF))
    s = score(CASES, runs)
    assert s["mean_model_calls"] == 1.0
    assert s["errors"] == 0
    # Even a perfect writer cannot cite passages the retriever never returned.
    assert 0.0 < s["pool_recall"] <= 1.0
    assert s["citation_correctness"] == 1.0


async def test_the_agent_filters_fabrications_that_the_baseline_ships():
    with tempfile.TemporaryDirectory() as tmp:
        agent = score(CASES, await run_agent_system(CASES, OracleLLM(CASES, extra_bad_finding=True), Path(tmp)))
    base = score(CASES, await run_baseline_system(CASES, OracleLLM(CASES, extra_bad_finding=True), ingest_pdf(DEFAULT_POLICY_PDF)))
    # Same fabricated finding proposed by the same model...
    assert agent["unsupported_claim_rate_proposed"] > 0 and base["unsupported_claim_rate_proposed"] > 0
    # ...but only the baseline ships it.
    assert agent["unsupported_claim_rate_shipped"] == 0.0
    assert base["unsupported_claim_rate_shipped"] == base["unsupported_claim_rate_proposed"] > 0
    # And the baseline can never abstain on cases where the model adds a fabricated claim to nothing.
    assert agent["abstention_correct"] == 1.0 and base["abstention_correct"] == 0.0


async def test_a_model_that_only_hallucinates_makes_the_agent_abstain_everywhere():
    with tempfile.TemporaryDirectory() as tmp:
        agent = score(CASES, await run_agent_system(CASES, HallucinatingLLM(), Path(tmp)))
    base = score(CASES, await run_baseline_system(CASES, HallucinatingLLM(), ingest_pdf(DEFAULT_POLICY_PDF)))
    assert agent["claims_shipped"] == 0 and agent["unsupported_claim_rate_shipped"] is None
    assert agent["answer_correctness_appeal"] == 0.0 and agent["false_abstention_rate"] == 1.0
    assert agent["abstention_correct"] == 1.0
    assert base["unsupported_claim_rate_shipped"] == 1.0 and base["abstention_correct"] == 0.0


async def test_model_failures_are_recorded_per_case_not_fatal():
    broken = ScriptedLLM([LLMError("Gemini API error 503")] * 40)
    runs = await run_baseline_system(CASES[:3], broken, ingest_pdf(DEFAULT_POLICY_PDF))
    assert all(r.error and "503" in r.error for r in runs)
    assert score(CASES[:3], runs)["errors"] == 3


async def test_full_report_and_markdown():
    report = await compare(OracleLLM(CASES), "scripted-oracle", ("baseline",), CASES[:4])
    assert report["model"] == "scripted-oracle" and report["n_cases"] == 4
    md = render_markdown(report)
    assert "| Metric | baseline |" in md and "Citation correctness" in md and "synthetic" in md


def test_error_lines_explain_why_cases_failed():
    from insurithm.evals.compare import error_lines

    report = {"systems": {
        "agent": {"per_case": [{"case": "A1", "error": "Gemini API error 404: model not found"},
                               {"case": "A2", "error": "Gemini API error 404: model not found"},
                               {"case": "A3", "error": None}]},
        "baseline": {"per_case": [{"case": "A1", "error": None}]},
    }}
    assert error_lines(report) == ["agent: Gemini API error 404: model not found (cases: A1, A2)"]
    assert error_lines({"systems": {"agent": {"per_case": [{"case": "A1"}]}}}) == []
