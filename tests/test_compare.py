"""Tests for the evaluation harness itself, using scripted models. They check the maths and the plumbing,
not how good any real model is."""

import json
import sys
import tempfile
from pathlib import Path

import pytest
from fakes import HallucinatingLLM, OracleLLM

from insurithm.agent.llm import LLMError, LLMQuotaError, ScriptedLLM
from insurithm.config import DEFAULT_POLICY_PDF
from insurithm.evals import compare as compare_module
from insurithm.evals.compare import (
    CaseRun, Progress, QuotaStopped, compare, load_cases, render_markdown, run_agent_system, run_baseline_system, score,
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


async def test_progress_lines_show_each_call_and_failures(capsys):
    from insurithm.agent.llm import LLMError
    from insurithm.evals.compare import ProgressLLM

    class Inner:
        async def complete_json(self, stage, system, prompt):
            if stage == "bad":
                raise LLMError("boom")
            return {"ok": True}

    llm = ProgressLLM(Inner())
    assert await llm.complete_json("draft", "s", "p") == {"ok": True}
    try:
        await llm.complete_json("bad", "s", "p")
    except LLMError:
        pass
    else:
        raise AssertionError("the error must still reach the caller")
    err = capsys.readouterr().err
    assert "call 1 (draft)" in err and "call 1 done" in err
    assert "call 2 failed" in err and "boom" in err


# ----------------------------------------------------------------------------- stopping on quota and resuming
QUOTA_MESSAGE = "Gemini's usage limit is used up; it asks to wait about 12 hours (Quota exceeded)"


class QuotaAfter:
    """Answers like the oracle for `limit` calls, then reports that the usage limit is used up."""

    def __init__(self, limit=None):
        self.limit, self.calls, self.inner = limit, 0, OracleLLM(CASES)

    async def complete_json(self, stage, system, prompt):
        if self.limit is not None and self.calls >= self.limit:
            raise LLMQuotaError(QUOTA_MESSAGE)
        self.calls += 1
        return await self.inner.complete_json(stage, system, prompt)


async def test_a_quota_error_stops_the_run_and_keeps_finished_cases(tmp_path):
    progress = Progress(tmp_path / "p.json", "key")
    chunks = ingest_pdf(DEFAULT_POLICY_PDF)
    with pytest.raises(QuotaStopped, match="12 hours"):
        await run_baseline_system(CASES[:4], QuotaAfter(limit=2), chunks, progress=progress)
    assert set(progress.done("baseline")) == {CASES[0]["id"], CASES[1]["id"]}

    resumed = Progress(tmp_path / "p.json", "key", resume=True)       # a new process reads the file
    llm = QuotaAfter()
    runs = await run_baseline_system(CASES[:4], llm, chunks, progress=resumed)
    assert llm.calls == 2                                              # only the two unfinished cases were asked
    assert [r.case_id for r in runs] == [c["id"] for c in CASES[:4]] and not any(r.error for r in runs)


def test_saved_progress_is_ignored_when_the_settings_differ_or_resume_is_off(tmp_path):
    path = tmp_path / "p.json"
    Progress(path, "key-a").add(CaseRun("C01", "baseline", model_calls=1))
    assert Progress(path, "key-a", resume=True).done("baseline")["C01"].model_calls == 1
    assert Progress(path, "key-b", resume=True).done("baseline") == {}   # other model or cases
    assert Progress(path, "key-a", resume=False).done("baseline") == {}  # not asked to resume
    path.write_text("not json")
    assert Progress(path, "key-a", resume=True).done("baseline") == {}   # a damaged file never crashes the run


def test_errors_are_not_saved_as_finished_and_the_report_says_the_run_is_incomplete():
    cases = [{"id": "A", "expected_outcome": "appeal", "expected": [{"page": 1, "clause": None, "passage": "p"}]}]
    report = {
        "model": "m", "run_at": "now", "n_cases": 1, "caveat": "caveat",
        "systems": {"baseline": score(cases, [CaseRun("A", "baseline", error="HTTP 503")])},
    }
    assert "Incomplete run" in render_markdown(report) and "--resume" in render_markdown(report)
    report["systems"]["baseline"] = score(cases, [CaseRun("A", "baseline")])
    assert "Incomplete run" not in render_markdown(report)


def run_cli(monkeypatch, tmp_path, llm, *extra):
    class Stub:
        model = "stub-model"

        @classmethod
        def from_env(cls):
            return cls()

        async def complete_json(self, stage, system, prompt):
            return await llm.complete_json(stage, system, prompt)

        async def aclose(self):
            pass

    monkeypatch.setattr(compare_module, "GeminiLLM", Stub)
    monkeypatch.setattr(sys, "argv", ["compare", "--systems", "baseline", "--limit", "3", "--min-interval", "0",
                                      "--out-dir", str(tmp_path), *extra])
    compare_module.main()


def test_command_line_stops_cleanly_on_quota_then_resumes(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit) as stop:
        run_cli(monkeypatch, tmp_path, QuotaAfter(limit=2))
    assert stop.value.code == 2
    err = capsys.readouterr().err
    assert "Stopped early" in err and "12 hours" in err and "--resume" in err and "baseline: 2 case(s)" in err
    assert not list(tmp_path.glob("comparison_*"))                     # no misleading half-finished report

    llm = QuotaAfter()
    run_cli(monkeypatch, tmp_path, llm, "--resume")
    assert llm.calls == 1                                              # only the one case that was left
    report = json.loads((tmp_path / "comparison_stub-model.json").read_text())
    assert report["n_cases"] == 3 and report["systems"]["baseline"]["errors"] == 0
