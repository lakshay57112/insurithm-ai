"""Agent vs. non-agentic baseline on the synthetic appeal cases.

For every case in ``data/gold/gold_cases.json`` both systems get the same letter and the same policy.
Every output is graded by the same deterministic checker (`insurithm.verify`) against the same evidence
pool, so the two are scored identically. What differs is only what each system *ships*:

* baseline: everything the model wrote;
* agent:    only the findings the checker accepted, or nothing at all if none survive.

Metrics (definitions in docs/evaluation.md):

* pool recall           share of expected policy passages that were in the retrieved evidence
* citation correctness  valid citations / all citations the model proposed (before any filtering)
* unsupported-claim rate  claims with no valid citation, among proposed (raw) and among shipped claims
* answer correctness    appeal cases: every expected passage cited by a supported finding;
                        abstain cases: nothing shipped
* abstention            how often nothing was shipped when nothing should have been, and when it should have

Run with a real model:  python -m insurithm.evals.compare --systems agent baseline
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..agent.baseline import run_baseline
from ..agent.llm import LLM, CountingLLM, GeminiLLM, LLMError, LLMQuotaError, ThrottledLLM
from ..agent.tools import Tools, ToolError
from ..agent.workflow import open_runner
from ..console import ensure_utf8_output
from ..config import DEFAULT_POLICY_PDF, GOLD_DIR, REPORT_DIR
from ..ingest import ingest_pdf
from ..models import Chunk
from ..retrieval import BM25Retriever
from ..verify import DEFAULT_MIN_OVERLAP, verify_findings

DEFAULT_CASES_FILE = GOLD_DIR / "gold_cases.json"
POLICY_ID = DEFAULT_POLICY_PDF.stem


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def load_cases(path: str | Path = DEFAULT_CASES_FILE) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["cases"]


@dataclass
class CaseRun:
    case_id: str
    system: str
    raw: list[dict[str, Any]] = field(default_factory=list)      # every finding the model proposed, checked
    shipped: list[dict[str, Any]] = field(default_factory=list)  # what the user would have received
    pool: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_calls: int = 0
    error: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CaseRun":
        return cls(**data)


def code_fingerprint() -> str:
    """Changes whenever the code that produces results changes, so results from older code are never mixed in."""
    root = Path(__file__).resolve().parent.parent
    files = sorted([*(root / "agent").glob("*.py"), *(root / "mcp_servers").glob("*.py"),
                    *(root / n for n in ("verify.py", "retrieval.py", "facts.py", "ingest.py", "models.py"))])
    digest = hashlib.sha256()
    for f in files:
        digest.update(f.name.encode())
        digest.update(f.read_bytes().replace(b"\r\n", b"\n"))  # same hash whether git checked out CRLF or LF
    return digest.hexdigest()[:16]


class QuotaStopped(RuntimeError):
    """The model's usage limit ran out part-way through. Whatever finished before that is saved."""


class Progress:
    """Finished cases, saved after each one, so a run that stopped can carry on where it left off."""

    def __init__(self, path: Path | None, key: str, resume: bool = False):
        self.path, self.key = path, key
        self.ignored = False  # True when a saved file was found but belongs to other code, model, settings or cases
        self.runs: dict[str, dict[str, CaseRun]] = {}
        if path and resume and path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
            if isinstance(data, dict) and data.get("key") != key:
                self.ignored = True
            if isinstance(data, dict) and data.get("key") == key:  # same code, model, settings and cases
                self.runs = {
                    system: {cid: CaseRun.from_dict(r) for cid, r in cases.items()}
                    for system, cases in data.get("runs", {}).items()
                }

    def done(self, system: str) -> dict[str, CaseRun]:
        return self.runs.get(system, {})

    def add(self, run: CaseRun) -> None:
        self.runs.setdefault(run.system, {})[run.case_id] = run
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            payload = {"key": self.key, "runs": {sy: {c: asdict(r) for c, r in cs.items()} for sy, cs in self.runs.items()}}
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp, self.path)

    def counts(self) -> str:
        return ", ".join(f"{system}: {len(cases)} case(s)" for system, cases in self.runs.items()) or "nothing yet"


# ----------------------------------------------------------------------------- scoring
def _is_letter(chunk: dict[str, Any]) -> bool:
    return str(chunk["doc_id"]).startswith("letter-")


def _passage_in(chunk: dict[str, Any], expected: dict[str, Any]) -> bool:
    return chunk["page"] == expected["page"] and normalize(expected["passage"]) in normalize(chunk["text"])


def _covered(run: CaseRun, expected: dict[str, Any]) -> bool:
    for finding in run.shipped:
        if not finding["supported"]:
            continue
        for cit in finding["citations"]:
            chunk = run.pool.get(cit["chunk_id"])
            if cit["valid"] and chunk is not None and _passage_in(chunk, expected):
                return True
    return False


def _name(expected: dict[str, Any]) -> str:
    return f"{expected.get('clause') or 'letter'} (page {expected['page']})"


def _cites(run: CaseRun, finding: dict[str, Any]) -> list[str]:
    out = []
    for cit in finding["citations"]:
        chunk = run.pool.get(cit["chunk_id"])
        where = f"page {chunk['page']}" + (f", clause {chunk['clause_id']}" if chunk and chunk.get("clause_id") else "") if chunk else "unknown chunk"
        out.append(where + ("" if cit["valid"] else " (quote did not check out)"))
    return out


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def score(cases: list[dict[str, Any]], runs: list[CaseRun]) -> dict[str, Any]:
    by_id = {c["id"]: c for c in cases}
    appeal = [r for r in runs if by_id[r.case_id]["expected_outcome"] == "appeal"]
    abstain = [r for r in runs if by_id[r.case_id]["expected_outcome"] != "appeal"]

    exp_total = exp_in_pool = exp_covered = 0
    correct_appeal = 0
    per_case = []
    for r in appeal:
        expected = by_id[r.case_id]["expected"]
        policy_chunks = [c for c in r.pool.values() if not _is_letter(c)]
        in_pool = sum(1 for e in expected if any(_passage_in(c, e) for c in policy_chunks))
        covered = sum(1 for e in expected if _covered(r, e)) if not r.error else 0
        exp_total += len(expected)
        exp_in_pool += in_pool
        exp_covered += covered
        ok = (not r.error) and covered == len(expected)
        correct_appeal += ok
        never_found = [e for e in expected if not any(_passage_in(c, e) for c in policy_chunks)]
        not_cited = [e for e in expected if e not in never_found and (r.error or not _covered(r, e))]
        why = None
        if r.error:
            why = f"error: {r.error}"
        elif not ok:
            parts = []
            if not r.shipped:
                parts.append("gave no appeal points")
            if never_found:
                parts.append("never retrieved " + ", ".join(_name(e) for e in never_found))
            if not_cited:
                parts.append("retrieved but not cited " + ", ".join(_name(e) for e in not_cited))
            why = "; ".join(parts)
        per_case.append({"case": r.case_id, "kind": "appeal", "correct": ok, "expected": len(expected),
                         "in_pool": in_pool, "covered": covered, "shipped": len(r.shipped), "error": r.error, "why": why,
                         "claims": [{"claim": f["claim"][:200], "supported": f["supported"], "cites": _cites(r, f)} for f in r.shipped]})
    correct_abstain = 0
    for r in abstain:
        ok = (not r.error) and not r.shipped
        correct_abstain += ok
        why = None if ok else (f"error: {r.error}" if r.error else f"gave {len(r.shipped)} appeal point(s) where the evidence is not enough")
        per_case.append({"case": r.case_id, "kind": "abstain", "correct": ok, "shipped": len(r.shipped), "error": r.error, "why": why})

    raw_claims = sum(len(r.raw) for r in runs)
    raw_unsupported = sum(1 for r in runs for f in r.raw if not f["supported"])
    shipped_claims = sum(len(r.shipped) for r in runs)
    shipped_unsupported = sum(1 for r in runs for f in r.shipped if not f["supported"])
    cits = [c for r in runs for f in r.raw for c in f["citations"]]
    valid_cits = sum(1 for c in cits if c["valid"])
    false_abstain = sum(1 for r in appeal if not r.error and not r.shipped)

    return {
        "cases": len(runs),
        "appeal_cases": len(appeal),
        "abstain_cases": len(abstain),
        "errors": sum(1 for r in runs if r.error),
        "pool_recall": _rate(exp_in_pool, exp_total),
        "citation_correctness": _rate(valid_cits, len(cits)),
        "citations_proposed": len(cits),
        "claims_proposed": raw_claims,
        "unsupported_claim_rate_proposed": _rate(raw_unsupported, raw_claims),
        "claims_shipped": shipped_claims,
        "unsupported_claim_rate_shipped": _rate(shipped_unsupported, shipped_claims),
        "answer_correctness_appeal": _rate(correct_appeal, len(appeal)),
        "evidence_coverage_appeal": _rate(exp_covered, exp_total),
        "abstention_correct": _rate(correct_abstain, len(abstain)),
        "false_abstention_rate": _rate(false_abstain, len(appeal)),
        "mean_model_calls": round(sum(r.model_calls for r in runs) / len(runs), 2) if runs else None,
        "per_case": per_case,
    }


# ----------------------------------------------------------------------------- running the systems
def _checked(findings: Any, pool: dict[str, dict[str, Any]], min_overlap: float) -> list[dict[str, Any]]:
    result = verify_findings(findings, {k: Chunk(**v) for k, v in pool.items()}, min_overlap)
    return [f.to_dict() for f in (*result.supported, *result.unsupported)]


async def run_baseline_system(
    cases: list[dict[str, Any]], llm: LLM, chunks: list[Chunk], *, k: int = 5, min_overlap: float = DEFAULT_MIN_OVERLAP,
    progress: Progress | None = None,
) -> list[CaseRun]:
    retriever = BM25Retriever(chunks)
    done = progress.done("baseline") if progress else {}
    runs = []
    for c in cases:
        if c["id"] in done:
            runs.append(done[c["id"]])
            continue
        counting = CountingLLM(llm)
        run = CaseRun(c["id"], "baseline")
        try:
            out = await run_baseline(f"eval-{c['id'].lower()}", c["letter"], retriever, counting, k=k)
            run.pool = out["evidence"]
            run.raw = _checked(out["findings"], run.pool, min_overlap)
            run.shipped = run.raw  # no verifier: the user gets everything
        except LLMQuotaError as err:
            raise QuotaStopped(str(err)) from None
        except LLMError as err:
            run.error = str(err)
        run.model_calls = counting.count
        if progress and not run.error:
            progress.add(run)
        runs.append(run)
    return runs


async def run_agent_system(
    cases: list[dict[str, Any]], llm: LLM, work_dir: Path, *, max_steps: int = 4, min_overlap: float = DEFAULT_MIN_OVERLAP,
    progress: Progress | None = None,
) -> list[CaseRun]:
    work_dir.mkdir(parents=True, exist_ok=True)
    done = progress.done("agent") if progress else {}
    runs = []
    async with Tools.local(work_dir / "tracker.sqlite") as tools:
        for c in cases:
            if c["id"] in done:
                runs.append(done[c["id"]])
                continue
            counting = CountingLLM(llm)
            case_id = f"eval-{c['id'].lower()}"
            run = CaseRun(c["id"], "agent")
            await tools.tracker.call("create_case", owner_id="eval", policy_id=POLICY_ID, letter_text=c["letter"], case_id=case_id)
            async with open_runner(counting, tools, work_dir / f"{case_id}.sqlite", max_steps=max_steps, min_overlap=min_overlap) as runner:
                try:
                    await runner.start(case_id, "eval")
                    values = await runner.state_values(case_id)
                    verification = values["verification"]
                    run.pool = values["evidence"]
                    run.raw = verification["supported"] + verification["unsupported"]
                    run.shipped = verification["supported"]
                except LLMQuotaError as err:
                    raise QuotaStopped(str(err)) from None
                except (LLMError, ToolError) as err:
                    run.error = str(err)
            run.model_calls = counting.count
            if progress and not run.error:
                progress.add(run)
            runs.append(run)
    return runs


# ----------------------------------------------------------------------------- reporting
METRIC_ROWS = [
    ("Pool recall (expected passages retrieved)", "pool_recall"),
    ("Citation correctness (valid / proposed)", "citation_correctness"),
    ("Unsupported-claim rate, as proposed by the model", "unsupported_claim_rate_proposed"),
    ("Unsupported-claim rate, as shipped to the user", "unsupported_claim_rate_shipped"),
    ("Answer correctness, appeal cases", "answer_correctness_appeal"),
    ("Evidence coverage, appeal cases", "evidence_coverage_appeal"),
    ("Abstained correctly, no-appeal cases", "abstention_correct"),
    ("False abstention, appeal cases", "false_abstention_rate"),
    ("Model calls per case", "mean_model_calls"),
    ("Cases that failed with an error", "errors"),
]


def render_markdown(report: dict[str, Any]) -> str:
    systems = list(report["systems"])
    lines = [
        f"# Comparison: {' vs '.join(systems)}",
        "",
        f"Model: `{report['model']}`. Run: {report['run_at']}. Cases: {report['n_cases']} "
        f"({report['systems'][systems[0]]['appeal_cases']} appeal, {report['systems'][systems[0]]['abstain_cases']} no-appeal).",
        "",
        "| Metric | " + " | ".join(systems) + " |",
        "| --- | " + " | ".join("---" for _ in systems) + " |",
    ]
    for title, key in METRIC_ROWS:
        cells = []
        for s in systems:
            v = report["systems"][s][key]
            cells.append("n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)))
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    if any(report["systems"][s]["errors"] for s in systems):
        lines += ["", "> **Incomplete run.** Cases that errored are counted as failures here, so these numbers understate the "
                      "system. Rerun with `--resume` to retry only those cases before quoting any of this."]
    misses = miss_lines(report)
    if misses:
        lines += ["", "Cases that were not fully right:", ""] + [f"- {m}" for m in misses]
    lines += ["", report["caveat"], ""]
    return "\n".join(lines)


CAVEAT = (
    "The policy and the cases are synthetic and were written by the same person, the set is small, and one model on one run "
    "is one sample. Read these as a check that the system behaves as designed, not as a general accuracy figure."
)


async def compare(
    llm: LLM, model_name: str, systems: tuple[str, ...] = ("agent", "baseline"), cases: list[dict[str, Any]] | None = None,
    max_steps: int = 4, min_overlap: float = DEFAULT_MIN_OVERLAP, progress: Progress | None = None,
) -> dict[str, Any]:
    cases = cases if cases is not None else load_cases()
    chunks = ingest_pdf(DEFAULT_POLICY_PDF)
    out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        if "baseline" in systems:
            out["baseline"] = score(cases, await run_baseline_system(cases, llm, chunks, min_overlap=min_overlap, progress=progress))
        if "agent" in systems:
            out["agent"] = score(cases, await run_agent_system(cases, llm, Path(tmp), max_steps=max_steps, min_overlap=min_overlap, progress=progress))
    ordered = {s: out[s] for s in systems if s in out}
    return {
        "model": model_name,
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_cases": len(cases),
        "max_steps": max_steps,
        "systems": ordered,
        "caveat": CAVEAT,
    }


class ProgressLLM:
    """Wraps another LLM and prints one line per call, so a slow run shows that it is working."""

    def __init__(self, inner: LLM):
        self._inner = inner
        self._n = 0

    async def complete_json(self, stage: str, system: str, prompt: str) -> dict[str, Any]:
        self._n += 1
        n, started = self._n, time.monotonic()
        print(f"  call {n} ({stage}): waiting for the model...", file=sys.stderr, flush=True)
        try:
            result = await self._inner.complete_json(stage, system, prompt)
        except Exception as err:
            print(f"  call {n} failed after {time.monotonic() - started:.0f}s: {err}", file=sys.stderr, flush=True)
            raise
        print(f"  call {n} done in {time.monotonic() - started:.0f}s", file=sys.stderr, flush=True)
        return result


def miss_lines(report: dict[str, Any]) -> list[str]:
    """Why each case that ran but was not fully right was not fully right (errors are listed by error_lines)."""
    out = []
    for system, data in report["systems"].items():
        for row in data["per_case"]:
            if not row.get("correct") and not row.get("error") and row.get("why"):
                out.append(f"{system} {row['case']}: {row['why']}")
    return out


def error_lines(report: dict[str, Any]) -> list[str]:
    """One line per distinct failure, so a run where every case errored explains itself."""
    seen: dict[tuple[str, str], list[str]] = {}
    for system, data in report["systems"].items():
        for row in data["per_case"]:
            if row.get("error"):
                seen.setdefault((system, row["error"]), []).append(row["case"])
    return [f"{system}: {message} (cases: {', '.join(cases)})" for (system, message), cases in seen.items()]


def main() -> None:
    ensure_utf8_output()
    parser = argparse.ArgumentParser(description="Compare the agent with the non-agentic baseline using a real Gemini model.")
    parser.add_argument("--systems", nargs="+", default=["agent", "baseline"], choices=["agent", "baseline"])
    parser.add_argument("--cases", default=str(DEFAULT_CASES_FILE))
    parser.add_argument("--limit", type=int, default=None, help="only the first N cases (a cheap smoke test)")
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--min-interval", type=float, default=5.0, help="seconds between model calls (free-tier rate limits)")
    parser.add_argument("--out-dir", default=str(REPORT_DIR))
    parser.add_argument("--resume", action="store_true", help="reuse cases finished by an earlier run of this same command and retry the rest")
    args = parser.parse_args()

    base = GeminiLLM.from_env()
    llm = ProgressLLM(ThrottledLLM(base, args.min_interval))
    cases = load_cases(args.cases)[: args.limit]
    out_dir = Path(args.out_dir)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", base.model)
    key = hashlib.sha256(json.dumps([code_fingerprint(), base.model, args.max_steps, DEFAULT_MIN_OVERLAP, [(c["id"], c["letter"]) for c in cases]]).encode()).hexdigest()
    progress = Progress(out_dir / f"progress_{slug}.json", key, args.resume)
    print(f"Running {len(cases)} case(s) with {base.model}. Each model call is listed below.", file=sys.stderr, flush=True)
    if args.resume and progress.ignored:
        print("The saved progress was made with different code, model, settings or cases, so it was ignored "
              "and everything runs again.", file=sys.stderr, flush=True)
    elif args.resume:
        print(f"Resuming. Already finished: {progress.counts()}.", file=sys.stderr, flush=True)

    async def go() -> dict[str, Any]:
        try:
            return await compare(llm, base.model, tuple(args.systems), cases, args.max_steps, progress=progress)
        finally:
            await base.aclose()

    try:
        report = asyncio.run(go())
    except QuotaStopped as stop:
        print(f"\nStopped early: {stop}", file=sys.stderr)
        print(f"Finished cases are saved ({progress.counts()}).", file=sys.stderr)
        print("Run the same command again with --resume once the limit resets, or set a different GEMINI_MODEL "
              "(each model has its own limit) and start again.", file=sys.stderr)
        sys.exit(2)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "comparison_" + slug
    (out_dir / f"{stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / f"{stem}.md").write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    problems = error_lines(report)
    if problems:
        print("What went wrong:")
        for line in problems:
            print("  " + line)
        print("Run the same command again with --resume to retry only the cases that failed.")
        print()
    print(f"written to {out_dir / stem}.json and .md")


if __name__ == "__main__":
    main()
