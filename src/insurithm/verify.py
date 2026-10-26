"""Deterministic citation checking.

The language model proposes findings and the passages that support them. Nothing
it proposes is trusted until this module has checked it:

* the cited chunk must be one the agent actually retrieved (the *evidence pool*),
* the quote must appear verbatim in that chunk (whitespace may differ, wording may not),
* at least one valid citation must be to the policy: the rejection letter is in
  the evidence pool so that facts can be read from it, but a point that rests
  only on the insurer's own letter has no policy behind it,
* the claim must share enough vocabulary with its valid quotes to rule out an
  obviously unrelated citation.

A finding that fails any of these is dropped from the output and reported as
unsupported. If nothing survives, the system answers "insufficient evidence".

What this does NOT do: prove that a quote *entails* a claim. The overlap test is
a cheap sanity check, so the human review step still matters, and the rendered
appeal always prints the quote next to the claim so a reviewer can judge it.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .models import Chunk
from .retrieval import tokenize

MIN_QUOTE_CHARS = 12
MAX_CLAIM_CHARS = 600
DEFAULT_MIN_OVERLAP = 0.2
LETTER_DOC_PREFIX = "letter-"  # chunks cut from the rejection letter, as opposed to the policy


@dataclass(frozen=True)
class CheckedCitation:
    chunk_id: str
    quote: str
    valid: bool
    reason: str | None = None
    doc_id: str | None = None
    page: int | None = None
    clause_id: str | None = None
    section: str | None = None
    # Offsets into the text of `page`, so the quote can be located in the source PDF.
    start_char: int | None = None
    end_char: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CheckedFinding:
    claim: str
    supported: bool
    citations: tuple[CheckedCitation, ...] = ()
    reason: str | None = None
    overlap: float = 0.0

    @property
    def valid_citations(self) -> list[CheckedCitation]:
        return [c for c in self.citations if c.valid]

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "supported": self.supported,
            "reason": self.reason,
            "overlap": round(self.overlap, 3),
            "citations": [c.to_dict() for c in self.citations],
        }


@dataclass
class VerificationResult:
    supported: list[CheckedFinding] = field(default_factory=list)
    unsupported: list[CheckedFinding] = field(default_factory=list)

    @property
    def outcome(self) -> str:
        return "draft_ready" if self.supported else "insufficient_evidence"

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "supported": [f.to_dict() for f in self.supported],
            "unsupported": [f.to_dict() for f in self.unsupported],
        }


def _quote_pattern(quote: str) -> re.Pattern[str]:
    """Match the quote token by token, letting any run of whitespace stand for a line break."""
    tokens = quote.split()
    return re.compile(r"\s+".join(re.escape(t) for t in tokens))


def check_citation(chunk_id: Any, quote: Any, pool: Mapping[str, Chunk]) -> CheckedCitation:
    """Check one citation against the evidence pool. Never raises on bad model output."""
    cid = chunk_id if isinstance(chunk_id, str) else repr(chunk_id)
    q = quote if isinstance(quote, str) else ""

    if not isinstance(chunk_id, str) or chunk_id not in pool:
        return CheckedCitation(cid, q, False, "unknown_chunk")
    if len(" ".join(q.split())) < MIN_QUOTE_CHARS:
        return CheckedCitation(cid, q, False, "quote_too_short")

    chunk = pool[chunk_id]
    match = _quote_pattern(q).search(chunk.text)
    if match is None:
        return CheckedCitation(cid, q, False, "quote_not_in_chunk")

    return CheckedCitation(
        chunk_id=cid,
        quote=chunk.text[match.start() : match.end()],  # canonical wording from the source
        valid=True,
        doc_id=chunk.doc_id,
        page=chunk.page,
        clause_id=chunk.clause_id,
        section=chunk.section,
        start_char=chunk.start_char + match.start(),
        end_char=chunk.start_char + match.end(),
    )


def _overlap(claim: str, quotes: list[str]) -> float:
    claim_tokens = set(tokenize(claim))
    if not claim_tokens:
        return 0.0
    quote_tokens = set(tokenize(" ".join(quotes)))
    return len(claim_tokens & quote_tokens) / len(claim_tokens)


def verify_finding(
    finding: Any, pool: Mapping[str, Chunk], min_overlap: float = DEFAULT_MIN_OVERLAP
) -> CheckedFinding:
    if not isinstance(finding, dict):
        return CheckedFinding(claim=repr(finding)[:MAX_CLAIM_CHARS], supported=False, reason="malformed_finding")

    claim = finding.get("claim")
    if not isinstance(claim, str) or not claim.strip():
        return CheckedFinding(claim="", supported=False, reason="missing_claim")
    claim = " ".join(claim.split())[:MAX_CLAIM_CHARS]

    raw = finding.get("citations")
    if not isinstance(raw, list) or not raw:
        return CheckedFinding(claim=claim, supported=False, reason="no_citations")

    checked = tuple(
        check_citation(c.get("chunk_id") if isinstance(c, dict) else None,
                       c.get("quote") if isinstance(c, dict) else None,
                       pool)
        for c in raw
    )
    valid = [c for c in checked if c.valid]
    if not valid:
        return CheckedFinding(claim=claim, supported=False, citations=checked, reason="no_valid_citation")

    if all(str(c.doc_id).startswith(LETTER_DOC_PREFIX) for c in valid):
        return CheckedFinding(claim=claim, supported=False, citations=checked, reason="cites_only_the_letter")

    overlap = _overlap(claim, [c.quote for c in valid])
    if overlap < min_overlap:
        return CheckedFinding(
            claim=claim,
            supported=False,
            citations=checked,
            reason="claim_not_supported_by_quote",
            overlap=overlap,
        )
    return CheckedFinding(claim=claim, supported=True, citations=checked, overlap=overlap)


def verify_findings(
    findings: Any, pool: Mapping[str, Chunk], min_overlap: float = DEFAULT_MIN_OVERLAP
) -> VerificationResult:
    """Split proposed findings into supported and unsupported ones."""
    result = VerificationResult()
    if not isinstance(findings, list):
        return result
    for f in findings:
        checked = verify_finding(f, pool, min_overlap)
        (result.supported if checked.supported else result.unsupported).append(checked)
    return result
