import pytest

from insurithm.config import DEFAULT_POLICY_PDF
from insurithm.ingest import extract_pages, ingest_pdf
from insurithm.verify import check_citation, verify_finding, verify_findings


@pytest.fixture(scope="module")
def pool():
    return {c.chunk_id: c for c in ingest_pdf(DEFAULT_POLICY_PDF)}


@pytest.fixture(scope="module")
def pages():
    return extract_pages(DEFAULT_POLICY_PDF)


def by_clause(pool, clause):
    return next(c for c in pool.values() if c.clause_id == clause)


def test_exact_quote_is_valid_and_points_at_the_page_text(pool, pages):
    chunk = by_clause(pool, "6.2")
    cit = check_citation(chunk.chunk_id, "the claims desk must be informed within 24 hours of\nadmission", pool)
    assert cit.valid
    assert cit.page == 6 and cit.clause_id == "6.2"
    # The offsets locate the quote in the original page, and the stored quote is the source wording.
    assert pages[cit.page - 1][cit.start_char : cit.end_char] == cit.quote


def test_whitespace_differences_are_tolerated_but_wording_is_not(pool):
    chunk = by_clause(pool, "6.2")
    flat = check_citation(chunk.chunk_id, "informed within 24 hours of admission", pool)
    assert flat.valid
    assert "\n" in flat.quote  # canonicalised back to the source text
    reworded = check_citation(chunk.chunk_id, "informed within 48 hours of admission", pool)
    assert not reworded.valid and reworded.reason == "quote_not_in_chunk"


def test_case_must_match(pool):
    chunk = by_clause(pool, "6.2")
    assert not check_citation(chunk.chunk_id, "THE CLAIMS DESK MUST BE INFORMED", pool).valid


def test_unknown_chunk_and_bad_types_never_raise(pool):
    assert check_citation("nope", "a long enough quote here", pool).reason == "unknown_chunk"
    assert check_citation(None, "a long enough quote here", pool).reason == "unknown_chunk"
    chunk = by_clause(pool, "6.2")
    assert check_citation(chunk.chunk_id, None, pool).reason == "quote_too_short"
    assert check_citation(chunk.chunk_id, "within 24", pool).reason == "quote_too_short"


def test_regex_characters_in_quote_are_literal(pool):
    chunk = by_clause(pool, "2.6")
    assert check_citation(chunk.chunk_id, "Rs. 2,000 per hospitalisation", pool).valid
    assert not check_citation(chunk.chunk_id, "Rs.* 2,000 per hospitalisation", pool).valid


def finding(pool, clause, claim, quote):
    return {"claim": claim, "citations": [{"chunk_id": by_clause(pool, clause).chunk_id, "quote": quote}]}


def test_supported_finding(pool):
    f = finding(pool, "6.6", "The insurer must settle or reject a claim within 30 days of the last document.",
                "We must settle or reject a claim within 30 days of receiving the last necessary\ndocument.")
    checked = verify_finding(f, pool)
    assert checked.supported and checked.reason is None
    assert checked.valid_citations[0].page == 6


def test_finding_without_citations_is_unsupported(pool):
    assert verify_finding({"claim": "Anything at all."}, pool).reason == "no_citations"
    assert verify_finding({"claim": "Anything.", "citations": []}, pool).reason == "no_citations"


def test_finding_with_only_invented_quote_is_unsupported(pool):
    f = finding(pool, "6.6", "The insurer must pay within 7 days.", "We must pay every claim within 7 days of admission")
    checked = verify_finding(f, pool)
    assert not checked.supported and checked.reason == "no_valid_citation"


def test_valid_quote_that_has_nothing_to_do_with_the_claim_is_dropped(pool):
    f = finding(pool, "2.6", "Maternity expenses are limited to fifty thousand rupees per delivery.",
                "road ambulance charges to take the\ninsured person to the hospital")
    checked = verify_finding(f, pool)
    assert not checked.supported and checked.reason == "claim_not_supported_by_quote"


def test_one_valid_citation_is_enough_even_if_another_is_bad(pool):
    f = finding(pool, "6.6", "The insurer must settle or reject a claim within 30 days.",
                "settle or reject a claim within 30 days")
    f["citations"].append({"chunk_id": "ghost", "quote": "this quote is made up entirely"})
    checked = verify_finding(f, pool)
    assert checked.supported
    assert [c.valid for c in checked.citations] == [True, False]


def test_malformed_findings_are_reported_not_raised(pool):
    result = verify_findings(["just a string", {"citations": []}, {"claim": 5}, None], pool)
    assert result.supported == []
    assert [f.reason for f in result.unsupported] == [
        "malformed_finding", "missing_claim", "missing_claim", "malformed_finding"
    ]
    assert verify_findings("not a list", pool).outcome == "insufficient_evidence"


def test_outcome_is_insufficient_evidence_when_nothing_survives(pool):
    bad = finding(pool, "6.6", "Invented.", "completely invented wording here")
    assert verify_findings([bad], pool).outcome == "insufficient_evidence"
    good = finding(pool, "6.6", "Settle or reject within 30 days.", "settle or reject a claim within 30 days")
    result = verify_findings([bad, good], pool)
    assert result.outcome == "draft_ready"
    assert len(result.supported) == 1 and len(result.unsupported) == 1


def test_citations_to_chunks_outside_the_pool_are_rejected(pool):
    """Only evidence the agent actually retrieved can be cited."""
    chunk = by_clause(pool, "6.6")
    small_pool = {k: v for k, v in pool.items() if k != chunk.chunk_id}
    cit = check_citation(chunk.chunk_id, "settle or reject a claim within 30 days", small_pool)
    assert not cit.valid and cit.reason == "unknown_chunk"
