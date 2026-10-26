"""The appeal cases must agree with the policy and with the fact extractor before any score means anything."""

from insurithm.config import DEFAULT_POLICY_PDF
from insurithm.evals.compare import load_cases, normalize
from insurithm.facts import extract_facts
from insurithm.ingest import ingest_pdf

CASES = load_cases()
CHUNKS = ingest_pdf(DEFAULT_POLICY_PDF)


def test_size_and_mix():
    appeal = [c for c in CASES if c["expected_outcome"] == "appeal"]
    other = [c for c in CASES if c["expected_outcome"] == "insufficient_evidence"]
    assert len(appeal) >= 10 and len(other) >= 5 and len(appeal) + len(other) == len(CASES)


def test_ids_are_unique_and_letters_are_long_enough():
    assert len({c["id"] for c in CASES}) == len(CASES)
    assert all(len(c["letter"]) >= 100 for c in CASES)


def test_every_expected_passage_sits_inside_one_chunk_on_the_right_page():
    for c in CASES:
        for e in c["expected"]:
            found = [ch for ch in CHUNKS if ch.page == e["page"] and normalize(e["passage"]) in normalize(ch.text)]
            assert found, f"{c['id']}: passage not in any single chunk on page {e['page']}"
            if e["clause"]:
                assert any(ch.clause_id == e["clause"] for ch in found), f"{c['id']}: wrong clause {e['clause']}"


def test_no_appeal_cases_expect_no_evidence():
    for c in CASES:
        assert (c["expected"] == []) == (c["expected_outcome"] != "appeal")


def test_letters_parse_cleanly():
    for c in CASES:
        facts = extract_facts(c["letter"])
        assert facts.letter_date is not None, c["id"]  # every letter carries a "Date:" line
        for f in facts.facts:
            assert c["letter"][f.start : f.end] == f.text
    # The clause each rejection cites is picked up, so intake can fetch it.
    assert extract_facts(next(c for c in CASES if c["id"] == "C01")["letter"]).clause_refs == ["4.2"]
    assert extract_facts(next(c for c in CASES if c["id"] == "C05")["letter"]).clause_refs == []


def test_every_policy_clause_in_a_letter_exists_or_is_deliberately_missing():
    clause_ids = {ch.clause_id for ch in CHUNKS if ch.clause_id}
    for c in CASES:
        for ref in extract_facts(c["letter"]).clause_refs:
            assert ref in clause_ids, f"{c['id']} cites clause {ref}, which the policy does not have"
