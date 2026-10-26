"""The gold set must be consistent with the policy before any score means anything."""

from insurithm.config import DEFAULT_GOLD_FILE, DEFAULT_POLICY_PDF
from insurithm.evals.retrieval_eval import load_gold, normalize
from insurithm.ingest import extract_pages, ingest_pdf


def test_gold_set_size_and_mix():
    gold = load_gold(DEFAULT_GOLD_FILE)
    answerable = [q for q in gold if q["answerable"]]
    assert len(answerable) >= 30
    assert len(gold) - len(answerable) >= 5  # unanswerable questions for later tests


def test_ids_are_unique():
    ids = [q["id"] for q in load_gold(DEFAULT_GOLD_FILE)]
    assert len(ids) == len(set(ids))


def test_expected_passage_is_on_expected_page():
    pages = extract_pages(DEFAULT_POLICY_PDF)
    for q in load_gold(DEFAULT_GOLD_FILE):
        if not q["answerable"]:
            continue
        page_text = normalize(pages[q["expected_page"] - 1])
        assert normalize(q["expected_passage"]) in page_text, q["id"]


def test_expected_passage_lies_inside_a_single_chunk():
    chunks = ingest_pdf(DEFAULT_POLICY_PDF)
    for q in load_gold(DEFAULT_GOLD_FILE):
        if not q["answerable"]:
            continue
        found = [
            c
            for c in chunks
            if c.page == q["expected_page"] and normalize(q["expected_passage"]) in normalize(c.text)
        ]
        assert found, f"{q['id']}: passage not inside any single chunk"


def test_expected_clause_matches_the_chunk():
    chunks = ingest_pdf(DEFAULT_POLICY_PDF)
    for q in load_gold(DEFAULT_GOLD_FILE):
        if not q["answerable"]:
            continue
        found = [
            c
            for c in chunks
            if c.page == q["expected_page"] and normalize(q["expected_passage"]) in normalize(c.text)
        ]
        assert any(c.clause_id == q["expected_clause"] for c in found), q["id"]


def test_unanswerable_questions_have_no_expected_evidence():
    for q in load_gold(DEFAULT_GOLD_FILE):
        if not q["answerable"]:
            assert "expected_passage" not in q and "expected_page" not in q
