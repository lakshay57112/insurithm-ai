from insurithm.config import DEFAULT_POLICY_PDF
from insurithm.ingest import chunk_page, extract_pages, ingest_pdf, read_chunks, write_chunks


def test_pdf_has_expected_pages():
    assert len(extract_pages(DEFAULT_POLICY_PDF)) == 8


def test_every_chunk_matches_its_page_text_exactly():
    pages = extract_pages(DEFAULT_POLICY_PDF)
    for chunk in ingest_pdf(DEFAULT_POLICY_PDF):
        page_text = pages[chunk.page - 1]
        assert page_text[chunk.start_char : chunk.end_char] == chunk.text
        assert chunk.start_char < chunk.end_char


def test_chunk_ids_are_unique():
    chunks = ingest_pdf(DEFAULT_POLICY_PDF)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_clauses_are_split_one_per_chunk():
    chunks = ingest_pdf(DEFAULT_POLICY_PDF)
    clause_ids = [c.clause_id for c in chunks if c.clause_id]
    # Every numbered clause appears exactly once.
    assert len(clause_ids) == len(set(clause_ids))
    assert {"1.1", "2.4", "4.2", "6.7", "8.2"} <= set(clause_ids)


def test_clause_text_starts_with_its_own_number():
    for chunk in ingest_pdf(DEFAULT_POLICY_PDF):
        if chunk.clause_id:
            assert chunk.text.startswith(chunk.clause_id)


def test_section_heading_is_attached_to_clauses():
    by_clause = {c.clause_id: c for c in ingest_pdf(DEFAULT_POLICY_PDF) if c.clause_id}
    assert by_clause["3.2"].section.startswith("SECTION 3")
    assert by_clause["8.1"].section.startswith("SECTION 8")
    assert by_clause["8.1"].page == 7


def test_wrapped_annexure_reference_is_not_a_heading():
    # Clause 1.1 wraps onto a line that can start with "Annexure A"; it must stay in the clause.
    by_clause = {c.clause_id: c for c in ingest_pdf(DEFAULT_POLICY_PDF) if c.clause_id}
    assert "Annexure A" in by_clause["1.1"].text


def test_section_carries_across_pages():
    text = "3.8 A clause that continues on the next page."
    chunks, section = chunk_page("doc", 2, text, section="SECTION 3 - WAITING PERIODS")
    assert chunks[0].section == "SECTION 3 - WAITING PERIODS"
    assert section == "SECTION 3 - WAITING PERIODS"


def test_long_clause_is_split_with_valid_offsets():
    sentence = "This sentence is padding for the splitter. "
    text = "9.1 " + sentence * 40
    chunks, _ = chunk_page("doc", 1, text, max_chars=200)
    assert len(chunks) > 1
    for c in chunks:
        assert text[c.start_char : c.end_char] == c.text
        assert len(c.text) <= 260  # one sentence of slack


def test_chunks_round_trip_through_jsonl(tmp_path):
    chunks = ingest_pdf(DEFAULT_POLICY_PDF)
    out = tmp_path / "chunks.jsonl"
    write_chunks(chunks, out)
    assert read_chunks(out) == chunks
