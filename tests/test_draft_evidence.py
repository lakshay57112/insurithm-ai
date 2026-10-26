"""The draft step must show the model the evidence it collected. An earlier version kept only the first 24
chunks sorted by page, which silently dropped the appeal and time-limit clauses at the back of the policy."""

from insurithm.agent.prompts import MAX_EVIDENCE_IN_PROMPT, chunks_for_draft, draft_prompt


def chunk(n, page, doc="policy"):
    return {"chunk_id": f"{doc}-c{n:03d}", "doc_id": doc, "page": page, "clause_id": f"{page}.{n}", "text": f"text of chunk {n}"}


def pool(count, pages=8):
    letter = chunk(0, 1, doc="letter-case1")
    rest = [chunk(n, 1 + (n - 1) * pages // count) for n in range(1, count + 1)]
    return {c["chunk_id"]: c for c in [letter, *rest]}


def test_a_pool_bigger_than_the_old_cap_still_reaches_the_model_in_full():
    evidence = pool(36)                                   # the old cap was 24
    prompt = draft_prompt("letter", {"facts": []}, evidence, [])
    for cid in evidence:
        assert f"[{cid}]" in prompt
    assert chunks_for_draft(evidence)[1] == 0


def test_the_last_page_is_not_the_first_thing_to_go():
    evidence = pool(36)
    last_page = max(c["page"] for c in evidence.values())
    prompt = draft_prompt("letter", {"facts": []}, evidence, [])
    assert any(f"page {last_page}" in line for line in prompt.splitlines() if line.startswith("["))


def test_if_something_must_go_it_is_the_last_found_and_the_count_is_reported():
    evidence = pool(MAX_EVIDENCE_IN_PROMPT + 15)
    shown, left_out = chunks_for_draft(evidence)
    assert len(shown) == MAX_EVIDENCE_IN_PROMPT and left_out == 16   # 15 extra plus the letter chunk
    found = list(evidence)
    kept_ids = {c["chunk_id"] for c in shown}
    assert kept_ids == set(found[:MAX_EVIDENCE_IN_PROMPT])           # the first found, not the lowest pages
    assert found[0] in kept_ids                                      # the letter is always kept


def test_what_is_shown_is_ordered_letter_first_then_by_page():
    shown, _ = chunks_for_draft(pool(20))
    assert shown[0]["doc_id"].startswith("letter-")
    pages = [c["page"] for c in shown[1:]]
    assert pages == sorted(pages)
