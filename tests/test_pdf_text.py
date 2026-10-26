import pytest

from insurithm.pdf_text import PdfProblem, extract_pdf_pages
from pdfs import blank_pdf, make_pdf, policy_pdf


async def test_reads_text_page_by_page():
    pages = await extract_pdf_pages(make_pdf([["first page text"], ["second page text"]]), max_pages=5)
    assert len(pages) == 2 and "first page text" in pages[0] and "second page text" in pages[1]


async def test_a_page_with_no_text_comes_back_empty():
    assert [p.strip() for p in await extract_pdf_pages(blank_pdf(), max_pages=5)] == [""]


async def test_too_many_pages_is_refused():
    with pytest.raises(PdfProblem) as err:
        await extract_pdf_pages(make_pdf([["a"], ["b"], ["c"]]), max_pages=2)
    assert err.value.code == "too_many_pages" and "limit" in err.value.message


async def test_damaged_files_are_reported_not_raised_as_crashes():
    with pytest.raises(PdfProblem) as err:
        await extract_pdf_pages(b"%PDF-1.4\nthis is not really a pdf", max_pages=5)
    assert err.value.code == "unreadable"


async def test_a_slow_read_is_stopped():
    with pytest.raises(PdfProblem) as err:
        await extract_pdf_pages(policy_pdf(), max_pages=5, timeout=0.01)  # a new Python process cannot start that fast
    assert err.value.code == "too_slow"
