"""Build small PDFs for tests (needs reportlab, a dev dependency)."""

import io

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

POLICY_LINES = [
    "SECTION 1 - COVER",
    "1.1 Hospitalisation: The company pays the reasonable cost of hospital treatment for an insured person",
    "when admitted as an in-patient for more than 24 hours on the advice of a doctor.",
    "1.2 Day care: Listed day care procedures are payable even when the stay is shorter than 24 hours.",
    "SECTION 2 - WAITING PERIODS",
    "2.1 Pre-existing diseases: Expenses for pre-existing diseases are payable only after 24 months of",
    "continuous coverage under this policy. The company will not reduce the waiting period on request.",
    "2.2 Accidents: Treatment after an accident is payable from the first day of cover without a waiting period.",
]


def make_pdf(pages: list[list[str]]) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    for lines in pages:
        y = 800
        for line in lines:
            pdf.drawString(50, y, line)
            y -= 18
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def policy_pdf() -> bytes:
    return make_pdf([POLICY_LINES])


def blank_pdf() -> bytes:
    """A PDF whose page has no text at all, like a scan."""
    return make_pdf([[]])
