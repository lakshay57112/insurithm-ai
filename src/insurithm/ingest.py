"""PDF ingestion: turn a policy PDF into page-numbered chunks with exact offsets.

Every chunk keeps its page number and its character offsets inside that
page's text, so any later citation can point at a real page and passage.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from pypdf import PdfReader

from .console import ensure_utf8_output
from .models import Chunk

# A clause starts with a number like "3.2" followed by text.
CLAUSE_RE = re.compile(r"^(\d+\.\d+)\s+\S")
# Headings such as "SECTION 3 - WAITING PERIODS" or "Annexure A - Day Care ...".
# The dash is required so a wrapped line that starts with "Annexure A are ..."
# is not mistaken for a heading.
HEADING_RE = re.compile(r"^(SECTION\s+\d+|Annexure\s+[A-Z])\s*[-:–]")
SENTENCE_BREAK_RE = re.compile(r"(?<=[.;])\s+")


def extract_pages(pdf_path: str | Path) -> list[str]:
    """Return the text of each page, in order (index 0 is page 1)."""
    reader = PdfReader(str(pdf_path))
    return [(page.extract_text() or "") for page in reader.pages]


def _split_span(page_text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    """Split a long span at sentence boundaries, keeping page offsets."""
    if end - start <= max_chars:
        return [(start, end)]
    spans: list[tuple[int, int]] = []
    cur = start
    last: tuple[int, int] | None = None
    for m in SENTENCE_BREAK_RE.finditer(page_text[start:end]):
        cut, nxt = start + m.start(), start + m.end()
        if last is not None and cut - cur > max_chars:
            spans.append((cur, last[0]))
            cur = last[1]
        last = (cut, nxt)
    spans.append((cur, end))
    return spans


def chunk_page(
    doc_id: str,
    page_no: int,
    text: str,
    max_chars: int = 900,
    section: str | None = None,
) -> tuple[list[Chunk], str | None]:
    """Split one page into clause-level chunks.

    Returns the chunks and the last section heading seen, so the next page
    can continue under the same section.
    """
    blocks: list[dict] = []
    cur: dict | None = None
    offset = 0
    current_section = section

    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        stripped = content.strip()
        line_start = offset
        offset += len(line)
        if not stripped:
            continue
        lead = len(content) - len(content.lstrip())
        span_start = line_start + lead
        span_end = line_start + len(content.rstrip())

        if HEADING_RE.match(stripped):
            if cur:
                blocks.append(cur)
                cur = None
            current_section = stripped
            continue

        clause = CLAUSE_RE.match(stripped)
        if clause or cur is None:
            if cur:
                blocks.append(cur)
            cur = {
                "start": span_start,
                "end": span_end,
                "clause": clause.group(1) if clause else None,
                "section": current_section,
            }
        else:
            cur["end"] = span_end
    if cur:
        blocks.append(cur)

    chunks: list[Chunk] = []
    index = 0
    for block in blocks:
        for start, end in _split_span(text, block["start"], block["end"], max_chars):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}-p{page_no}-c{index}",
                    doc_id=doc_id,
                    page=page_no,
                    section=block["section"],
                    clause_id=block["clause"],
                    text=text[start:end],
                    start_char=start,
                    end_char=end,
                )
            )
            index += 1
    return chunks, current_section


def ingest_pdf(pdf_path: str | Path, doc_id: str | None = None, max_chars: int = 900) -> list[Chunk]:
    """Read a PDF and return page-numbered chunks for the whole document."""
    pdf_path = Path(pdf_path)
    doc_id = doc_id or pdf_path.stem
    chunks: list[Chunk] = []
    section: str | None = None
    for page_no, text in enumerate(extract_pages(pdf_path), start=1):
        page_chunks, section = chunk_page(doc_id, page_no, text, max_chars, section)
        chunks.extend(page_chunks)
    return chunks


def write_chunks(chunks: list[Chunk], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")


def read_chunks(path: str | Path) -> list[Chunk]:
    with Path(path).open(encoding="utf-8") as f:
        return [Chunk.model_validate(json.loads(line)) for line in f if line.strip()]


def main() -> None:
    ensure_utf8_output()
    parser = argparse.ArgumentParser(description="Split a policy PDF into page-numbered chunks.")
    parser.add_argument("pdf", help="Path to the policy PDF")
    parser.add_argument("--out", help="Write chunks to this JSONL file")
    parser.add_argument("--max-chars", type=int, default=900)
    args = parser.parse_args()

    chunks = ingest_pdf(args.pdf, max_chars=args.max_chars)
    pages = len({c.page for c in chunks})
    print(f"{len(chunks)} chunks from {pages} pages")
    if args.out:
        write_chunks(chunks, args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
