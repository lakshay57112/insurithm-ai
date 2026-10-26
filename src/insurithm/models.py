"""Data models shared across ingestion, retrieval and evaluation."""

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A passage of a policy document, tied to an exact place in the source.

    `start_char` and `end_char` are offsets into the text of `page`, so
    `page_text[start_char:end_char] == text` always holds. That is what lets
    every later citation point at a real page and passage.
    """

    chunk_id: str
    doc_id: str
    page: int = Field(ge=1, description="1-based page number in the source PDF")
    section: str | None = None
    clause_id: str | None = None
    text: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)


class Hit(BaseModel):
    """One retrieval result."""

    chunk: Chunk
    score: float
    rank: int = Field(ge=1)
