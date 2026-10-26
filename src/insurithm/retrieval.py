"""Retrieval over policy chunks.

Week 1 ships a keyword (BM25) retriever. It is the baseline every later
retriever, and the agent, will be measured against. New retrievers only need
a `search(query, k)` method that returns ranked `Hit` objects.
"""

from __future__ import annotations

import re
from typing import Protocol

from rank_bm25 import BM25Okapi

from .models import Chunk, Hit

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have
    how i if in into is it its me my of on or our so than that the their them then
    there these they this to up was we were what when where which who why will with
    would you your""".split()
)
_SUFFIXES = ("ations", "ation", "ing", "ed", "es", "s")


def stem(token: str) -> str:
    """Very light suffix stripping so 'claims' matches 'claim'."""
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [stem(w) for w in words if w not in STOPWORDS]


class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[Hit]: ...


class BM25Retriever:
    """Keyword retriever. Searches each chunk's section heading plus its text."""

    name = "bm25"

    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("BM25Retriever needs at least one chunk")
        self.chunks = chunks
        corpus = [tokenize(f"{c.section or ''} {c.text}") for c in chunks]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            Hit(chunk=self.chunks[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(order, start=1)
        ]


def main() -> None:
    """Search a policy from the command line:  python -m insurithm.retrieval "emergency admission notice" -k 3"""
    import argparse

    from .config import DEFAULT_POLICY_PDF
    from .console import ensure_utf8_output
    from .ingest import ingest_pdf

    ensure_utf8_output()
    parser = argparse.ArgumentParser(description="Search policy document clauses using the BM25 retriever.")
    parser.add_argument("query", help="Insurance question or search term")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY_PDF), help="Path to the policy PDF")
    parser.add_argument("-k", type=int, default=3, help="Number of results to return")
    args = parser.parse_args()

    hits = BM25Retriever(ingest_pdf(args.policy)).search(args.query, k=args.k)
    print(f"\nSearch results for: \"{args.query}\" (top {len(hits)} hits)\n" + "-" * 60)
    for hit in hits:
        c = hit.chunk
        clause = f" [Clause {c.clause_id}]" if c.clause_id else ""
        section = f" ({c.section})" if c.section else ""
        print(f"Rank {hit.rank} (Score: {hit.score:.2f}) | Page {c.page}{clause}{section}")
        print(f"Text: {c.text}\n")


if __name__ == "__main__":
    main()
