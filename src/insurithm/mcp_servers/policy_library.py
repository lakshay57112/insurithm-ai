"""policy-library MCP server.

Read-only access to policy wordings. It can search clauses, fetch one clause or
chunk, and read a page. It has no write tools at all.

Configuration (environment variables, all optional):

* ``INSURITHM_POLICY_PDFS``      policy PDFs to load, separated by the OS path separator
                                 (default: the synthetic demo policy)
* ``INSURITHM_ALLOWED_POLICIES`` comma-separated policy ids this server may serve
                                 (default: every policy it loaded). Anything else is refused
                                 with ``access_denied``.
* ``INSURITHM_POLICY_DIR``       a folder of uploaded policy PDFs. It is re-read on every call, so a
                                 PDF dropped in becomes searchable at once and one removed disappears.
                                 Files that cannot be read are skipped. The allow-list above does not
                                 apply to this folder: who may use an uploaded policy is decided by
                                 whoever put the file there (the API keeps each one private to its uploader).

Every tool returns ``{"ok": true, ...}`` or ``{"ok": false, "error": {"code", "message"}}``.
Malformed arguments are rejected by the tool's JSON schema before any code runs.

Run it on its own:  python -m insurithm.mcp_servers.policy_library
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from ..config import DEFAULT_POLICY_PDF
from ..ingest import extract_pages, ingest_pdf
from ..models import Chunk
from ..retrieval import BM25Retriever

POLICY_ID = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.\-]{1,64}$", description="Policy id, e.g. the PDF file stem")]
CLAUSE_ID = Annotated[str, Field(pattern=r"^\d{1,3}(\.\d{1,3}){0,3}$", description="Clause number such as 4.2")]
CHUNK_ID = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.\-]{1,100}$")]
QUERY = Annotated[str, Field(min_length=2, max_length=500)]
TOP_K = Annotated[int, Field(ge=1, le=10)]


def _error(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message}}


def _chunk_dict(chunk: Chunk, **extra: Any) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "policy_id": chunk.doc_id,
        "page": chunk.page,
        "section": chunk.section,
        "clause_id": chunk.clause_id,
        "text": chunk.text,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
        **extra,
    }


class PolicyLibrary:
    """Loads policies and answers lookups. Plain Python, so it can be tested without MCP."""

    def __init__(self, pdf_paths: list[Path], allowed: set[str] | None = None, upload_dir: Path | None = None):
        if not pdf_paths:
            raise ValueError("PolicyLibrary needs at least one policy PDF")
        self.chunks: dict[str, list[Chunk]] = {}
        self.pages: dict[str, list[str]] = {}
        self._retrievers: dict[str, BM25Retriever] = {}
        for path in pdf_paths:
            self._load(path)
        self.allowed = set(self.chunks) if allowed is None else set(allowed)
        self.upload_dir = Path(upload_dir) if upload_dir else None
        self._uploaded: set[str] = set()
        self._unreadable: dict[str, int] = {}  # policy id -> file mtime of a version that failed to load
        self._lock = threading.Lock()

    def _load(self, path: Path) -> str:
        chunks = ingest_pdf(path)
        if not chunks:
            raise ValueError(f"{path.name}: no readable text")
        policy_id = chunks[0].doc_id
        self.chunks[policy_id] = chunks
        self.pages[policy_id] = extract_pages(path)
        self._retrievers[policy_id] = BM25Retriever(chunks)
        return policy_id

    def refresh(self) -> None:
        """Pick up PDFs added to (or removed from) the uploads folder."""
        if self.upload_dir is None:
            return
        with self._lock:
            present = {p.stem: p for p in self.upload_dir.glob("*.pdf") if not p.name.startswith(".")}
            for policy_id in self._uploaded - set(present):
                for table in (self.chunks, self.pages, self._retrievers):
                    table.pop(policy_id, None)
                self._uploaded.discard(policy_id)
            for policy_id, path in present.items():
                if policy_id in self.chunks:  # already loaded, or a built-in policy with the same name
                    continue
                try:
                    mtime = path.stat().st_mtime_ns
                except OSError:
                    continue
                if self._unreadable.get(policy_id) == mtime:
                    continue
                try:
                    if self._load(path) != policy_id:
                        raise ValueError("policy id does not match the file name")
                    self._uploaded.add(policy_id)
                except Exception as exc:  # a bad upload must never break lookups for everyone else
                    self._unreadable[policy_id] = mtime
                    for table in (self.chunks, self.pages, self._retrievers):
                        table.pop(policy_id, None)
                    print(f"policy-library: skipped {path.name}: {exc!r}", file=sys.stderr)

    def _servable(self) -> set[str]:
        return (self.allowed | self._uploaded) & set(self.chunks)

    @classmethod
    def from_env(cls) -> "PolicyLibrary":
        raw = os.environ.get("INSURITHM_POLICY_PDFS", "")
        paths = [Path(p) for p in raw.split(os.pathsep) if p.strip()] or [DEFAULT_POLICY_PDF]
        allowed_raw = os.environ.get("INSURITHM_ALLOWED_POLICIES", "")
        allowed = {p.strip() for p in allowed_raw.split(",") if p.strip()} or None
        upload_dir = os.environ.get("INSURITHM_POLICY_DIR", "").strip() or None
        return cls(paths, allowed, Path(upload_dir) if upload_dir else None)

    # -- access control -------------------------------------------------
    def _check_policy(self, policy_id: str) -> dict[str, Any] | None:
        self.refresh()
        if policy_id not in self.chunks:
            return _error("unknown_policy", f"No policy with id {policy_id!r} is loaded.")
        if policy_id not in self._servable():
            return _error("access_denied", f"This server is not allowed to serve policy {policy_id!r}.")
        return None

    # -- tools ------------------------------------------------------------
    def list_policies(self) -> dict[str, Any]:
        self.refresh()
        return {
            "ok": True,
            "policies": [
                {"policy_id": pid, "pages": len(self.pages[pid]), "chunks": len(self.chunks[pid])}
                for pid in sorted(self._servable())
            ],
        }

    def search(self, query: str, k: int = 5, policy_id: str | None = None) -> dict[str, Any]:
        self.refresh()
        if policy_id is None:
            servable = sorted(self._servable())
            if len(servable) != 1:
                return _error("policy_id_required", "More than one policy is available; pass policy_id.")
            policy_id = servable[0]
        if (problem := self._check_policy(policy_id)) is not None:
            return problem
        hits = self._retrievers[policy_id].search(query, k=k)
        return {
            "ok": True,
            "policy_id": policy_id,
            "results": [_chunk_dict(h.chunk, score=round(h.score, 4), rank=h.rank) for h in hits],
        }

    def get_clause(self, policy_id: str, clause_id: str) -> dict[str, Any]:
        if (problem := self._check_policy(policy_id)) is not None:
            return problem
        found = [c for c in self.chunks[policy_id] if c.clause_id == clause_id]
        if not found:
            return _error("not_found", f"Policy {policy_id!r} has no clause {clause_id!r}.")
        return {"ok": True, "policy_id": policy_id, "results": [_chunk_dict(c) for c in found]}

    def get_chunk(self, chunk_id: str) -> dict[str, Any]:
        self.refresh()
        for policy_id, chunks in self.chunks.items():
            for chunk in chunks:
                if chunk.chunk_id == chunk_id:
                    if (problem := self._check_policy(policy_id)) is not None:
                        return problem
                    return {"ok": True, "chunk": _chunk_dict(chunk)}
        return _error("not_found", f"No chunk with id {chunk_id!r}.")

    def read_page(self, policy_id: str, page: int) -> str:
        if (problem := self._check_policy(policy_id)) is not None:
            raise ValueError(problem["error"]["message"])
        pages = self.pages[policy_id]
        if not 1 <= page <= len(pages):
            raise ValueError(f"Policy {policy_id!r} has pages 1 to {len(pages)}, not {page}.")
        return pages[page - 1]


APPEAL_GUIDE = """\
Structure for a health-insurance appeal grounded in the policy wording:
1. State the claim and the decision being challenged (claim number, date of the letter).
2. For each point, name the policy clause relied on and quote it exactly, with its page.
3. Say plainly where the insurer's own letter is inconsistent with the quoted clause.
4. List any documents the insured person still needs to supply.
5. Make no claim the policy text does not support. If the policy is silent, say so.
"""


def build_server(library: PolicyLibrary) -> FastMCP:
    mcp = FastMCP("policy-library", log_level="WARNING")

    @mcp.tool()
    def list_policies() -> dict:
        """List the policies this server may serve, with page and chunk counts."""
        return library.list_policies()

    @mcp.tool()
    def search_policy_clauses(query: QUERY, k: TOP_K = 5, policy_id: POLICY_ID | None = None) -> dict:
        """Keyword-search a policy and return the best-matching clauses with page numbers and character offsets."""
        return library.search(query, k, policy_id)

    @mcp.tool()
    def get_clause(policy_id: POLICY_ID, clause_id: CLAUSE_ID) -> dict:
        """Return the text of one numbered clause (for example 4.2) of a policy."""
        return library.get_clause(policy_id, clause_id)

    @mcp.tool()
    def get_chunk(chunk_id: CHUNK_ID) -> dict:
        """Return one chunk by id. Use this to re-check a quotation against the source."""
        return library.get_chunk(chunk_id)

    @mcp.resource("policy://{policy_id}/page/{page}")
    def policy_page(policy_id: str, page: str) -> str:
        """Full text of one policy page."""
        if not page.isdigit():
            raise ValueError("page must be a positive integer")
        return library.read_page(policy_id, int(page))

    @mcp.prompt()
    def appeal_structure() -> str:
        """How an evidence-grounded appeal should be organised."""
        return APPEAL_GUIDE

    return mcp


def main() -> None:
    build_server(PolicyLibrary.from_env()).run(transport="stdio")


if __name__ == "__main__":
    main()
