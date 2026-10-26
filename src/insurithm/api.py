"""HTTP API for Insurithm.

    POST /cases                    open a case from the text of a rejection letter (and an optional question)
    POST /cases/{id}/run           investigate, draft and verify; stops at the human review step
    GET  /cases/{id}               case, deadlines, notes and workflow status (including the pending draft)
    POST /cases/{id}/decision      the human approves or rejects the draft; the workflow resumes
    GET  /cases/{id}/audit         every tool call made on the case
    GET  /policies                 the shared policy plus any you uploaded
    POST /policies?name=...        upload your own policy PDF (raw bytes in the body); only you can use it
    POST /extract-letter           turn an uploaded PDF or text file into letter text (raw bytes in the body)
    GET  /health, GET /config
    GET  /                         the web page (static files in insurithm/web)

Callers identify themselves with the ``X-Owner-Id`` header. That is a stand-in for real authentication:
in a deployment, put an auth layer in front and derive the owner id from the verified token.
Everything is scoped to that owner; another owner's case looks like a case that does not exist.

Start it with a real model:  uvicorn insurithm.api:app_from_env --factory
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent.llm import LLM, GeminiLLM, LLMConfigError, LLMError
from .agent.tools import ToolError, Tools
from .agent.workflow import AppealRunner, open_runner
from .pdf_text import PdfProblem, extract_pdf_pages

_OWNER_RE = re.compile(r"^[A-Za-z0-9_.@+\-]{1,64}$")

WEB_DIR = Path(__file__).parent / "web"

MAX_LETTER_CHARS = 20000
MAX_LETTER_BYTES = 5 * 1024 * 1024
MAX_LETTER_PAGES = 30
MAX_POLICY_BYTES = 15 * 1024 * 1024
MAX_POLICY_PAGES = 200
MAX_POLICIES_PER_OWNER = 5
MIN_POLICY_CHARS = 400  # fewer than this and the PDF is probably a scan with no text layer
_UPLOAD_ID = re.compile(r"^u[0-9a-f]{16}-")  # uploaded policies are named u<owner hash>-<name>
# The page loads its own script and styles plus two font files from Google Fonts. It sets no inline
# scripts or styles, so scripts can be restricted to this server. (/docs is separate and unaffected.)
_PAGE_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; "
        "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-cache",
}


class NewCase(BaseModel):
    letter_text: str = Field(min_length=20, max_length=MAX_LETTER_CHARS, description="Text of the rejection or settlement letter")
    question: str = Field(default="", max_length=1000, description="What the policyholder wants to know (optional)")
    policy_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.\-]{1,64}$")
    case_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_\-]{3,64}$")


class Decision(BaseModel):
    decision: Literal["approve", "reject"]
    reviewer: str = Field(min_length=1, max_length=64)
    comment: str = Field(default="", max_length=1000)


def owner_id(x_owner_id: Annotated[str | None, Header()] = None) -> str:
    if not x_owner_id or not _OWNER_RE.match(x_owner_id):
        raise HTTPException(401, "Send a valid X-Owner-Id header (1-64 characters: letters, digits, _ . @ + -).")
    return x_owner_id


Owner = Annotated[str, Depends(owner_id)]

def owner_prefix(owner: str) -> str:
    """Private name prefix for one owner's uploaded policies (64 bits of a hash, so it cannot be guessed)."""
    return "u" + hashlib.sha256(owner.encode()).hexdigest()[:16] + "-"


def policy_visible(policy_id: str, owner: str | None) -> bool:
    """Shared policies are visible to everyone; an uploaded policy only to the owner who uploaded it."""
    if not _UPLOAD_ID.match(policy_id):
        return True
    return owner is not None and policy_id.startswith(owner_prefix(owner))


def _slug(name: str) -> str:
    stem = re.sub(r"\.pdf$", "", name.strip(), flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")[:40] or "policy"


def _tidy(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _decode_text(data: bytes) -> str:
    if b"\x00" in data:
        raise HTTPException(415, "That doesn't look like a text file or a PDF. Upload a .pdf or .txt file, or paste the text.")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")  # what Notepad on Windows often saves


async def read_capped(request: Request, limit: int) -> bytes:
    """Read an uploaded body, refusing anything over `limit` bytes before it is all in memory."""
    declared = request.headers.get("content-length", "")
    too_big = HTTPException(413, f"That file is too large. The limit is {limit // (1024 * 1024)} MB.")
    if declared.isdigit() and int(declared) > limit:
        raise too_big
    parts: list[bytes] = []
    total = 0
    async for part in request.stream():
        total += len(part)
        if total > limit:
            raise too_big
        parts.append(part)
    data = b"".join(parts)
    if not data:
        raise HTTPException(422, "No file was received.")
    return data


_CODE_TO_HTTP = {"not_found": 404, "already_exists": 409, "conflict": 409, "case_closed": 409}


def _http(err: ToolError) -> HTTPException:
    if err.code in _CODE_TO_HTTP:
        return HTTPException(_CODE_TO_HTTP[err.code], err.message)
    if err.code.startswith("invalid") or err.code == "policy_id_required":
        return HTTPException(422, err.message)
    if err.code == "server_unavailable":
        return HTTPException(503, "A tool server is unavailable.")
    return HTTPException(500, "Unexpected tool error.")


def create_app(
    llm: LLM, data_dir: str | Path, *, policy_pdfs: list[str | Path] | None = None,
    allowed_policies: list[str] | None = None, max_steps: int = 4, demo: bool = False,
) -> FastAPI:
    data = Path(data_dir)
    policy_dir = data / "policies"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        policy_dir.mkdir(parents=True, exist_ok=True)
        async with AsyncExitStack() as stack:
            tools = await stack.enter_async_context(
                Tools.local(data / "tracker.sqlite", policy_pdfs, allowed_policies, policy_dir=policy_dir)
            )
            app.state.tools = tools
            app.state.runner = await stack.enter_async_context(
                open_runner(llm, tools, data / "checkpoints.sqlite", max_steps=max_steps)
            )
            app.state.locks = {}
            yield

    app = FastAPI(title="Insurithm", version="0.1.0", lifespan=lifespan,
                  description="Evidence-grounded appeals for rejected health-insurance claims. Research project, not legal or insurance advice.")

    def tools_of() -> Tools:
        return app.state.tools

    def runner_of() -> AppealRunner:
        return app.state.runner

    def lock_for(case_id: str) -> asyncio.Lock:
        return app.state.locks.setdefault(case_id, asyncio.Lock())

    async def owned_case(owner: str, case_id: str) -> dict[str, Any]:
        try:
            return await tools_of().tracker.call("get_case", owner_id=owner, case_id=case_id)
        except ToolError as err:
            raise _http(err) from err

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/config")
    async def config() -> dict[str, bool]:
        """Lets the web page say when the model is a scripted stand-in."""
        return {"demo": demo}

    def optional_owner(x_owner_id: str | None) -> str | None:
        return x_owner_id if x_owner_id and _OWNER_RE.match(x_owner_id) else None

    async def visible_policies(owner: str | None) -> list[dict[str, Any]]:
        listing = (await tools_of().policies.call("list_policies"))["policies"]
        return [p for p in listing if policy_visible(p["policy_id"], owner)]

    @app.get("/policies")
    async def policies(x_owner_id: Annotated[str | None, Header()] = None) -> list[dict[str, Any]]:
        """The shared policy, plus the ones this owner uploaded (send X-Owner-Id to see those)."""
        out = []
        for p in await visible_policies(optional_owner(x_owner_id)):
            mine = bool(_UPLOAD_ID.match(p["policy_id"]))
            out.append({**p, "title": _UPLOAD_ID.sub("", p["policy_id"]), "mine": mine})
        return out

    @app.post("/policies", status_code=201)
    async def upload_policy(request: Request, owner: Owner, name: Annotated[str, Query(max_length=120)] = "policy") -> dict[str, Any]:
        prefix = owner_prefix(owner)
        async with lock_for(f"upload-{prefix}"):  # one upload at a time per owner, so the count limit holds
            if len(list(policy_dir.glob(f"{prefix}*.pdf"))) >= MAX_POLICIES_PER_OWNER:
                raise HTTPException(409, f"You can keep up to {MAX_POLICIES_PER_OWNER} uploaded policies. Use one of them instead.")
            body = await read_capped(request, MAX_POLICY_BYTES)
            if not body.startswith(b"%PDF-"):
                raise HTTPException(415, "That isn't a PDF. Upload the policy wording as a .pdf file.")
            try:
                pages = await extract_pdf_pages(body, max_pages=MAX_POLICY_PAGES)
            except PdfProblem as problem:
                raise HTTPException(422, problem.message) from problem
            if sum(len(p.strip()) for p in pages) < MIN_POLICY_CHARS:
                raise HTTPException(422, "This PDF has almost no readable text. If it is a scan or photos, it needs a text version first.")

            slug = _slug(name)
            policy_id = prefix + slug
            for n in range(2, 10):
                if not (policy_dir / f"{policy_id}.pdf").exists():
                    break
                policy_id = f"{prefix}{slug[:37]}-{n}"
            else:
                raise HTTPException(409, "You already have policies with that name. Rename the file and try again.")
            temp = policy_dir / f".{policy_id}.tmp"
            temp.write_bytes(body)
            os.replace(temp, policy_dir / f"{policy_id}.pdf")  # the library only looks at finished .pdf files

            try:
                listing = (await tools_of().policies.call("list_policies"))["policies"]
            except ToolError as err:
                (policy_dir / f"{policy_id}.pdf").unlink(missing_ok=True)
                raise _http(err) from err
            found = next((p for p in listing if p["policy_id"] == policy_id), None)
            if found is None:
                (policy_dir / f"{policy_id}.pdf").unlink(missing_ok=True)
                raise HTTPException(422, "This PDF could not be indexed. Check that it has selectable text.")
            return {**found, "title": policy_id[len(prefix):], "mine": True}

    @app.post("/extract-letter")
    async def extract_letter(request: Request, owner: Owner) -> dict[str, Any]:
        """Turn an uploaded PDF or text file into text for the letter box. Nothing is stored."""
        body = await read_capped(request, MAX_LETTER_BYTES)
        pages_read: int | None = None
        if body.startswith(b"%PDF-"):
            kind = "pdf"
            try:
                pages = await extract_pdf_pages(body, max_pages=MAX_LETTER_PAGES)
            except PdfProblem as problem:
                raise HTTPException(422, problem.message) from problem
            pages_read, text = len(pages), _tidy("\n\n".join(pages))
        else:
            kind, text = "text", _tidy(_decode_text(body))
        if len(text) < 20:
            raise HTTPException(422, "No readable text was found in that file. If it is a scan or a photo, paste the text instead.")
        return {"text": text[:MAX_LETTER_CHARS], "kind": kind, "pages": pages_read, "truncated": len(text) > MAX_LETTER_CHARS}

    @app.post("/cases", status_code=201)
    async def create_case(body: NewCase, owner: Owner) -> dict[str, Any]:
        try:
            available = [p["policy_id"] for p in await visible_policies(owner)]
        except ToolError as err:
            raise _http(err) from err
        policy_id = body.policy_id
        if policy_id is None:
            if len(available) != 1:
                raise HTTPException(422, "Several policies are available; send policy_id.")
            policy_id = available[0]
        elif policy_id not in available:  # only shared policies and this owner's own uploads are listed
            raise HTTPException(404, f"No policy {policy_id!r} is available.")
        try:
            made = await tools_of().tracker.call(
                "create_case", owner_id=owner, policy_id=policy_id, letter_text=body.letter_text, case_id=body.case_id,
                question=body.question.strip(),
            )
        except ToolError as err:
            raise _http(err) from err
        return {"case_id": made["case_id"], "status": made["status"], "policy_id": policy_id}

    @app.post("/cases/{case_id}/run")
    async def run_case(case_id: str, owner: Owner) -> dict[str, Any]:
        await owned_case(owner, case_id)
        async with lock_for(case_id):
            runner = runner_of()
            current = await runner.status(case_id)
            try:
                if current["status"] == "not_started":
                    return await runner.start(case_id, owner)
                if current["status"] == "incomplete":
                    return await runner.resume_after_failure(case_id)
            except LLMError as err:
                status = 503 if isinstance(err, LLMConfigError) else 502
                raise HTTPException(status, f"The language model failed: {err}. Call run again to continue from where it stopped.") from err
            except ToolError as err:
                raise _http(err) from err
            return current  # already paused or finished: running again changes nothing

    @app.get("/cases/{case_id}")
    async def get_case(case_id: str, owner: Owner) -> dict[str, Any]:
        stored = await owned_case(owner, case_id)
        workflow = await runner_of().status(case_id)
        stored["case"].pop("owner_id", None)
        return {**stored, "workflow": workflow}

    @app.post("/cases/{case_id}/decision")
    async def decide(case_id: str, body: Decision, owner: Owner) -> dict[str, Any]:
        await owned_case(owner, case_id)
        async with lock_for(case_id):
            current = await runner_of().status(case_id)
            if current["status"] != "awaiting_review":
                raise HTTPException(409, f"This case is not waiting for a decision (workflow status: {current['status']}).")
            try:
                return await runner_of().decide(case_id, body.model_dump())
            except ToolError as err:
                raise _http(err) from err

    @app.get("/cases/{case_id}/audit")
    async def audit(case_id: str, owner: Owner) -> dict[str, Any]:
        try:
            return await tools_of().tracker.call("get_audit_log", owner_id=owner, case_id=case_id)
        except ToolError as err:
            raise _http(err) from err

    @app.get("/", include_in_schema=False)
    async def web_page() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html", headers=_PAGE_HEADERS)

    app.mount("/assets", StaticFiles(directory=WEB_DIR / "assets"), name="assets")
    return app


def app_from_env() -> FastAPI:
    """Factory for uvicorn. Needs GEMINI_API_KEY and GEMINI_MODEL; data goes to INSURITHM_DATA_DIR (default ./insurithm_data)."""
    return create_app(GeminiLLM.from_env(), os.environ.get("INSURITHM_DATA_DIR", "insurithm_data"))
