"""Read the text of an uploaded PDF in a separate process.

Uploaded files are untrusted. A broken or hostile PDF can make a parser slow or hungry for memory,
so the API never parses one in its own process. It hands the bytes to a child process that has a
time limit (and, where the operating system allows it, a memory limit), and uses only the text
that comes back.

    extract_pdf_pages(data, max_pages=30)  ->  ["text of page 1", "text of page 2", ...]

The child is this same file run as ``python -m insurithm.pdf_text <max_pages>``: bytes in on stdin,
one JSON object out on stdout.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys

MEMORY_LIMIT_BYTES = 1 << 30  # 1 GiB, where the OS supports it
_SECRET_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


class PdfProblem(Exception):
    """The PDF cannot be used. `message` is written for the person who uploaded it."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def _child_env() -> dict[str, str]:
    import insurithm
    from pathlib import Path

    env = {k: v for k, v in os.environ.items() if not any(w in k.upper() for w in _SECRET_WORDS)}
    parent = str(Path(insurithm.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(p for p in [parent, env.get("PYTHONPATH", "")] if p)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run_child(data: bytes, max_pages: int, timeout: float) -> bytes:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # no console window flashing up on Windows
    try:
        done = subprocess.run(
            [sys.executable, "-m", "insurithm.pdf_text", str(max_pages)],
            input=data, capture_output=True, timeout=timeout, env=_child_env(), creationflags=flags,
        )  # on timeout, subprocess.run kills the child before raising
    except subprocess.TimeoutExpired:
        raise PdfProblem("too_slow", "This PDF took too long to read. Try a smaller or simpler file.") from None
    return done.stdout


async def extract_pdf_pages(data: bytes, *, max_pages: int, timeout: float = 30.0) -> list[str]:
    # A thread runs the blocking call, so this works with any event loop (including Windows' selector loop).
    out = await asyncio.to_thread(_run_child, data, max_pages, timeout)
    try:
        reply = json.loads(out.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise PdfProblem("unreadable", "This PDF could not be read. It may be damaged.") from None
    if not reply.get("ok"):
        raise PdfProblem(str(reply.get("code", "unreadable")), str(reply.get("message", "This PDF could not be read.")))
    return [str(p) for p in reply["pages"]]


def _reply(**fields: object) -> None:
    sys.stdout.write(json.dumps(fields))
    sys.stdout.flush()


def _child(max_pages: int) -> None:
    try:
        import resource  # not available on Windows

        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    except Exception:
        pass
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(sys.stdin.buffer.read()))
        if reader.is_encrypted:
            return _reply(ok=False, code="encrypted", message="This PDF is password-protected. Remove the password and upload it again.")
        count = len(reader.pages)
        if count == 0:
            return _reply(ok=False, code="empty", message="This PDF has no pages.")
        if count > max_pages:
            return _reply(ok=False, code="too_many_pages", message=f"This PDF has {count} pages. The limit here is {max_pages}.")
        _reply(ok=True, pages=[(page.extract_text() or "") for page in reader.pages])
    except MemoryError:
        _reply(ok=False, code="too_big", message="This PDF is too large to read safely.")
    except Exception:
        _reply(ok=False, code="unreadable", message="This PDF could not be read. It may be damaged.")


if __name__ == "__main__":
    _child(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
