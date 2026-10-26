"""Console helpers shared by the command-line tools."""

from __future__ import annotations

import sys


def ensure_utf8_output() -> None:
    """Stop printing from crashing on consoles that are not UTF-8 (older Windows terminals use cp1252).

    Characters the console cannot show are written as escapes instead of raising UnicodeEncodeError.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="backslashreplace")
            except Exception:
                pass
