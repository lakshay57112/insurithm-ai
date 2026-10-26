"""The command-line tools must not crash on consoles that cannot show every character (older Windows terminals)."""

import io
import sys

from insurithm import retrieval
from insurithm.console import ensure_utf8_output


def test_search_command_prints_page_and_clause(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["insurithm.retrieval", "emergency admission notice", "-k", "2"])
    retrieval.main()
    out = capsys.readouterr().out
    assert "[Clause 6.2]" in out and "Page 6" in out and "Rank 1" in out


def test_non_utf8_console_does_not_crash(monkeypatch):
    legacy = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", legacy)
    try:
        print("Rs. 5,00,000 ₹ → clause 4.2")  # characters cp1252 cannot encode
        raised = False
    except UnicodeEncodeError:
        raised = True
    assert raised  # this is the failure being guarded against

    legacy2 = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", legacy2)
    ensure_utf8_output()
    print("Rs. 5,00,000 ₹ → clause 4.2")
    legacy2.flush()
    assert b"clause 4.2" in legacy2.buffer.getvalue()
