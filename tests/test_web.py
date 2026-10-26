"""The web page is served by the API. These tests check it is reachable, locked down, and free of the
patterns that would let text from a letter or from the model run as HTML."""

import re

from fastapi.testclient import TestClient

from insurithm.agent.llm import ScriptedLLM
from insurithm.api import WEB_DIR, create_app


def test_page_and_assets_are_served_with_a_strict_policy(tmp_path):
    with TestClient(create_app(ScriptedLLM([]), tmp_path)) as client:
        page = client.get("/")
        assert page.status_code == 200 and "text/html" in page.headers["content-type"]
        assert "Insurithm" in page.text
        csp = page.headers["content-security-policy"]
        assert "script-src 'self'" in csp and "unsafe-inline" not in csp and "frame-ancestors 'none'" in csp
        assert page.headers["x-content-type-options"] == "nosniff"

        assert client.get("/assets/app.js").status_code == 200
        assert client.get("/assets/app.css").status_code == 200
        assert client.get("/assets/missing.js").status_code == 404
        assert client.get("/docs").status_code == 200  # the API docs page is untouched
        assert client.get("/health").json() == {"status": "ok"}


def test_demo_flag_is_reported(tmp_path):
    with TestClient(create_app(ScriptedLLM([]), tmp_path)) as client:
        assert client.get("/config").json() == {"demo": False}
    with TestClient(create_app(ScriptedLLM([]), tmp_path / "d", demo=True)) as client:
        assert client.get("/config").json() == {"demo": True}


def test_page_code_never_turns_text_into_html():
    script = (WEB_DIR / "assets" / "app.js").read_text(encoding="utf-8")
    for banned in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert banned not in script, banned
    # No inline scripts, styles or event handlers: the policy forbids them.
    page = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", page)
    assert not re.search(r"<style|\sstyle=|\son[a-z]+=", page)
    assert "style:" not in script
