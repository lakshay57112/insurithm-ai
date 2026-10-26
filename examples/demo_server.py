"""Run the web page without an API key.

A scripted model stands in for Gemini, so the replies are canned and only fit the sample letter
(use "Use a sample letter" on the start page). Everything else is real: both MCP servers run as
separate processes, citations are checked against the policy, the run pauses for your decision
and the decision is saved.

    python examples/demo_server.py          # then open http://127.0.0.1:8000
"""

import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import scenarios as S  # noqa: E402
import uvicorn  # noqa: E402
from insurithm.agent.llm import FunctionLLM  # noqa: E402
from insurithm.api import create_app  # noqa: E402
from insurithm.console import ensure_utf8_output  # noqa: E402

_LETTER_CHUNK = re.compile(r"letter-([A-Za-z0-9_\-]+?)-p1-c0")


def scripted(stage: str, system: str, prompt: str) -> dict:
    if stage == "investigate":
        # The first investigation step has all its actions left; after that, finish.
        return S.investigate_replies()[0] if "Actions left: 4" in prompt else S.investigate_replies()[1]
    found = _LETTER_CHUNK.search(prompt)
    return S.draft_reply(found.group(1) if found else "demo")


def main() -> None:
    ensure_utf8_output()
    data = Path(tempfile.mkdtemp(prefix="insurithm-demo-"))
    print(f"Demo mode: the model is scripted. Data for this run is in {data}")
    print("Open http://127.0.0.1:8000 and choose 'Use a sample letter'.\n")
    uvicorn.run(create_app(FunctionLLM(scripted), data, demo=True), host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
