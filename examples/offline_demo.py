"""Run the whole appeal workflow without an API key.

A scripted model stands in for Gemini (its replies are in tests/scenarios.py, including two bad
findings that the verifier should drop). Everything else is real: both MCP servers run as
separate processes, citations are checked against the policy, the run pauses for human review,
and the decision is recorded in the claims tracker.

    python examples/offline_demo.py
"""

import asyncio
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import scenarios as S  # noqa: E402
from insurithm.agent.llm import ScriptedLLM  # noqa: E402
from insurithm.console import ensure_utf8_output  # noqa: E402
from insurithm.agent.tools import Tools  # noqa: E402
from insurithm.agent.workflow import open_runner  # noqa: E402


async def main() -> None:
    ensure_utf8_output()
    case = "demo-001"
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        async with Tools.local(work / "tracker.sqlite") as tools:
            await tools.tracker.call("create_case", owner_id="asha", policy_id=S.POLICY_ID, letter_text=S.LETTER, case_id=case)
            async with open_runner(ScriptedLLM(S.full_script(case)), tools, work / "checkpoints.sqlite") as runner:
                paused = await runner.start(case, "asha")
                print(f"workflow status: {paused['status']}\n")
                print(paused["review"]["appeal_text"])
                print("dropped by the citation checker:")
                for f in paused["review"]["dropped_findings"]:
                    print(f"  - {f['claim']}  [{f['reason']}]")

                print("\n-- a human approves the draft --\n")
                done = await runner.decide(case, {"decision": "approve", "reviewer": "asha", "comment": "looks right"})
                print(f"workflow status: {done['status']}, final status: {done['final_status']}")

            record = await tools.tracker.call("get_case", owner_id="asha", case_id=case)
            print(f"tracker: status={record['case']['status']}, decision={record['decision']['decision']}, deadlines={record['deadlines']}")


if __name__ == "__main__":
    asyncio.run(main())
