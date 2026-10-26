"""Helper for test_restart_process.py. Run as:  python restart_worker.py start|resume WORKDIR CASE_ID

`start`  creates a case, runs the workflow until it pauses for human review, reports that it paused,
         then idles until the test kills the process (a crash while waiting for the human).
`resume` is a brand-new process: it opens the saved state, checks the draft is still there, and
         applies the human's approval. Its model has no replies, so any model call would fail.
"""

import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

import scenarios as S
from insurithm.agent.llm import ScriptedLLM
from insurithm.agent.tools import Tools
from insurithm.agent.workflow import open_runner


def emit(**event):
    print(json.dumps(event), flush=True)


async def start(work: Path, case: str) -> None:
    async with Tools.local(work / "tracker.sqlite") as tools:
        await tools.tracker.call("create_case", owner_id="asha", policy_id=S.POLICY_ID, letter_text=S.LETTER, case_id=case)
        llm = ScriptedLLM(S.full_script(case))
        async with open_runner(llm, tools, work / "checkpoints.sqlite") as runner:
            result = await runner.start(case, "asha")
            digest = hashlib.sha256(result["review"]["appeal_text"].encode()).hexdigest()
            emit(event="paused", status=result["status"], model_calls=len(llm.calls), appeal_sha=digest)
            time.sleep(600)


async def resume(work: Path, case: str) -> None:
    llm = ScriptedLLM([])
    async with Tools.local(work / "tracker.sqlite") as tools:
        async with open_runner(llm, tools, work / "checkpoints.sqlite") as runner:
            before = await runner.status(case)
            digest = hashlib.sha256(before["review"]["appeal_text"].encode()).hexdigest()
            done = await runner.decide(case, {"decision": "approve", "reviewer": "asha", "comment": "after restart"})
        stored = await tools.tracker.call("get_case", owner_id="asha", case_id=case)
    emit(
        event="resumed",
        status_before=before["status"],
        appeal_sha=digest,
        final_status=done["final_status"],
        model_calls=len(llm.calls),
        tracker_status=stored["case"]["status"],
        decision=stored["decision"]["decision"],
        comment=stored["decision"]["comment"],
    )


if __name__ == "__main__":
    mode, workdir, case_id = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    asyncio.run({"start": start, "resume": resume}[mode](workdir, case_id))
