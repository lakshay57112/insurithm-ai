"""The approval step must survive a real crash: kill the process while it waits, then resume in a new one."""

import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
SRC = str(HERE.parent / "src")
WORKER = str(HERE / "restart_worker.py")


def env():
    return {**os.environ, "PYTHONPATH": os.pathsep.join([SRC, os.environ.get("PYTHONPATH", "")])}


def read_event(proc, seconds=90):
    deadline = time.time() + seconds
    while time.time() < deadline:
        ready, _, _ = select.select([proc.stdout], [], [], 1.0)
        if ready:
            line = proc.stdout.readline()
            if line.strip().startswith("{"):
                return json.loads(line)
        if proc.poll() is not None:
            break
    raise AssertionError("worker produced no event")


def test_kill_while_waiting_for_the_human_then_resume_in_a_new_process(tmp_path):
    first = subprocess.Popen(
        [sys.executable, WORKER, "start", str(tmp_path), "kill-001"],
        stdout=subprocess.PIPE, text=True, env=env(),
    )
    try:
        paused = read_event(first)
        assert paused["status"] == "awaiting_review" and paused["model_calls"] == 3
    finally:
        first.kill()  # SIGKILL: no clean shutdown, no chance to save anything more
        first.wait()
    assert first.returncode != 0

    second = subprocess.run(
        [sys.executable, WORKER, "resume", str(tmp_path), "kill-001"],
        capture_output=True, text=True, env=env(), timeout=120,
    )
    assert second.returncode == 0, second.stderr[-2000:]
    resumed = json.loads(second.stdout.strip().splitlines()[-1])

    assert resumed["status_before"] == "awaiting_review"
    assert resumed["appeal_sha"] == paused["appeal_sha"]      # the same draft came back from disk
    assert resumed["model_calls"] == 0                        # nothing was regenerated
    assert resumed["final_status"] == "appeal_approved"
    assert resumed["tracker_status"] == "appeal_approved"
    assert resumed["decision"] == "approved" and resumed["comment"] == "after restart"


def test_a_clean_exit_between_the_two_steps_works_too(tmp_path):
    first = subprocess.Popen(
        [sys.executable, WORKER, "start", str(tmp_path), "kill-002"],
        stdout=subprocess.PIPE, text=True, env=env(),
    )
    try:
        assert read_event(first)["status"] == "awaiting_review"
    finally:
        first.terminate()  # SIGTERM
        first.wait(timeout=30)
    second = subprocess.run(
        [sys.executable, WORKER, "resume", str(tmp_path), "kill-002"],
        capture_output=True, text=True, env=env(), timeout=120,
    )
    assert second.returncode == 0, second.stderr[-2000:]
    assert json.loads(second.stdout.strip().splitlines()[-1])["final_status"] == "appeal_approved"
