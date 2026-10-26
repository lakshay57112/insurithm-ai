"""The policyholder's question is saved with the case and given to the model as untrusted text."""

from fastapi.testclient import TestClient

import scenarios as S
from insurithm.agent.llm import FunctionLLM
from insurithm.agent.prompts import draft_prompt, investigate_prompt
from insurithm.api import create_app

H = {"X-Owner-Id": "asha"}
QUESTION = "Why was my claim cut to Rs. 70,000, and can I still appeal?"


def test_prompts_carry_the_question_only_when_there_is_one():
    args = ("letter text", {"facts": []}, {}, [])
    assert "POLICYHOLDER'S QUESTION" not in investigate_prompt(*args, 4) and "POLICYHOLDER'S QUESTION" not in draft_prompt(*args)
    asked = investigate_prompt(*args, 4, QUESTION)
    assert QUESTION in asked and "(untrusted)" in asked
    drafted = draft_prompt(*args, QUESTION)
    assert QUESTION in drafted and "say so under gaps" in drafted
    assert " ".join(draft_prompt(*args, "a\n\n   b").split()).count("a b") == 1  # whitespace is flattened


def test_the_question_reaches_the_model_and_is_shown_with_the_case(tmp_path):
    prompts = []

    def model(stage, system, prompt):
        prompts.append((stage, prompt))
        return {"action": "finish"} if stage == "investigate" else {"findings": [], "gaps": [], "documents_to_gather": []}

    with TestClient(create_app(FunctionLLM(model), tmp_path)) as client:
        made = client.post("/cases", json={"letter_text": S.LETTER, "case_id": "q-001", "question": f"  {QUESTION}  "}, headers=H)
        assert made.status_code == 201
        assert client.get("/cases/q-001", headers=H).json()["case"]["question"] == QUESTION
        client.post("/cases/q-001/run", headers=H)
        assert [stage for stage, prompt in prompts if QUESTION in prompt] == ["investigate", "draft"]

        client.post("/cases", json={"letter_text": S.LETTER, "case_id": "q-002"}, headers=H)
        prompts.clear()
        client.post("/cases/q-002/run", headers=H)
        assert prompts and not any("QUESTION" in prompt for _, prompt in prompts)


def test_an_overlong_question_is_refused(tmp_path):
    with TestClient(create_app(FunctionLLM(lambda *a: {}), tmp_path)) as client:
        r = client.post("/cases", json={"letter_text": S.LETTER, "question": "x" * 1001}, headers=H)
        assert r.status_code == 422
