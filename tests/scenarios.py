"""A worked example used by the workflow tests and by the restart-process helper.

The letter is about a settlement that was cut by a proportionate deduction (policy clause 4.2).
The scripted model replies quote the real synthetic policy, plus two deliberately bad findings
that the verifier must drop.
"""

from __future__ import annotations

POLICY_ID = "synthetic_health_policy"

LETTER = """Northstar Health Assurance
Date: 12 September 2026
Claim No: NHA/CL/2026/00417   Policy No: FFP-889120

Dear Policyholder,

Your claim of Rs. 1,25,000 for the hospitalisation from 01/09/2026 to 04/09/2026 has been settled at Rs. 70,000.
We applied a proportionate deduction under clause 4.2 because the room occupied was above the permitted limit.
You may appeal within 30 days of this letter.
"""

LETTER_UNKNOWN_CLAUSE = LETTER.replace("clause 4.2", "clause 9.9")

C41 = f"{POLICY_ID}-p4-c0"
C42 = f"{POLICY_ID}-p4-c1"
C68 = f"{POLICY_ID}-p6-c7"


def letter_chunk_id(case_id: str) -> str:
    return f"letter-{case_id}-p1-c0"


def investigate_replies() -> list[dict]:
    return [
        {"action": "search", "query": "room rent limit proportionate deduction"},
        {"action": "finish"},
    ]


def draft_reply(case_id: str) -> dict:
    return {
        "findings": [
            {
                "claim": "A proportionate deduction under clause 4.2 applies only when the room occupied is above the limit in clause 4.1.",
                "citations": [{"chunk_id": C42, "quote": "If the insured person occupies a room above the limit in clause 4.1, all related expenses"}],
            },
            {
                "claim": "The room rent limit is a single private room up to 1% of the Sum Insured per day.",
                "citations": [{"chunk_id": C41, "quote": "Room rent is limited to a single private room up to 1% of the Sum Insured per day"}],
            },
            {
                "claim": "The insurer's letter says the deduction was applied because the room occupied was above the permitted limit.",
                "citations": [{"chunk_id": letter_chunk_id(case_id), "quote": "because the room occupied was above the permitted limit"}],
            },
            # Two findings the verifier must drop:
            {
                "claim": "The insurer must pay 12% interest on any delayed amount.",
                "citations": [{"chunk_id": C68, "quote": "we must pay interest at 12% per annum on all delays"}],
            },
            {"claim": "The policy covers experimental treatment.", "citations": []},
        ],
        "gaps": ["The room category actually occupied is not stated in the letter."],
        "documents_to_gather": ["Hospital bill showing the room category and daily rent"],
    }


def full_script(case_id: str) -> list[dict]:
    return [*investigate_replies(), draft_reply(case_id)]
