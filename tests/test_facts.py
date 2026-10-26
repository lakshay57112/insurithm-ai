from insurithm.facts import extract_facts

LETTER = """Northstar Health Assurance
Date: 12 September 2026
Claim No: NHA/CL/2026/00417   Policy No. FFP-889120
Dear Mr Rao,
Your claim for Rs. 1,25,000 is rejected under clause 4.2 and exclusion 5.3 of the policy.
We have paid Rs 60,000/- after proportionate deduction. The 1.5 lakh room upgrade was disallowed.
You may appeal within 30 days of this letter. Treatment on 03/09/2026 was in Annexure B.
"""


def test_every_fact_points_at_its_own_text():
    facts = extract_facts(LETTER)
    assert facts.facts
    for f in facts.facts:
        assert LETTER[f.start : f.end] == f.text


def test_clause_references_including_lists():
    facts = extract_facts("Rejected under clauses 3.2 and 3.3, and section 6.2.")
    assert facts.clause_refs == ["3.2", "3.3", "6.2"]


def test_amounts_are_normalised():
    amounts = [f.value for f in extract_facts(LETTER).of_kind("amount")]
    assert amounts == ["125000", "60000"]
    assert extract_facts("Rs. 1.5 lakh and INR 2 crore and ₹ 4,500").of_kind("amount")[0].value == "150000"
    assert [f.value for f in extract_facts("Rs. 1.5 lakh and INR 2 crore and ₹ 4,500").of_kind("amount")] == [
        "150000", "20000000", "4500"
    ]


def test_dates_in_common_formats():
    text = "On 12th Sep 2026, then September 15, 2026, then 2026-10-01, then 03/09/2026."
    assert [f.value for f in extract_facts(text).of_kind("date")] == [
        "2026-09-12", "2026-09-15", "2026-10-01", "2026-09-03"
    ]


def test_impossible_dates_are_ignored():
    assert extract_facts("on 31/02/2026 and 45 March 2026").of_kind("date") == []


def test_day_first_is_assumed_but_month_first_is_a_fallback():
    assert extract_facts("on 03/09/2026").of_kind("date")[0].value == "2026-09-03"
    assert extract_facts("on 09/25/2026").of_kind("date")[0].value == "2026-09-25"


def test_letter_date_and_appeal_deadline():
    facts = extract_facts(LETTER)
    assert facts.letter_date == "2026-09-12"
    due, why = facts.appeal_deadline
    assert due == "2026-10-12"
    assert "30 days" in why


def test_no_deadline_without_a_dated_letter_or_with_working_days():
    assert extract_facts("You may appeal within 30 days.").appeal_deadline is None
    working = "Date: 12 September 2026. Appeal within 15 working days."
    assert extract_facts(working).appeal_deadline is None


def test_reference_numbers():
    facts = extract_facts(LETTER)
    assert facts.of_kind("claim_number")[0].value == "NHA/CL/2026/00417"
    assert facts.of_kind("policy_number")[0].value == "FFP-889120"
    assert extract_facts("claim number pending").of_kind("claim_number") == []


def test_decision_types():
    assert extract_facts(LETTER).decision_types == ["rejected", "short_paid"]
    assert extract_facts("Your claim is still pending and delayed.").decision_types == ["delayed"]
    assert extract_facts("Thank you for your letter.").decision_types == []


def test_annexure_reference():
    assert extract_facts(LETTER).of_kind("annexure_ref")[0].value == "B"


def test_empty_letter():
    assert extract_facts("").facts == []
