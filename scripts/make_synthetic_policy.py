"""Generate the synthetic test policy PDF.

The policy is fictional and exists only so the system can be tested without
real customer documents. Each entry in PAGES becomes exactly one PDF page, so
page numbers in the gold question set are stable.

Run from the repo root:
    python scripts/make_synthetic_policy.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parents[1] / "data" / "policies" / "synthetic_health_policy.pdf"

# Each page: a list of (kind, text). kind is "title", "note", "heading", "clause" or "para".
PAGES: list[list[tuple[str, str]]] = [
    # Page 1
    [
        ("title", "NORTHSTAR HEALTH ASSURANCE (FICTIONAL)"),
        ("title", "Family Floater Plus - Policy Wording"),
        ("note", "SYNTHETIC DOCUMENT FOR SOFTWARE TESTING ONLY. This is not a real insurance policy."),
        ("para", "Policy schedule: Sum Insured of Rs. 5,00,000 per policy year on a floater basis. The policy period is 12 months from the date of commencement."),
        ("heading", "SECTION 1 - DEFINITIONS"),
        ("clause", "1.1 Hospitalisation means admission to a hospital as an in-patient for a minimum period of 24 consecutive hours, except for day care procedures listed in Annexure A."),
        ("clause", "1.2 Pre-existing Disease means any condition, ailment, injury or disease that was diagnosed, or for which medical advice or treatment was received, within 36 months before the first policy start date."),
        ("clause", "1.3 Day Care Procedure means a medical treatment that needs less than 24 hours of hospitalisation because of technological advancement, and that is listed in Annexure A."),
        ("clause", "1.4 Medical Practitioner means a person who holds a valid registration from a recognised medical council and who acts within the scope of that registration."),
        ("clause", "1.5 Room Rent means the charge for accommodation and nursing care in a hospital room, and it does not include intensive care unit charges."),
        ("clause", "1.6 Medically Necessary means a treatment that is required for the diagnosis or treatment of an illness or injury, that is consistent with accepted medical practice, and that is not for the convenience of the patient or the hospital."),
    ],
    # Page 2
    [
        ("heading", "SECTION 2 - COVERAGE"),
        ("clause", "2.1 In-patient Hospitalisation: We pay the medically necessary expenses of hospitalisation for the treatment of an illness or injury, up to the Sum Insured."),
        ("clause", "2.2 Pre-hospitalisation: We pay related medical expenses incurred during the 30 days immediately before the date of admission."),
        ("clause", "2.3 Post-hospitalisation: We pay related medical expenses incurred during the 60 days immediately after the date of discharge."),
        ("clause", "2.4 Day Care Procedures: Day care procedures listed in Annexure A are covered irrespective of the duration of the stay in hospital."),
        ("clause", "2.5 Domiciliary Hospitalisation: We pay for treatment taken at home for more than 3 consecutive days if the insured person cannot be moved to a hospital on the advice of a medical practitioner."),
        ("clause", "2.6 Ambulance Cover: We pay up to Rs. 2,000 per hospitalisation for road ambulance charges to take the insured person to the hospital."),
        ("clause", "2.7 Organ Donor Expenses: We pay the in-patient hospitalisation expenses of the organ donor for a transplant covered under this policy, but not the cost of the organ."),
    ],
    # Page 3
    [
        ("heading", "SECTION 3 - WAITING PERIODS"),
        ("clause", "3.1 Initial Waiting Period: No claim is payable for an illness during the first 30 days from the policy start date, except a claim arising from an accident."),
        ("clause", "3.2 Pre-existing Diseases: Expenses for pre-existing diseases are payable only after 36 months of continuous coverage under this policy."),
        ("clause", "3.3 Specified Diseases: Cataract, hernia, hysterectomy, joint replacement and kidney stones are covered only after 24 months of continuous coverage."),
        ("clause", "3.4 Maternity: Maternity expenses are covered only after 24 months of continuous coverage and are limited to Rs. 50,000 per delivery."),
        ("clause", "3.5 Accidents: Waiting periods do not apply to hospitalisation caused by an accidental injury."),
        ("clause", "3.6 Continuity Credit: If this policy is renewed without a break, or ported from another insurer, the waiting period already served is credited to the insured person."),
        ("clause", "3.7 Break in Cover: A break in cover of more than 30 days after the due date of renewal ends the credit for waiting periods."),
    ],
    # Page 4
    [
        ("heading", "SECTION 4 - LIMITS AND CO-PAYMENT"),
        ("clause", "4.1 Room Rent Limit: Room rent is limited to a single private room up to 1% of the Sum Insured per day, and intensive care unit charges are limited to 2% of the Sum Insured per day."),
        ("clause", "4.2 Proportionate Deduction: If the insured person occupies a room above the limit in clause 4.1, all related expenses, including professional fees, are reduced in the same proportion as the room rent bears to the permitted limit."),
        ("clause", "4.3 Co-payment: A co-payment of 10% applies to every claim for an insured person who was aged 60 years or more at entry."),
        ("clause", "4.4 Cataract Sub-limit: Expenses for cataract surgery are limited to Rs. 40,000 per eye."),
        ("clause", "4.5 Annual Limit: The total amount payable in a policy year, including pre-hospitalisation and post-hospitalisation expenses, cannot exceed the Sum Insured."),
        ("clause", "4.6 Cumulative Bonus: The Sum Insured increases by 5% for each claim-free year, up to a maximum of 50% of the Sum Insured."),
        ("clause", "4.7 Floater Basis: The Sum Insured applies to all insured persons together and any one of them may use all of it."),
    ],
    # Page 5
    [
        ("heading", "SECTION 5 - EXCLUSIONS"),
        ("clause", "5.1 Cosmetic Surgery: We do not pay for cosmetic or plastic surgery, unless it is needed because of an accident or to rebuild after cancer or burns."),
        ("clause", "5.2 Dental Treatment: We do not pay for dental treatment, unless it results from an accident and needs hospitalisation."),
        ("clause", "5.3 Non-medical Expenses: Items listed in Annexure B, such as registration fees, admission kits, toiletries and service charges, are not payable."),
        ("clause", "5.4 War and Nuclear Risks: We do not pay for hospitalisation caused by war, invasion, nuclear reaction or radiation."),
        ("clause", "5.5 Treatment Outside India: We do not pay for treatment taken outside India."),
        ("clause", "5.6 Self-inflicted Injury and Substance Abuse: We do not pay for intentional self-inflicted injury or for treatment of alcohol or drug abuse."),
        ("clause", "5.7 Experimental Treatment: We do not pay for treatment that is experimental, unproven or not approved by a recognised medical authority."),
        ("clause", "5.8 Weight-control Surgery: We do not pay for obesity or weight-control surgery, unless the body mass index is above 40 and the insured person has a related medical condition."),
        ("clause", "5.9 Hospitalisation for Investigation Only: We do not pay for admission only for diagnostic tests or evaluation where no active treatment is given."),
    ],
    # Page 6
    [
        ("heading", "SECTION 6 - CLAIMS PROCEDURE"),
        ("clause", "6.1 Cashless Claims: For a planned admission, the insured person must send a pre-authorisation request to the claims desk at least 48 hours before admission."),
        ("clause", "6.2 Emergency Admission: In an emergency, the claims desk must be informed within 24 hours of admission."),
        ("clause", "6.3 Reimbursement Documents: For a reimbursement claim, the insured person must send the claim form, original bills, discharge summary, investigation reports and photo identification within 15 days of discharge."),
        ("clause", "6.4 Delay Condonation: A delay in giving notice or documents may be condoned if the insured person shows a reasonable cause for the delay."),
        ("clause", "6.5 Additional Documents: We may ask for additional documents within 7 days of receiving the claim, and we must give the reason for each request in writing."),
        ("clause", "6.6 Settlement Timeline: We must settle or reject a claim within 30 days of receiving the last necessary document."),
        ("clause", "6.7 Reasons for Rejection: A rejection must be in writing and must name the clause of this policy on which we rely."),
        ("clause", "6.8 Interest on Delay: If we do not settle an admitted claim within the period in clause 6.6, we pay interest on the amount at 2% above the bank rate from the date of receiving the last necessary document."),
    ],
    # Page 7
    [
        ("heading", "SECTION 7 - DISCLOSURE AND RENEWAL"),
        ("clause", "7.1 Duty of Disclosure: The proposer must disclose all material facts, including existing conditions, in the proposal form."),
        ("clause", "7.2 Moratorium: After 60 continuous months of coverage, no claim can be contested on grounds of non-disclosure or misrepresentation, except for proven fraud."),
        ("clause", "7.3 Grace Period: A grace period of 30 days is allowed for paying the renewal premium, and no claim is payable for an event during the grace period."),
        ("clause", "7.4 Renewal: The policy can be renewed for life, and we will not refuse renewal only because a claim was made."),
        ("heading", "SECTION 8 - GRIEVANCES"),
        ("clause", "8.1 Grievance Officer: A complaint may be sent in writing to the Grievance Officer, who must acknowledge it within 3 working days and resolve it within 15 days."),
        ("clause", "8.2 Escalation: If the complaint is not resolved, or the insured person is not satisfied, the matter may be taken to the Insurance Ombudsman or the regulator's grievance portal."),
        ("clause", "8.3 Communication: Every notice under this policy must be in writing and sent to the address on the policy schedule."),
    ],
    # Page 8
    [
        ("heading", "Annexure A - Day Care Procedures (illustrative list)"),
        ("para", "Haemodialysis, chemotherapy, radiotherapy, cataract surgery, tonsillectomy, lithotripsy for kidney stones, coronary angiography, and dilatation and curettage."),
        ("heading", "Annexure B - Non-payable Items (illustrative list)"),
        ("para", "Registration and admission charges, admission kit, toiletries, telephone and television charges, attendant food, gloves and masks when billed separately, service charges, and surcharge."),
    ],
]


def build() -> None:
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName="Helvetica", fontSize=10, leading=14, spaceAfter=6)
    heading = ParagraphStyle("heading", parent=body, fontName="Helvetica-Bold", fontSize=11, spaceBefore=8, spaceAfter=8)
    title = ParagraphStyle("title", parent=body, fontName="Helvetica-Bold", fontSize=14, leading=18, spaceAfter=4)
    note = ParagraphStyle("note", parent=body, fontName="Helvetica-Oblique", spaceAfter=10)
    style_for = {"title": title, "note": note, "heading": heading, "clause": body, "para": body}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title="Synthetic Health Policy (test document)",
    )
    story = []
    for i, page in enumerate(PAGES):
        for kind, text in page:
            story.append(Paragraph(text, style_for[kind]))
        story.append(Spacer(1, 1))
        if i < len(PAGES) - 1:
            story.append(PageBreak())
    doc.build(story)
    print(f"wrote {OUT} ({len(PAGES)} pages)")


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass
    build()
