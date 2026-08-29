"""One-time generator for the synthetic EOB corpus used by this eval suite.

Calls Claude Haiku once per document spec and writes plain-text EOBs to
data/synthetic_eobs/. Every name, ID, and amount is fictional; no real PHI.
The generated documents are committed, so this script only needs to run again
to change the corpus. Usage: uv run src/generate_docs.py [doc_id ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import anthropic

GEN_MODEL = "claude-haiku-4-5"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic_eobs"

SYSTEM = """You generate synthetic Explanation of Benefits (EOB) documents as plain text,
used to test document-extraction software.

Rules:
- Everything is fictional. Invent the member name, member ID, claim number,
  and dollar amounts. Never use real insurer names, real provider names, or
  real personal data.
- One member, one claim number, one provider per document. State each key
  fact once; if a fact must repeat, keep it exactly identical.
- Keep arithmetic coherent: billed >= allowed, and allowed = plan paid +
  member responsibility, except where the scenario says a charge was denied.
- Dollar amounts have cents. Service dates fall in 2026.
- Output only the document text: no markdown, no code fences, no commentary."""

LAYOUTS: dict[str, str] = {
    "boxed": "Boxed statement: header with plan name, a labeled key-value block "
             "(member, ID, claim number, dates), a line-item charge table drawn "
             "with spaces, a totals section, and a 'THIS IS NOT A BILL' notice.",
    "letter": "Business letter to the member: 'Dear ...', facts woven into prose "
              "paragraphs rather than labeled fields, closing with a customer "
              "service paragraph.",
    "remit": "Dense remittance-advice style: abbreviated column headers (SVC DATE, "
             "CPT, BILLED, ALLOWED, PAID, PT RESP), fixed-width columns, terse "
             "footer codes section.",
    "card": "Compact summary card: short labeled lines, a prominent 'What you owe' "
            "callout near the end, minimal decoration.",
    "sections": "Two titled sections: 'Claim details' (who/what/when) then "
                "'Payment details' (amounts), using the label 'Subscriber ID' "
                "instead of member ID.",
    "typewriter": "Plain typewriter style: ALL-CAPS section headings, colon-separated "
                  "fields, ruled lines made of dashes.",
}

SPECS: list[dict] = [
    {"doc_id": "eob_01", "payer": "Cascadia Health Plan", "provider": "Dr. Alan Reyes, Family Medicine",
     "layout": "boxed", "period": "January 2026",
     "scenario": "Routine office visit, fully covered: plan pays the entire allowed amount, member owes 0.00. No denials.",
     "quirks": []},
    {"doc_id": "eob_02", "payer": "Harbor Point Mutual", "provider": "Lakeview Diagnostics Laboratory",
     "layout": "remit", "period": "January 2026",
     "scenario": "Blood panel with three lab line items, 20 percent member coinsurance after the allowed amount. No denials.",
     "quirks": []},
    {"doc_id": "eob_03", "payer": "Evergreen Benefits Group", "provider": "Summit Imaging Center",
     "layout": "card", "period": "February 2026",
     "scenario": "Knee MRI where most of the allowed amount goes to the member's remaining deductible; plan pays a small portion. No denials.",
     "quirks": ["Show one single consistent calculation. Never print alternative, preliminary, or corrected figures."]},
    {"doc_id": "eob_04", "payer": "Meridian Care Alliance", "provider": "Dr. Priya Natarajan, Dermatology",
     "layout": "letter", "period": "February 2026",
     "scenario": "Office visit plus skin biopsy, member coinsurance applies. No denials.",
     "quirks": ["Do NOT print the member ID anywhere. Include one sentence noting the ID was omitted from mailed statements for privacy."]},
    {"doc_id": "eob_05", "payer": "Northlake Assurance", "provider": "St. Brigid Medical Center Emergency Dept",
     "layout": "boxed", "period": "January 2026",
     "scenario": "Emergency room visit with four line items; member owes deductible plus coinsurance, a substantial amount. No denials.",
     "quirks": []},
    {"doc_id": "eob_06", "payer": "Cascadia Health Plan", "provider": "Riverbend Physical Therapy",
     "layout": "sections", "period": "March into April 2026",
     "scenario": "Six physical therapy visits spanning about five weeks (service date range, not a single date), 20 percent coinsurance. No denials.",
     "quirks": []},
    {"doc_id": "eob_07", "payer": "Harbor Point Mutual", "provider": "Dr. Samuel Osei, Cardiology",
     "layout": "letter", "period": "March 2026",
     "scenario": "Cardiology consultation and echocardiogram two days apart, coinsurance applies. No denials.",
     "quirks": ["Express every service date ONLY inside prose sentences (e.g. 'when you saw Dr. ... on March 9' and '... performed on March 11'). No labeled date field anywhere."]},
    {"doc_id": "eob_08", "payer": "Evergreen Benefits Group", "provider": "ClearPath Telehealth",
     "layout": "card", "period": "February 2026",
     "scenario": "Telehealth visit, member owes only a flat copay, plan pays the rest of the allowed amount. No denials.",
     "quirks": []},
    {"doc_id": "eob_09", "payer": "Meridian Care Alliance", "provider": "Foothills Orthopedic Group",
     "layout": "remit", "period": "April 2026",
     "scenario": "Knee arthroscopy claim with five line items. One line is reduced with adjustment code CO-45 (charge exceeds fee schedule), reason stated in the codes footer. Member owes deductible plus coinsurance.",
     "quirks": []},
    {"doc_id": "eob_10", "payer": "Northlake Assurance", "provider": "Dr. Elena Vasquez, Endocrinology",
     "layout": "typewriter", "period": "March 2026",
     "scenario": "Endocrinology follow-up plus an A1C lab test, coinsurance applies. No denials.",
     "quirks": []},
    {"doc_id": "eob_11", "payer": "Cascadia Health Plan", "provider": "Granite Bay Laboratory",
     "layout": "letter", "period": "April 2026",
     "scenario": "Vitamin D screening test denied as not medically necessary: denial code CO-50 with remark code N386. Plan pays nothing; member owes the full billed amount.",
     "quirks": ["Mention BOTH codes only inside a prose paragraph explaining the denial. No code table or codes section."]},
    {"doc_id": "eob_12", "payer": "Harbor Point Mutual", "provider": "Northgate Surgery Center",
     "layout": "boxed", "period": "February 2026",
     "scenario": "Preventive screening colonoscopy covered at 100 percent: member owes 0.00. No denials.",
     "quirks": []},
    {"doc_id": "eob_13", "payer": "Evergreen Benefits Group", "provider": "Dr. Marcus Hale, Psychiatry",
     "layout": "sections", "period": "May 2026",
     "scenario": "Two psychiatry visits in the same month (service date range), flat copay per visit. No denials.",
     "quirks": []},
    {"doc_id": "eob_14", "payer": "Meridian Care Alliance", "provider": "Pinecrest Family Clinic",
     "layout": "typewriter", "period": "January 2026",
     "scenario": "Sick visit with a rapid strep test, coinsurance applies. No denials.",
     "quirks": ["Label the member ID as 'Subscriber No.' and the claim number as 'Acct Ref'.",
                "Write every dollar amount WITHOUT a dollar sign, as plain numbers followed by ' USD'."]},
    {"doc_id": "eob_15", "payer": "Northlake Assurance", "provider": "Cedar Ridge Home Medical Supply",
     "layout": "remit", "period": "March 2026",
     "scenario": "One-month wheelchair rental denied entirely with code PR-204 (service not covered under the plan). Plan pays 0.00; member owes the full billed amount.",
     "quirks": []},
    {"doc_id": "eob_16", "payer": "Cascadia Health Plan", "provider": "Dr. Hannah Kim, OB-GYN",
     "layout": "card", "period": "April 2026",
     "scenario": "Annual preventive well-woman exam covered in full: member owes 0.00. No denials.",
     "quirks": []},
    {"doc_id": "eob_17", "payer": "Harbor Point Mutual", "provider": "Westbrook Urgent Care",
     "layout": "letter", "period": "May 2026",
     "scenario": "Urgent care visit with a wrist X-ray; plan paid part, member owes the rest.",
     "quirks": ["Never state a total allowed amount anywhere.",
                "Phrase the member's cost only as a sentence like 'you may owe up to 84.60' (pick your own amount)."]},
    {"doc_id": "eob_18", "payer": "Evergreen Benefits Group", "provider": "Dr. Robert Ellison, Gastroenterology",
     "layout": "boxed", "period": "May 2026",
     "scenario": "Diagnostic upper endoscopy with separate physician and facility line items; member owes deductible plus coinsurance. No denials.",
     "quirks": []},
    {"doc_id": "eob_19", "payer": "Meridian Care Alliance", "provider": "Silver Lake Sleep Center",
     "layout": "sections", "period": "April 2026",
     "scenario": "Overnight sleep study denied for missing precertification: code CO-197, with code CO-252 requesting additional documentation. Plan pays nothing yet; member responsibility pending appeal is the full billed amount.",
     "quirks": ["Express the service date ONLY in a prose sentence, no labeled date field.",
                "Mention both codes only inside the explanatory paragraph, no codes table.",
                "Print the Subscriber ID with internal spaces, like 'MCA 88 41 902' (invent your own digits)."]},
    {"doc_id": "eob_20", "payer": "Northlake Assurance", "provider": "Dr. Grace Obi, Pediatrics",
     "layout": "typewriter", "period": "February 2026",
     "scenario": "Pediatric sick visit with a nebulizer treatment; copay plus small coinsurance. No denials.",
     "quirks": []},
]


def build_prompt(spec: dict) -> str:
    """Render one document spec as the user prompt for the generator model."""
    quirks = "\n".join(f"- {q}" for q in spec["quirks"]) or "- none"
    return (
        "Write one synthetic EOB document.\n"
        f"Payer: {spec['payer']}\n"
        f"Provider: {spec['provider']}\n"
        f"Claim period: {spec['period']}\n"
        f"Layout: {LAYOUTS[spec['layout']]}\n"
        f"Scenario: {spec['scenario']}\n"
        f"Special instructions:\n{quirks}\n"
        "Length: roughly 30 to 60 lines."
    )


def generate(client: anthropic.Anthropic, spec: dict) -> str:
    """Generate one document and return its plain text."""
    response = client.messages.create(
        model=GEN_MODEL,
        max_tokens=2000,
        system=SYSTEM,
        messages=[{"role": "user", "content": build_prompt(spec)}],
    )
    text = next(b.text for b in response.content if b.type == "text").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
    return text + "\n"


def main() -> None:
    only = set(sys.argv[1:])
    client = anthropic.Anthropic()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for spec in SPECS:
        if only and spec["doc_id"] not in only:
            continue
        path = OUT_DIR / f"{spec['doc_id']}.txt"
        path.write_text(generate(client, spec))
        print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
