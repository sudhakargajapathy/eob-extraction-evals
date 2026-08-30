"""Member-friendly claim summary from extracted fields, with a groundedness check.

The summarizer never sees the source document. It receives only extracted
fields at or above the confidence threshold, so it cannot state anything
that was not extracted; low-confidence fields are withheld and routed to
human review. A deterministic check then maps every dollar amount, date,
code-like token, and percentage in the prose back to an extracted value and
flags any fact with no source. Usage: uv run src/summarize.py v1|v2 doc_id
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import anthropic

SUMMARY_MODEL = "claude-sonnet-5"
CONFIDENCE_THRESHOLD = 0.8
ROOT = Path(__file__).resolve().parent.parent

AMOUNT_FIELDS = ["billed_amount", "allowed_amount", "paid_amount", "member_responsibility"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
MONEY_RE = re.compile(r"\$\s?([0-9][\d,]*(?:\.\d{1,2})?)")
CODE_RE = re.compile(r"\b([A-Z]{1,3}-\d{1,4}|N\d{3})\b")
PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s?%")
DATE_RE = re.compile(
    rf"\b({'|'.join(MONTHS)}) (\d{{1,2}}), (\d{{4}})\b"
    r"|\b(\d{4})-(\d{2})-(\d{2})\b"
    r"|\b(\d{1,2})/(\d{1,2})/(\d{4})\b"
)

SUMMARY_PROMPT = """Write a short plain-language summary of this health insurance claim for the member.

Rules:
- Use ONLY the facts listed below. Do not add, compute, or infer any number,
  date, percentage, or code that is not listed.
- Write dollar amounts with a dollar sign; write dates from the facts.
- For fields listed as under review you may say the value is still being
  reviewed, but never state a value for them.
- Warm and clear, about a sixth-grade reading level, at most 120 words.

Facts:
{facts}
{review_note}"""


def split_fields(extraction: dict) -> tuple[dict, list[str]]:
    """Partition fields into trusted values and low-confidence review routing."""
    trusted: dict[str, object] = {}
    review: list[str] = []
    for field, conf in extraction["confidence"].items():
        if conf >= CONFIDENCE_THRESHOLD:
            trusted[field] = extraction[field]
        else:
            review.append(field)
    return trusted, review


def render_value(value: object) -> object:
    """Say what a null means so the summary does not call absence a review."""
    if value is None:
        return "not stated on this document"
    if isinstance(value, list) and not value:
        return "none"
    return value


def generate_summary(client: anthropic.Anthropic, extraction: dict) -> tuple[str, list[str]]:
    """Summarize one claim from its trusted extracted fields only."""
    trusted, review = split_fields(extraction)
    facts = "\n".join(f"- {field}: {render_value(value)}" for field, value in trusted.items())
    review_note = (
        "\nFields under review, no value available: " + ", ".join(review) + "\n"
        if review else ""
    )
    response = client.messages.create(
        model=SUMMARY_MODEL,
        max_tokens=3000,
        messages=[{
            "role": "user",
            "content": SUMMARY_PROMPT.format(facts=facts, review_note=review_note),
        }],
    )
    text = next(b.text for b in response.content if b.type == "text").strip()
    return text, review


def summary_dates(summary: str) -> list[tuple[str, str]]:
    """Return (raw text, ISO date) for every date expression in the summary."""
    found = []
    for m in DATE_RE.finditer(summary):
        if m.group(1):
            iso = f"{m.group(3)}-{MONTHS.index(m.group(1)) + 1:02d}-{int(m.group(2)):02d}"
        elif m.group(4):
            iso = f"{m.group(4)}-{m.group(5)}-{m.group(6)}"
        else:
            iso = f"{m.group(9)}-{int(m.group(7)):02d}-{int(m.group(8)):02d}"
        found.append((m.group(0), iso))
    return found


def ground_check(summary: str, trusted: dict) -> list[dict]:
    """Map every checkable fact in the summary to the extracted fields backing it."""
    findings = []
    amounts = {f: trusted.get(f) for f in AMOUNT_FIELDS if trusted.get(f) is not None}
    for m in MONEY_RE.finditer(summary):
        value = float(m.group(1).replace(",", ""))
        grounds = [f for f, v in amounts.items() if abs(float(v) - value) < 0.005]
        findings.append({"fact": m.group(0), "grounds": grounds})

    service_dates = {trusted.get("service_date_start"), trusted.get("service_date_end")} - {None}
    for raw, iso in summary_dates(summary):
        grounds = ["service_date_start/end"] if iso in service_dates else []
        findings.append({"fact": raw, "grounds": grounds})

    identifiers = [str(trusted.get(f) or "") for f in ("member_id", "claim_number")]
    codes = {str(c).upper() for c in trusted.get("denial_codes") or []}
    for m in CODE_RE.finditer(summary):
        token = m.group(1)
        if token.upper() in codes:
            findings.append({"fact": token, "grounds": ["denial_codes"]})
        elif any(token in ident for ident in identifiers):
            findings.append({"fact": token, "grounds": ["member_id/claim_number"]})
        else:
            findings.append({"fact": token, "grounds": []})

    for m in PERCENT_RE.finditer(summary):
        findings.append({"fact": m.group(0), "grounds": []})
    return findings


def print_report(doc_id: str, version: str, summary: str, findings: list[dict], review: list[str]) -> None:
    flagged = [f for f in findings if not f["grounds"]]
    print(f"\nMEMBER SUMMARY ({doc_id}, prompt {version}):\n")
    print(summary)
    print(f"\nGROUNDEDNESS CHECK: {len(findings)} facts, "
          f"{len(findings) - len(flagged)} grounded, {len(flagged)} flagged")
    for f in findings:
        status = "OK  " if f["grounds"] else "FLAG"
        print(f"  {status}  {f['fact']}  ->  {', '.join(f['grounds']) or 'no extracted field'}")
    if review:
        print(f"\n{len(review)} fields routed to human review, withheld from summary: "
              + ", ".join(review))
    else:
        print("\n0 fields routed to human review")


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in ("v1", "v2"):
        sys.exit("usage: uv run src/summarize.py v1|v2 doc_id")
    version, doc_id = sys.argv[1], sys.argv[2]
    extractions = json.loads((ROOT / "results" / f"extractions_{version}.json").read_text())
    if doc_id not in extractions:
        sys.exit(f"unknown doc_id: {doc_id}")

    client = anthropic.Anthropic()
    summary, review = generate_summary(client, extractions[doc_id])
    trusted, _ = split_fields(extractions[doc_id])
    print_report(doc_id, version, summary, ground_check(summary, trusted), review)


if __name__ == "__main__":
    main()
