"""Extract structured claim fields from EOB documents, one API call per document.

Raw Anthropic API plus a Pydantic schema; no frameworks. The schema is the
fixed output contract (field names, types, formats); deciding which document
values belong in each field is the prompt's job, and only the prompt iterates
between versions. Results land in results/extractions_<version>.json.
Usage: uv run src/extract.py v1|v2 [doc_id ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anthropic
from pydantic import BaseModel, Field

EXTRACT_MODEL = "claude-sonnet-5"
ROOT = Path(__file__).resolve().parent.parent
DOC_DIR = ROOT / "data" / "synthetic_eobs"
PROMPT_DIR = ROOT / "prompts"
RESULTS_DIR = ROOT / "results"


class FieldConfidence(BaseModel):
    """Confidence from 0 to 1 for each extracted field."""

    member_id: float = Field(ge=0, le=1)
    claim_number: float = Field(ge=0, le=1)
    service_date_start: float = Field(ge=0, le=1)
    service_date_end: float = Field(ge=0, le=1)
    provider_name: float = Field(ge=0, le=1)
    billed_amount: float = Field(ge=0, le=1)
    allowed_amount: float = Field(ge=0, le=1)
    paid_amount: float = Field(ge=0, le=1)
    member_responsibility: float = Field(ge=0, le=1)
    denial_codes: float = Field(ge=0, le=1)
    denial_reasons: float = Field(ge=0, le=1)


class EOBExtraction(BaseModel):
    """Claim fields extracted from one EOB document."""

    member_id: str | None = Field(description="Member or subscriber ID exactly as printed, null if absent")
    claim_number: str | None = Field(description="Claim number exactly as printed, null if absent")
    service_date_start: str | None = Field(description="First date of service, YYYY-MM-DD, null if absent")
    service_date_end: str | None = Field(description="Last date of service, YYYY-MM-DD, null if absent")
    provider_name: str | None = Field(description="Provider name, null if absent")
    billed_amount: float | None = Field(description="Total billed in dollars, null if absent")
    allowed_amount: float | None = Field(description="Total allowed in dollars, null if absent")
    paid_amount: float | None = Field(description="Total the plan paid in dollars, null if absent")
    member_responsibility: float | None = Field(description="Total the member owes in dollars, null if absent")
    denial_codes: list[str] = Field(description="Denial codes, empty list if none")
    denial_reasons: list[str] = Field(description="Denial reasons, empty list if none")
    confidence: FieldConfidence


def extract(client: anthropic.Anthropic, prompt_template: str, doc_text: str) -> EOBExtraction:
    """Run one extraction call and return the validated result."""
    response = client.messages.parse(
        model=EXTRACT_MODEL,
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt_template.replace("{document}", doc_text)}],
        output_format=EOBExtraction,
    )
    if response.parsed_output is None:
        raise RuntimeError(f"no parsed output (stop_reason={response.stop_reason})")
    return response.parsed_output


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("v1", "v2"):
        sys.exit("usage: uv run src/extract.py v1|v2 [doc_id ...]")
    version = sys.argv[1]
    only = set(sys.argv[2:])
    prompt_template = (PROMPT_DIR / f"extract_{version}.txt").read_text()

    out_path = RESULTS_DIR / f"extractions_{version}.json"
    results: dict[str, dict] = {}
    if only and out_path.exists():
        results = json.loads(out_path.read_text())

    client = anthropic.Anthropic()
    for doc_path in sorted(DOC_DIR.glob("*.txt")):
        doc_id = doc_path.stem
        if only and doc_id not in only:
            continue
        results[doc_id] = extract(client, prompt_template, doc_path.read_text()).model_dump()
        print(f"{doc_id}: extracted")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path.relative_to(ROOT)} ({len(results)} documents)")


if __name__ == "__main__":
    main()
