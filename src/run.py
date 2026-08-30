"""One-command report: v1 and v2 side by side, PDF column, routing, one summary.

Reads the committed extraction results when present and calls the API only
for missing runs (--fresh forces a full re-extraction) and for the single
live grounded-summary example at the end.
Usage: uv run src/run.py [--fresh]
"""

from __future__ import annotations

import json
import sys

import anthropic

import extract as ex
import score as scoring
import summarize as summ

RUNS = [("v1", False), ("v2", False), ("v2", True)]
COLUMNS = ["V1-TEXT", "V2-TEXT", "V2-PDF"]
SUMMARY_DOC = "eob_11"


def run_path(version: str, pdf: bool):
    return ex.RESULTS_DIR / f"extractions_{version}{'_pdf' if pdf else ''}.json"


def ensure_extractions(client: anthropic.Anthropic, fresh: bool) -> None:
    """Extract any run whose results file is missing (or all, with --fresh)."""
    for version, pdf in RUNS:
        path = run_path(version, pdf)
        if path.exists() and not fresh:
            continue
        template = (ex.PROMPT_DIR / f"extract_{version}.txt").read_text()
        source_dir, pattern = (ex.PDF_DIR, "*.pdf") if pdf else (ex.DOC_DIR, "*.txt")
        results = {}
        for doc_path in sorted(source_dir.glob(pattern)):
            content = ex.build_content(doc_path, template, pdf)
            results[doc_path.stem] = ex.extract(client, content).model_dump()
            print(f"{doc_path.stem}: extracted ({version}{' pdf' if pdf else ''})")
        path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


def routed_count(extractions: dict) -> int:
    return sum(
        1 for doc in extractions.values()
        for conf in doc["confidence"].values() if conf < scoring.CONFIDENCE_THRESHOLD
    )


def report() -> None:
    """Print the per-field comparison table across all three runs."""
    golden = {
        k: v for k, v in json.loads((ex.ROOT / "data" / "golden.json").read_text()).items()
        if not k.startswith("_")
    }
    split = json.loads((ex.ROOT / "data" / "split.json").read_text())
    doc_ids = sorted(golden)
    runs = [json.loads(run_path(version, pdf).read_text()) for version, pdf in RUNS]

    print(f"\nPER-FIELD ACCURACY ({len(doc_ids)} documents, {len(doc_ids) * len(scoring.FIELDS)} cells per run)")
    print(f"{'FIELD':<26}" + "".join(f"{c:>9}" for c in COLUMNS))
    for field in scoring.FIELDS:
        accs = [scoring.field_stats(field, golden, run, doc_ids)["accuracy"] for run in runs]
        print(f"{field:<26}" + "".join(f"{a:>9.2f}" for a in accs))
    for label, ids in [("OVERALL", doc_ids), ("  analysis set (15)", split["analysis"]),
                       ("  holdout set (5)", split["holdout"])]:
        accs = [scoring.cell_accuracy(golden, run, ids) for run in runs]
        print(f"{label:<26}" + "".join(f"{a:>9.3f}" for a in accs))
    routed = [routed_count(run) for run in runs]
    print(f"{'routed to human review':<26}" + "".join(f"{r:>9}" for r in routed))
    print(f"\n(confidence threshold {scoring.CONFIDENCE_THRESHOLD}; per-field precision "
          "and recall: uv run src/score.py v1|v2 [--pdf])")


def live_summary(client: anthropic.Anthropic) -> None:
    """Generate and check one member summary from the v2 extraction."""
    extractions = json.loads(run_path("v2", False).read_text())
    extraction = extractions[SUMMARY_DOC]
    summary, review = summ.generate_summary(client, extraction)
    trusted, _ = summ.split_fields(extraction)
    summ.print_report(SUMMARY_DOC, "v2", summary, summ.ground_check(summary, trusted), review)


def main() -> None:
    client = anthropic.Anthropic()
    ensure_extractions(client, "--fresh" in sys.argv)
    report()
    live_summary(client)


if __name__ == "__main__":
    main()
