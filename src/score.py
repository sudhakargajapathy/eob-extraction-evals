"""Score extractions against the golden set, field by field.

Comparison rules: exact match for IDs and dates, cents tolerance for
amounts, case-insensitive fuzzy match for provider names and denial
reasons, order-insensitive set match for denial codes. Prints per-field
accuracy, precision, and recall plus the human-review routing count, and
writes the full detail to results/scores_<version>.json.

Failure detail is printed for analysis-set documents only: holdout
failures stay unread during prompt iteration (data/split.json). Pass
--show-holdout only for final reporting, after iteration is frozen.
Usage: uv run src/score.py v1|v2 [--show-holdout]
"""

from __future__ import annotations

import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIELDS = [
    "member_id", "claim_number", "service_date_start", "service_date_end",
    "provider_name", "billed_amount", "allowed_amount", "paid_amount",
    "member_responsibility", "denial_codes", "denial_reasons",
]
AMOUNT_FIELDS = {"billed_amount", "allowed_amount", "paid_amount", "member_responsibility"}
CONFIDENCE_THRESHOLD = 0.8


def normalize(text: str) -> str:
    """Lowercase and collapse everything but letters and digits to single spaces."""
    return " ".join("".join(c.lower() if c.isalnum() else " " for c in text).split())


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def names_match(a: str, b: str) -> bool:
    na, nb = normalize(a), normalize(b)
    return na in nb or nb in na or similarity(a, b) >= 0.85


def reasons_match(golden: list[str], predicted: list[str]) -> bool:
    """Every golden reason must pair with a distinct prediction that contains it
    or reaches 0.6 similarity; predictions may elaborate but not disagree."""
    if len(golden) != len(predicted):
        return False
    unused = list(predicted)
    for g in golden:
        candidates = [
            p for p in unused if normalize(g) in normalize(p) or similarity(g, p) >= 0.6
        ]
        if not candidates:
            return False
        unused.remove(max(candidates, key=lambda p: similarity(g, p)))
    return True


def values_match(field: str, golden: object, predicted: object) -> bool:
    if golden is None or predicted is None:
        return golden is None and predicted is None
    if field in AMOUNT_FIELDS:
        return abs(float(golden) - float(predicted)) < 0.005
    if field == "provider_name":
        return names_match(str(golden), str(predicted))
    if field == "denial_codes":
        return {str(c).upper().strip() for c in golden} == {str(c).upper().strip() for c in predicted}
    if field == "denial_reasons":
        return reasons_match(list(golden), list(predicted))
    return str(golden).strip() == str(predicted).strip()


def is_present(value: object) -> bool:
    return bool(value) if isinstance(value, list) else value is not None


def field_stats(field: str, golden: dict, extractions: dict, doc_ids: list[str]) -> dict:
    """Accuracy, precision, and recall for one field across the given documents."""
    tp = fp = fn = correct = 0
    for doc_id in doc_ids:
        g, p = golden[doc_id][field], extractions[doc_id][field]
        match = values_match(field, g, p)
        correct += match
        if is_present(p) and is_present(g) and match:
            tp += 1
        elif is_present(p) and not (is_present(g) and match):
            fp += 1
        if is_present(g) and not match:
            fn += 1
    return {
        "accuracy": correct / len(doc_ids),
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / (tp + fn) if tp + fn else 1.0,
    }


def cell_accuracy(golden: dict, extractions: dict, doc_ids: list[str]) -> float:
    cells = [
        values_match(f, golden[d][f], extractions[d][f]) for d in doc_ids for f in FIELDS
    ]
    return sum(cells) / len(cells)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("v1", "v2"):
        sys.exit("usage: uv run src/score.py v1|v2 [--show-holdout]")
    version = sys.argv[1]
    show_holdout = "--show-holdout" in sys.argv

    golden = {
        k: v for k, v in json.loads((ROOT / "data" / "golden.json").read_text()).items()
        if not k.startswith("_")
    }
    split = json.loads((ROOT / "data" / "split.json").read_text())
    extractions = json.loads((ROOT / "results" / f"extractions_{version}.json").read_text())
    doc_ids = sorted(golden)

    per_field = {f: field_stats(f, golden, extractions, doc_ids) for f in FIELDS}
    routed = [
        (d, f) for d in doc_ids for f in FIELDS
        if extractions[d]["confidence"][f] < CONFIDENCE_THRESHOLD
    ]
    failures = {
        d: {
            f: {"golden": golden[d][f], "predicted": extractions[d][f]}
            for f in FIELDS if not values_match(f, golden[d][f], extractions[d][f])
        }
        for d in doc_ids
    }
    failures = {d: fs for d, fs in failures.items() if fs}

    print(f"\nPER-FIELD SCORES ({version}, {len(doc_ids)} documents)")
    print(f"{'FIELD':<24}{'ACC':>7}{'PREC':>7}{'REC':>7}")
    for f in FIELDS:
        s = per_field[f]
        print(f"{f:<24}{s['accuracy']:>7.2f}{s['precision']:>7.2f}{s['recall']:>7.2f}")
    overall = cell_accuracy(golden, extractions, doc_ids)
    print(f"{'OVERALL (all cells)':<24}{overall:>7.2f}")
    print(f"{'  analysis set (15)':<24}{cell_accuracy(golden, extractions, split['analysis']):>7.2f}")
    print(f"{'  holdout set (5)':<24}{cell_accuracy(golden, extractions, split['holdout']):>7.2f}")
    print(f"\n{len(routed)} fields routed to human review (confidence < {CONFIDENCE_THRESHOLD})")

    shown = doc_ids if show_holdout else split["analysis"]
    print(f"\nFAILURES ({'all documents' if show_holdout else 'analysis set only'}):")
    for d in sorted(failures):
        if d not in shown:
            continue
        for f, detail in failures[d].items():
            print(f"  {d}.{f}: golden={detail['golden']!r} predicted={detail['predicted']!r}")
    hidden = sorted(set(failures) - set(shown))
    if hidden:
        print(f"  (failures in holdout docs not shown: {', '.join(hidden)})")

    out = {
        "version": version,
        "overall_accuracy": overall,
        "analysis_accuracy": cell_accuracy(golden, extractions, split["analysis"]),
        "holdout_accuracy": cell_accuracy(golden, extractions, split["holdout"]),
        "per_field": per_field,
        "human_review_count": len(routed),
        "human_review_fields": [f"{d}.{f}" for d, f in routed],
        "failures": failures,
    }
    out_path = ROOT / "results" / f"scores_{version}.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
