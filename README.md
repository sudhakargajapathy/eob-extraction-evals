# eob-extraction-evals

Evaluation-driven LLM extraction on healthcare claim documents (EOBs):
a hand-labeled golden set, per-field precision/recall, one measured prompt
iteration, native PDF document input, and a grounded plain-language summary
layer with human-review routing. Raw Anthropic API plus Pydantic, no
frameworks. All 20 documents are synthetic; zero PHI.

## Results

20 documents, 11 fields, 220 scored cells per run. Extraction with
claude-sonnet-5; documents generated once with claude-haiku-4-5.

| | v1 text | v2 text | v2 PDF |
|---|---|---|---|
| overall accuracy | 0.995 | 1.000 | 1.000 |
| analysis set (15 docs) | 0.994 | 1.000 | 1.000 |
| holdout set (5 docs) | 1.000 | 1.000 | 1.000 |
| fields routed to human review (< 0.8 confidence) | 10 | 0 | 0 |

v1 was a deliberately plain prompt. Its one real error extracted an allowed
amount of 0.00 where the document never states one (correct answer: null),
and its review queue was flooded by correct-but-hedged cells: right answers
under an underspecified contract. Both true errors carried low confidence
and were routed, so routing recall was 100 percent. v2 changed only the
prompt: it defines the null-vs-zero rule, the code and date conventions,
and confidence as the probability the answer is correct with null as an
answer like any other. Per-field detail lives in `results/scores_*.json`.

The PDF column reuses the frozen v2 prompt through native PDF document
input (the model reads the PDF; no parsing or OCR libraries exist in this
repo) against the same golden set. Its first run scored 0.945 and the
harness localized every failure to the four letter-format documents: a
line-wrapping bug in our own PDF renderer, which the model degraded on
safely, returning low-confidence nulls that routed to review instead of
fabricated values. After the render fix, PDF input reaches parity.

## What this demonstrates

1. Golden set before scaling: ground truth labeled independently of the
   extraction pipeline, mechanically verified, human-adjudicated (five
   corrections, each its own commit; provenance in `data/golden.json`).
2. Measured iteration with a pre-registered 15/5 analysis/holdout split:
   failure analysis read only analysis documents, the scorer was frozen
   before the prompt iterated, and the delta is reported per field.
3. Grounded generation: the member summary is built from extracted fields
   only and a deterministic check maps every stated amount, date, and code
   back to a field, flagging any fact with no source.
4. Human-in-the-loop in code: sub-0.8-confidence fields are withheld from
   summaries and routed to review, and routing quality is measured as
   recall and precision, not just a count.
5. An honest stopping rule: no v3. At 220 cells the eval is saturated, and
   further tuning would fit noise; the next legitimate steps are more
   documents, scanned-image input as a fourth column, and regression tests.

The harness is provider-agnostic: the scorer consumes extraction JSON and
does not care which model or vendor produced it, so comparing providers or
models is a one-function change and an empirical question.

## How to run

```
export ANTHROPIC_API_KEY=...
uv run src/run.py            # full report from committed results + one live summary
```

Pipeline stages, each runnable alone: `generate_docs.py` (one-time corpus
generation), `render_pdfs.py`, `extract.py v1|v2 [--pdf]`,
`score.py v1|v2 [--pdf]`, `summarize.py v1|v2 <doc_id>`.
