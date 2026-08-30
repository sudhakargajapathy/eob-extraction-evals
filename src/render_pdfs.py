"""Render the synthetic EOB corpus as PDF documents for native PDF input.

Faithful monospace rendering: every field value stays byte-identical to the
text corpus, so data/golden.json applies to both modalities; box-drawing
decoration maps to ASCII for the base PDF fonts. reportlab only writes the
PDFs. Nothing in this project ever parses one; extraction reads them through
native document input. Usage: uv run src/render_pdfs.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen.canvas import Canvas

ROOT = Path(__file__).resolve().parent.parent
DOC_DIR = ROOT / "data" / "synthetic_eobs"
PDF_DIR = ROOT / "data" / "pdf_eobs"

DECORATION = str.maketrans({
    "─": "-", "━": "-", "═": "=", "–": "-", "—": "-",
    "│": "|", "║": "|",
    "┌": "+", "┐": "+", "└": "+", "┘": "+",
    "╔": "+", "╗": "+", "╚": "+", "╝": "+",
})

MARGIN = 54
FONT = ("Courier", 9)
LEADING = 11
WRAP_WIDTH = 93  # printable width at 9pt Courier; longer lines would clip off-page


def render(text: str, out_path: Path) -> None:
    """Write one document as a monospace letter-size PDF."""
    canvas = Canvas(str(out_path), pagesize=LETTER)
    _, height = LETTER
    y = height - MARGIN
    canvas.setFont(*FONT)
    for raw_line in text.translate(DECORATION).splitlines():
        for line in textwrap.wrap(raw_line, WRAP_WIDTH, drop_whitespace=False) or [""]:
            if y < MARGIN:
                canvas.showPage()
                canvas.setFont(*FONT)
                y = height - MARGIN
            canvas.drawString(MARGIN, y, line.encode("latin-1", "replace").decode("latin-1"))
            y -= LEADING
    canvas.save()


def main() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    for doc_path in sorted(DOC_DIR.glob("*.txt")):
        out_path = PDF_DIR / f"{doc_path.stem}.pdf"
        render(doc_path.read_text(), out_path)
        print(f"wrote {out_path.name}")


if __name__ == "__main__":
    main()
