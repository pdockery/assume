#!/usr/bin/env python
"""Export the WSCC 9-bus teaching notebooks to code-free methodology PDFs.

Runs in the `assume` conda env from this directory:
    python export_methodology_pdf.py                       # all notebooks
    python export_methodology_pdf.py 202_market_seams.ipynb   # just one

Pipeline (per notebook): strip print/log stream output -> nbconvert
(--no-input drops all code) -> LaTeX -> xelatex x2.
We inject `\\newcounter{none}` into the preamble to work around a known
nbconvert+pandoc longtable bug (markdown tables emit `\\def\\LTcaptype{none}`
but the article template never defines a `none` counter -> "No counter 'none'
defined"). Figures and tables come through as cell outputs; DataFrames carry a
booktabs LaTeX representation (registered in each notebook's imports cell).
"""
import io
import json
import os
import re
import subprocess
import sys

# notebook -> output PDF basename
NOTEBOOKS = {
    "101_nodal_market_fundamentals.ipynb": "101_nodal_market_fundamentals_methodology",
    "112_three_ba_fundamentals.ipynb": "112_three_ba_fundamentals_methodology",
    "201_congestion_revenue_allocation.ipynb": "201_congestion_revenue_allocation_methodology",
    "202_market_seams.ipynb": "202_market_seams_methodology",
    "212_two_market_transfer.ipynb": "212_two_market_transfer_methodology",
    "301_two_settlement_congestion.ipynb": "301_two_settlement_congestion_methodology",
}
CLEAN_NB = "_export_clean.ipynb"


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def make_clean_copy(nb_path):
    """Temp copy that keeps markdown + figures + result tables but drops
    `print`/log stream output (INFO lines, 'wrote ...csv', list prints) that is
    neither a graph nor a table. Also lifts the first H1 heading out of the body
    to use as the document title (so it isn't both the title and section 1).
    Returns that title."""
    nb = json.load(io.open(nb_path, encoding="utf-8"))
    title = None
    for c in nb["cells"]:
        if title is None and c.get("cell_type") == "markdown":
            lines = "".join(c["source"]).splitlines()
            for k, ln in enumerate(lines):
                if ln.startswith("# "):
                    title = ln[2:].strip()
                    del lines[k]
                    c["source"] = "\n".join(lines).lstrip("\n").splitlines(keepends=True)
                    break
        if c.get("cell_type") == "code":
            c["outputs"] = [o for o in c.get("outputs", [])
                            if o.get("output_type") in ("display_data", "execute_result")]
    json.dump(nb, io.open(CLEAN_NB, "w", encoding="utf-8"), indent=1)
    return title


def export(nb_path, out):
    tex = out + ".tex"
    print(f"\n=== {nb_path} -> {out}.pdf ===")
    title = make_clean_copy(nb_path)
    run(["jupyter", "nbconvert", "--to", "latex", "--no-input",
         "--output", out, CLEAN_NB])

    src = io.open(tex, encoding="utf-8").read()
    # Preamble fix for the longtable 'none' counter (idempotent).
    if r"\newcounter{none}" not in src:
        src = src.replace(r"\begin{document}",
                          "\\newcounter{none}\n" + r"\begin{document}", 1)
    # Real document title instead of the temp filename.
    if title:
        src = re.sub(r"\\title\{[^}]*\}", r"\\title{" + title + "}", src, count=1)
    io.open(tex, "w", encoding="utf-8").write(src)

    for i in (1, 2):
        run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex])
        print(f"  xelatex pass {i} OK")

    # tidy LaTeX build artifacts, keep the .pdf
    for ext in (".tex", ".aux", ".log", ".out"):
        f = out + ext
        if os.path.exists(f):
            os.remove(f)
    files_dir = out + "_files"
    if os.path.isdir(files_dir):
        import shutil
        shutil.rmtree(files_dir)
    print(f"  -> {out}.pdf")


def main():
    targets = sys.argv[1:] or list(NOTEBOOKS)
    for nb_path in targets:
        export(nb_path, NOTEBOOKS[nb_path])
    if os.path.exists(CLEAN_NB):
        os.remove(CLEAN_NB)
    print("\nDONE:", ", ".join(NOTEBOOKS[t] + ".pdf" for t in targets))


if __name__ == "__main__":
    sys.exit(main())
