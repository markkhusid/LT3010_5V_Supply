"""Export all project notebooks to PDF under ``pdfs/``.

Pipeline:
1. Execute notebook in-place
2. ``jupyter nbconvert --to latex``
3. Patch known pandoc longtable / LTcaptype LaTeX issues
4. ``tectonic`` compile → PDF in ``pdfs/``

Usage (from project root)::

    python export_notebooks_pdf.py
    python export_notebooks_pdf.py --no-execute
    python export_notebooks_pdf.py --only 01 04
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB_DIR = ROOT / "notebooks"
OUT = ROOT / "pdfs"
OUT.mkdir(parents=True, exist_ok=True)

NOTEBOOKS = sorted(NB_DIR.glob("*.ipynb"))


def run(cmd: list[str], cwd: Path | None = None, timeout: float | None = None) -> None:
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd or ROOT, check=True, timeout=timeout)


def fix_latex(tex: Path) -> None:
    text = tex.read_text(encoding="utf-8", errors="replace")
    text2 = re.sub(r"\{\\def\\LTcaptype\{none\}[^\n]*\n", "", text)
    text2 = re.sub(r"(\\end\{longtable\})\n\}\n", r"\1\n", text2)
    text2 = text2.replace(r"\LTcaptype{none}", r"\LTcaptype{table}")
    if r"\geometry{" in text2 and "margin=1in" in text2:
        text2 = text2.replace("margin=1in", "margin=0.75in")
    if r"\usepackage{graphicx}" in text2 and r"\setkeys{Gin}" not in text2:
        text2 = text2.replace(
            r"\usepackage{graphicx}",
            r"\usepackage{graphicx}" + "\n" + r"\setkeys{Gin}{width=\linewidth,keepaspectratio}",
        )
    unicode_map = {
        "∂": r"$\partial$", "≈": r"$\approx$", "⋅": r"$\cdot$", "µ": r"$\mu$",
        "Ω": r"$\Omega$", "≤": r"$\leq$", "≥": r"$\geq$", "±": r"$\pm$",
        "×": r"$\times$", "→": r"$\rightarrow$", "−": r"--",
    }
    for u, rep in unicode_map.items():
        text2 = text2.replace(u, rep)
    if text2 != text:
        tex.write_text(text2, encoding="utf-8")
        print(f"  patched LaTeX: {tex.name}")


def execute_notebook(nb: Path, timeout_s: int = 3600) -> None:
    run(
        [
            sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
            "--execute", "--inplace",
            f"--ExecutePreprocessor.timeout={timeout_s}",
            "--ExecutePreprocessor.kernel_name=python3",
            str(nb),
        ],
        cwd=ROOT,
        timeout=timeout_s + 120,
    )


def export_one(nb: Path, *, do_execute: bool) -> Path:
    if not nb.is_file():
        raise FileNotFoundError(nb)
    stem = nb.stem
    print(f"\n=== Exporting {nb.name} ===", flush=True)
    if do_execute:
        print("  executing notebook…", flush=True)
        execute_notebook(nb)
    run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "latex",
         f"--output-dir={OUT}", str(nb)],
        cwd=ROOT,
    )
    tex = OUT / f"{stem}.tex"
    if not tex.is_file():
        raise FileNotFoundError(f"expected {tex}")
    fix_latex(tex)
    run(["tectonic", "-X", "compile", "--keep-logs", str(tex.name)], cwd=OUT)
    pdf = OUT / f"{stem}.pdf"
    if not pdf.is_file():
        candidates = list(OUT.glob(f"{stem}*.pdf"))
        if not candidates:
            raise FileNotFoundError(f"PDF not produced for {stem}")
        pdf = candidates[0]
    print(f"  OK → {pdf} ({pdf.stat().st_size / 1e6:.2f} MB)")
    return pdf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-execute", action="store_true")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    for tool in ("pandoc", "tectonic"):
        path = shutil.which(tool)
        if not path:
            print(f"ERROR: {tool} not on PATH", file=sys.stderr)
            return 1
        print(f"{tool}: {path}")
    nbs = NOTEBOOKS
    if args.only:
        prefixes = tuple(args.only)
        nbs = [n for n in NOTEBOOKS if n.stem.startswith(prefixes)]
        if not nbs:
            print("No notebooks matched --only", file=sys.stderr)
            return 1
    pdfs: list[Path] = []
    errors: list[tuple[str, BaseException]] = []
    for nb in nbs:
        try:
            pdfs.append(export_one(nb, do_execute=not args.no_execute))
        except Exception as exc:  # noqa: BLE001
            errors.append((nb.name, exc))
            print(f"FAILED {nb.name}: {exc}", file=sys.stderr)
    print("\n=== Summary ===")
    for pdf in pdfs:
        print(f"  {pdf}")
    if errors:
        for name, exc in errors:
            print(f"  ERROR {name}: {exc}")
        return 1
    print(f"Wrote {len(pdfs)} PDF(s) to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
