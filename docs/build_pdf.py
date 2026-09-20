"""Render a Markdown file in docs/ to a PDF with the same stem.

Usage:  python docs/build_pdf.py docs/proposal.md

Images referenced from the Markdown (the FTI diagram) are inlined so the PDF is
self-contained. Rendering uses a headless Chromium; set CHROME to its path if it
is not on PATH.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

CSS = """
@page { size: A4; margin: 15mm 17mm 15mm 17mm; }
html { font-family: "Liberation Sans", "DejaVu Sans", Arial, sans-serif; font-size: 9.6pt;
       line-height: 1.32; color: #111; }
body { margin: 0; }
h1 { font-size: 15pt; margin: 0 0 2pt 0; line-height: 1.2; }
h2 { font-size: 11.5pt; margin: 9pt 0 3pt 0; border-bottom: 1px solid #999; padding-bottom: 1pt; }
p { margin: 0 0 4.5pt 0; text-align: justify; }
ul { margin: 0 0 4.5pt 0; padding-left: 16pt; }
li { margin: 0 0 1.5pt 0; }
code { font-family: "Liberation Mono", "DejaVu Sans Mono", monospace; font-size: 8.6pt; }
table { border-collapse: collapse; width: 100%; font-size: 8.8pt; margin: 3pt 0 5pt 0; }
th, td { border: 1px solid #bbb; padding: 2pt 4pt; vertical-align: top; text-align: left; }
th { background: #f0f0f0; }
img, svg { width: 100%; height: auto; display: block; margin: 3pt 0 4pt 0; }
a { color: #111; text-decoration: none; }
"""


def inline_images(html: str, base: Path) -> str:
    def repl(match: re.Match[str]) -> str:
        src = match.group(1)
        path = base / src
        if path.suffix.lower() == ".svg" and path.exists():
            svg = path.read_text(encoding="utf-8")
            svg = re.sub(r"<\?xml[^>]*\?>", "", svg)
            return svg
        return match.group(0)

    return re.sub(r'<img[^>]*src="([^"]+)"[^>]*/?>', repl, html)


def main(src: str) -> None:
    md_path = Path(src)
    body = markdown.markdown(md_path.read_text(encoding="utf-8"), extensions=["tables"])
    body = inline_images(body, md_path.parent)
    html_path = md_path.with_suffix(".html")
    html_path.write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head>"
        f"<body>{body}</body></html>",
        encoding="utf-8",
    )
    chrome = os.environ.get("CHROME") or shutil.which("chromium") or shutil.which("google-chrome")
    if not chrome:
        sys.exit("no Chromium found; set CHROME=/path/to/chrome")
    pdf_path = md_path.with_suffix(".pdf")
    subprocess.run(
        [
            chrome,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            html_path.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
    )
    html_path.unlink()
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/proposal.md")
