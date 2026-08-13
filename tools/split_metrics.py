#!/usr/bin/env python3
"""Split the Metrics notebooks out of their single-markdown-cell form.

Both Metrics notebooks were authored as ONE giant markdown cell:

    metrics_Python.ipynb -- 31 KB, 25 fenced ```python blocks
    metrics_SQL.ipynb    -- 26 KB, 22 fenced ```sql blocks

Nothing was executable, nothing was verified at build time, and there was no
way to run a single metric while drilling.

This script rewrites them:

  * ```python fences  -> real code cells (executed at build)
  * ```sql fences     -> kept as markdown (no SQL kernel), but the prose is
                         split per ``##`` heading so each metric is its own cell
  * prose between fences -> its own markdown cell, split on ``##`` headings

Text is preserved byte-for-byte apart from the fence delimiters themselves,
so tools/inventory.py reports these cells as ADDED, never LOST.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FENCE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.rstrip().splitlines(keepends=True),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.rstrip().splitlines(keepends=True),
    }


def split_prose(text: str) -> list[dict]:
    """Break a prose run into one markdown cell per ``##`` heading."""
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"\n(?=## )", text)
    return [md_cell(p) for p in parts if p.strip()]


def convert(path: Path, execute_python: bool) -> int:
    nb = json.loads(path.read_text(encoding="utf-8"))
    if len(nb["cells"]) != 1:
        print(f"  skip {path.name}: already has {len(nb['cells'])} cells")
        return len(nb["cells"])

    source = "".join(nb["cells"][0]["source"])
    cells: list[dict] = []
    cursor = 0

    for m in FENCE.finditer(source):
        cells.extend(split_prose(source[cursor : m.start()]))
        lang, body = m.group(1).lower(), m.group(2)
        if lang == "python" and execute_python:
            cells.append(code_cell(body))
        else:
            # SQL (and anything else) stays rendered markdown -- there is no
            # SQL kernel in this book, and the queries are copy-paste targets.
            cells.append(md_cell(f"```{lang or 'text'}\n{body}```"))
        cursor = m.end()

    cells.extend(split_prose(source[cursor:]))

    nb["cells"] = cells
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(cells)


def main() -> None:
    targets = [
        (ROOT / "Metrics" / "metrics_Python.ipynb", True),
        (ROOT / "Metrics" / "metrics_SQL.ipynb", False),
    ]
    for path, execute in targets:
        before = len(json.loads(path.read_text(encoding="utf-8"))["cells"])
        after = convert(path, execute)
        print(f"  {path.name}: {before} cell -> {after} cells")


if __name__ == "__main__":
    main()
