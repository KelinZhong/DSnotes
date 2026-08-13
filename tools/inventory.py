#!/usr/bin/env python3
"""Content inventory for DSnotes.

Emits a SHA-256 per cell of source text (whitespace-normalised) for every
notebook and markdown file in the book. The *set* of hashes is the invariant:

  - hygiene + move phases  -> set must be IDENTICAL
  - additive phases        -> set may only GAIN entries

Usage:
    python tools/inventory.py > tools/inventory_baseline.txt     # record
    python tools/inventory.py | diff tools/inventory_baseline.txt -   # verify
    python tools/inventory.py --check tools/inventory_baseline.txt    # summary
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {"_build", ".git", ".github", "tools", "__pycache__"}


def norm(text: str) -> str:
    """Collapse whitespace so pure reformatting is not flagged as a change."""
    return re.sub(r"\s+", " ", text).strip()


def cell_hashes(path: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text(encoding="utf-8"))
        for i, cell in enumerate(nb.get("cells", [])):
            src = norm("".join(cell.get("source", [])))
            if not src:
                continue
            h = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
            out.append((h, f"{cell['cell_type']}:{i}"))
    elif path.suffix == ".md":
        src = norm(path.read_text(encoding="utf-8"))
        if src:
            out.append((hashlib.sha256(src.encode("utf-8")).hexdigest()[:16], "md:0"))
    return out


def walk() -> list[Path]:
    files = []
    for p in sorted(ROOT.rglob("*")):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix in {".ipynb", ".md"} and p.is_file():
            files.append(p)
    return files


def main() -> int:
    rows = []
    for p in walk():
        rel = p.relative_to(ROOT)
        for h, loc in cell_hashes(p):
            rows.append(f"{h}  {rel}  {loc}")

    if len(sys.argv) > 2 and sys.argv[1] == "--check":
        baseline = Path(sys.argv[2]).read_text().splitlines()
        old = {ln.split("  ")[0] for ln in baseline if ln.strip()}
        new = {r.split("  ")[0] for r in rows}
        lost, added = old - new, new - old
        print(f"baseline cells: {len(old)}")
        print(f"current  cells: {len(new)}")
        print(f"LOST:  {len(lost)}")
        print(f"ADDED: {len(added)}")
        if lost:
            print("\n!! content lost — these hashes are gone:")
            for ln in baseline:
                if ln.split("  ")[0] in lost:
                    print("   ", ln)
        return 1 if lost else 0

    print("\n".join(sorted(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
