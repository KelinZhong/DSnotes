#!/usr/bin/env python3
"""Turn plain-text notebook references into real MyST cross-reference links.

The audit found ZERO markdown hyperlinks across all 48 notebooks, but ~30
plain-text pointers such as:

    see `Data Science Workflow/ds7_deployment.ipynb`
    Companion to: `ds1_problem_framing.ipynb`
    see `cml3` §6
    | `Exp_Design`, `Exp_Analysis` |

Since the book is explicitly designed to interconnect, none of that was
traversable. This rewrites them as ``{doc}`` roles, which -- unlike plain
markdown links -- handle the spaces in the folder names correctly:

    {doc}`ds7_deployment <../Data Science Workflow/ds7_deployment>`

Rules:
  * a notebook is never linked to itself
  * text already inside a link is left alone
  * the visible label keeps the author's original wording
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {"_build", ".git", ".github", "tools"}

# Stems that are too generic / ambiguous to auto-link.
AMBIGUOUS = {"ds0", "ML0", "exp0", "cml0", "metrics0"}


def build_index() -> dict[str, Path]:
    index: dict[str, Path] = {}
    for p in sorted(ROOT.rglob("*.ipynb")):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        index[p.stem] = p.relative_to(ROOT).with_suffix("")
    return index


def relative_target(src: Path, dst: Path) -> str:
    """Doc-role target from one notebook to another (no extension)."""
    if src.parent == dst.parent:
        return dst.name
    up = "../" * len(src.parent.parts)
    return f"{up}{dst.as_posix()}"


def make_pattern(index: dict[str, Path]) -> re.Pattern:
    stems = sorted(index, key=len, reverse=True)
    alt = "|".join(re.escape(s) for s in stems)
    # optional folder prefix, optional .ipynb, optionally wrapped in backticks
    return re.compile(
        r"(?P<tick>`?)"
        r"(?P<prefix>(?:[A-Za-z ]+/)?)"
        r"(?P<stem>" + alt + r")"
        r"(?P<ext>\.ipynb)?"
        r"(?P=tick)"
    )


def linkify_text(text: str, src: Path, index: dict[str, Path], pattern: re.Pattern) -> tuple[str, int]:
    count = 0

    # Protect fenced code, inline math and existing links from rewriting.
    guards: list[str] = []

    def stash(m: re.Match) -> str:
        guards.append(m.group(0))
        return f"\x00{len(guards) - 1}\x00"

    protected = re.sub(r"```.*?```|\$\$.*?\$\$|\{doc\}`[^`]*`|\[[^\]]*\]\([^)]*\)",
                       stash, text, flags=re.DOTALL)

    def repl(m: re.Match) -> str:
        nonlocal count
        stem = m.group("stem")
        if stem in AMBIGUOUS and stem != src.stem:
            # index pages: still link them, they are the chapter landing pages
            pass
        target_path = index.get(stem)
        if target_path is None or stem == src.stem:
            return m.group(0)
        count += 1
        target = relative_target(src, target_path)
        return f"{{doc}}`{stem} <{target}>`"

    out = pattern.sub(repl, protected)

    def unstash(m: re.Match) -> str:
        return guards[int(m.group(1))]

    out = re.sub(r"\x00(\d+)\x00", unstash, out)
    return out, count


def main() -> None:
    index = build_index()
    pattern = make_pattern(index)
    total = 0
    touched = 0

    for p in sorted(ROOT.rglob("*.ipynb")):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        rel = p.relative_to(ROOT).with_suffix("")
        nb = json.loads(p.read_text(encoding="utf-8"))
        n = 0
        for cell in nb["cells"]:
            if cell["cell_type"] != "markdown":
                continue
            src_text = "".join(cell["source"])
            new_text, c = linkify_text(src_text, rel, index, pattern)
            if c:
                cell["source"] = new_text.splitlines(keepends=True)
                n += c
        if n:
            p.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"  {rel}: {n} links")
            touched += 1
            total += n

    print(f"\n{total} cross-reference links created across {touched} notebooks")


if __name__ == "__main__":
    main()
