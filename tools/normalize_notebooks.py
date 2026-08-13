#!/usr/bin/env python3
"""Normalise notebook metadata across the book.

Fixes three things found in the audit:

1. 24 of 48 notebooks carried no ``kernelspec`` at all (all of Causal ML, all
   of Machine Learning, most of Data Science Workflow).
2. The ones that did declare four different Python versions -- 3.9.0, 3.10.0,
   3.11, 3.12.3 -- while CI pins 3.11.
3. The Experiment chapter was the only one storing outputs: 1.3 MB of base64
   PNGs in Exp_Analysis alone. Since ``execute_notebooks: "force"`` re-runs
   everything at build time, the stored bytes buy nothing.

Cell *source* is never touched, so tools/inventory.py hashes are unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY_VERSION = "3.11"  # matches .github/workflows/deploy.yml

KERNELSPEC = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

LANGUAGE_INFO = {
    "codemirror_mode": {"name": "ipython", "version": 3},
    "file_extension": ".py",
    "mimetype": "text/x-python",
    "name": "python",
    "nbconvert_exporter": "python",
    "pygments_lexer": "ipython3",
    "version": PY_VERSION,
}


def normalize(path: Path) -> dict:
    nb = json.loads(path.read_text(encoding="utf-8"))
    stats = {"outputs_stripped": 0, "kernel_added": False, "version_changed": False}

    meta = nb.setdefault("metadata", {})
    if "kernelspec" not in meta:
        stats["kernel_added"] = True
    old_version = meta.get("language_info", {}).get("version")
    if old_version and old_version != PY_VERSION:
        stats["version_changed"] = True
    meta["kernelspec"] = dict(KERNELSPEC)
    meta["language_info"] = dict(LANGUAGE_INFO)

    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            if cell.get("outputs"):
                stats["outputs_stripped"] += 1
            cell["outputs"] = []
            cell["execution_count"] = None
            cell.setdefault("metadata", {})

    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    total = {"files": 0, "outputs_stripped": 0, "kernel_added": 0, "version_changed": 0}
    for p in sorted(ROOT.rglob("*.ipynb")):
        if any(part in {"_build", ".git"} for part in p.parts):
            continue
        s = normalize(p)
        total["files"] += 1
        total["outputs_stripped"] += s["outputs_stripped"]
        total["kernel_added"] += int(s["kernel_added"])
        total["version_changed"] += int(s["version_changed"])
    print(
        f"normalised {total['files']} notebooks | "
        f"kernelspec added to {total['kernel_added']} | "
        f"python version aligned in {total['version_changed']} | "
        f"outputs stripped from {total['outputs_stripped']} cells"
    )


if __name__ == "__main__":
    main()
