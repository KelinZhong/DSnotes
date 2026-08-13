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

IMPORTANT -- outputs are stripped ONLY from notebooks that execute at build
time. Notebooks listed in ``_config.yml``'s ``execute.exclude_patterns`` are
snippet references that cannot run standalone (undefined ``df``/``arr``, a
missing ``data.csv``); their stored outputs are the ONLY outputs they will ever
have, so stripping them silently blanks those cells forever.

Cell *source* is never touched, so tools/inventory.py hashes are unchanged.
"""
from __future__ import annotations

import fnmatch
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY_VERSION = "3.11"  # matches .github/workflows/deploy.yml

# Kept in sync with execute.exclude_patterns in _config.yml.
NEVER_STRIP_OUTPUTS = [
    "Python Data Libs/DM_*.ipynb",
    "Python Data Libs/Reference_Pandas.ipynb",
    "Python Data Libs/Reference_NumPy.ipynb",
    "Metrics/*",
]


def is_excluded(rel: Path) -> bool:
    s = rel.as_posix()
    return any(fnmatch.fnmatch(s, pat) for pat in NEVER_STRIP_OUTPUTS)

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
    rel = path.relative_to(ROOT)
    stats = {"outputs_stripped": 0, "kernel_added": False,
             "version_changed": False, "outputs_kept": False}

    meta = nb.setdefault("metadata", {})
    if "kernelspec" not in meta:
        stats["kernel_added"] = True
    old_version = meta.get("language_info", {}).get("version")
    if old_version and old_version != PY_VERSION:
        stats["version_changed"] = True
    meta["kernelspec"] = dict(KERNELSPEC)
    meta["language_info"] = dict(LANGUAGE_INFO)

    keep_outputs = is_excluded(rel)
    stats["outputs_kept"] = keep_outputs

    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            if keep_outputs:
                # Build-excluded: these outputs never regenerate. Leave them.
                cell.setdefault("outputs", [])
                cell.setdefault("execution_count", None)
            else:
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
