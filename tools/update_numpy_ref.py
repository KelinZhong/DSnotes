#!/usr/bin/env python3
"""Add missing coverage to Reference_NumPy.

Fills the 11 gaps found by the audit. The largest cluster -- unique/bincount/
searchsorted/digitize/set operations -- had no home at all, so it becomes a new
``§10 — Set Operations, Binning & Search`` rather than being scattered.

Also adds the nan-aware aggregations (a surprising omission for a DS audience),
correlation/covariance, and the memory-and-layout tools (sliding_window_view,
structured arrays, masked arrays, memmap, savez).
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "Python Data Libs" / "Reference_NumPy.ipynb"


def md(t): return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8],
                   "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)}


def code(t): return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
                     "execution_count": None, "outputs": [],
                     "source": t.strip("\n").splitlines(keepends=True)}


# ── §5 additions: nan-aware + correlation ────────────────────────────────────
S5 = [md("""
**NaN-aware aggregations and correlation.** Ordinary NumPy reductions return `nan` if a single element is `nan`, which is almost never what you want on real data.
"""), code("""
a = np.array([1.0, 2.0, np.nan, 4.0])

a.mean()                    # nan  <- one missing value poisons the whole result
np.nanmean(a)               # 2.333...
np.nansum(a)                # 7.0
np.nanstd(a), np.nanvar(a)
np.nanmin(a), np.nanmax(a)
np.nanmedian(a)
np.nanpercentile(a, 90)
np.nanargmax(a)             # position of the max, ignoring NaN

# Detecting non-finite values
np.isnan(a).sum()           # count NaN
np.isfinite(a)              # False for NaN and +/-inf
np.isinf(a)
a[~np.isnan(a)]             # drop NaN

# NaN never equals itself -- this is why `a == np.nan` is always False
np.nan == np.nan            # False
np.isclose(a, 2.0, equal_nan=True)
np.allclose(a, a, equal_nan=True)     # True only with equal_nan

# ── Correlation and covariance ───────────────────────────────────────────────
x = np.random.default_rng(0).normal(size=100)
y = 2 * x + np.random.default_rng(1).normal(size=100)

np.corrcoef(x, y)           # 2x2 correlation MATRIX, not a scalar
np.corrcoef(x, y)[0, 1]     # the scalar you usually want
np.cov(x, y)                # 2x2 covariance matrix
np.cov(x, y, ddof=0)        # population (default ddof=1 -> sample)
np.corrcoef(M, rowvar=False)          # columns as variables, like a DataFrame

# Percentiles / quantiles
np.percentile(x, [25, 50, 75])
np.quantile(x, [0.25, 0.5, 0.75])     # same thing on a 0-1 scale
np.percentile(x, 50, method='linear') # interpolation strategy
"""), md("""
**Common mistakes:**
- Using `.mean()` on data with NaN and getting `nan` — reach for `np.nanmean`
- Treating `np.corrcoef(x, y)` as a number; it returns a 2×2 matrix, take `[0, 1]`
- Forgetting `rowvar=False` when variables are columns — NumPy assumes rows by default, the opposite of pandas
- Comparing with `== np.nan` rather than `np.isnan`
- Mixing `ddof` conventions: `np.var` defaults to population (`ddof=0`), `np.cov` defaults to sample (`ddof=1`)
""")]

# ── §9 additions: layout / memory ────────────────────────────────────────────
S9 = [md("""
**Memory layout and out-of-core tools.** `sliding_window_view` gives rolling windows with no copy at all; `memmap` reads arrays larger than RAM.
"""), code("""
from numpy.lib.stride_tricks import sliding_window_view

x = np.arange(10)
w = sliding_window_view(x, 4)     # shape (7, 4) -- a VIEW, zero copying
w.mean(axis=1)                    # rolling mean, fully vectorised
sliding_window_view(M, (3, 3))    # 2-D windows, e.g. image patches
# WARNING: the result is read-only and shares memory; .copy() before mutating.

# ── Structured / record arrays: heterogeneous columns in one array ───────────
dt = np.dtype([('name', 'U10'), ('age', 'i4'), ('score', 'f8')])
people = np.array([('Ana', 31, 88.5), ('Bo', 25, 91.0)], dtype=dt)
people['age']                     # field access by name
people[people['score'] > 90]
people.nbytes

# ── Masked arrays: carry a validity mask alongside the data ─────────────────
m = np.ma.masked_invalid(np.array([1.0, np.nan, 3.0]))
m.mean()                          # 2.0 -- masked entries excluded automatically
np.ma.masked_where(arr < 0, arr)  # mask by condition
m.filled(0)                       # back to a plain array

# ── memmap: array-like access to a file on disk, larger than RAM ────────────
mm = np.memmap('big.dat', dtype='float32', mode='w+', shape=(1_000_000, 10))
mm[:1000] = 1.0                   # only touched pages load into memory
mm.flush()
mm2 = np.memmap('big.dat', dtype='float32', mode='r', shape=(1_000_000, 10))

# ── Saving arrays: .npy / .npz beat CSV for numeric data ────────────────────
np.save('arr.npy', arr)                       # single array, binary, exact dtype
np.savez('bundle.npz', train=X, labels=y)     # several arrays
np.savez_compressed('bundle.npz', train=X)    # zlib-compressed
d = np.load('bundle.npz')
d['train']                                    # lazy per-array access
np.load('arr.npy', mmap_mode='r')             # memory-map instead of reading

# Layout checks
arr.flags['C_CONTIGUOUS']         # row-major
np.ascontiguousarray(arr)         # force a contiguous copy
arr.strides                       # bytes to step per axis
"""), md("""
**Common mistakes:**
- Writing to a `sliding_window_view` result — it is read-only and overlapping; `.copy()` first
- Assuming `memmap` loads nothing: slicing a large span still pages it in
- Using CSV to persist float arrays — `.npy` is faster, smaller, and preserves dtype exactly
- Building structured arrays for tabular work when a DataFrame is the better tool; structured arrays pay off mainly for fixed-layout binary I/O
""")]

# ── new §10 ───────────────────────────────────────────────────────────────────
S10 = [md("""
---
## §10 — Set Operations, Binning & Search

Counting distinct values, assigning values to buckets, and locating insertion points. These replace slow Python loops and `Counter` on numeric data.
"""), code("""
a = np.array([3, 1, 2, 3, 3, 1])

# ── Unique and counting ──────────────────────────────────────────────────────
np.unique(a)                                   # sorted distinct values
vals, counts = np.unique(a, return_counts=True)     # value_counts equivalent
vals, idx = np.unique(a, return_index=True)         # first position of each value
vals, inv = np.unique(a, return_inverse=True)       # inv rebuilds a from vals
np.unique(M, axis=0)                                # unique ROWS of a 2-D array

# bincount -- much faster than unique for small non-negative integers
np.bincount(a)                                  # counts at index 0..max
np.bincount(a, weights=w)                       # weighted sum per bucket (groupby-sum)
np.bincount(labels, minlength=10)               # force a fixed-length output

# ── Set operations on 1-D arrays ─────────────────────────────────────────────
np.intersect1d(a, b)                            # sorted common values
np.union1d(a, b)
np.setdiff1d(a, b)                              # in a but not b
np.setxor1d(a, b)                               # in exactly one
np.isin(a, [1, 3])                              # boolean mask -- vectorised `in`
np.isin(a, b, invert=True)                      # NOT IN

# ── Binning ──────────────────────────────────────────────────────────────────
edges = np.array([0, 10, 20, 30])
np.digitize([5, 15, 25], edges)                 # -> 1, 2, 3 (which bin)
np.digitize([5, 15], edges, right=True)         # half-open the other way

counts, bin_edges = np.histogram(x, bins=10)
counts, bin_edges = np.histogram(x, bins=edges)         # explicit edges
np.histogram2d(x, y, bins=20)                           # 2-D
np.histogram_bin_edges(x, bins='auto')                  # let NumPy choose

# ── Sorted search ────────────────────────────────────────────────────────────
s = np.array([1, 3, 5, 7, 9])
np.searchsorted(s, 4)                           # 2 -- where 4 would be inserted
np.searchsorted(s, [0, 4, 10])                  # vectorised
np.searchsorted(s, 5, side='right')             # 3 (after existing 5s)
# searchsorted is O(log n) and is how merge_asof works underneath.

# Partial sorting -- top-k without a full sort
np.argsort(x)[-5:]                              # O(n log n)
np.argpartition(x, -5)[-5:]                     # O(n), unordered top-5
idx = np.argpartition(x, -5)[-5:]
idx[np.argsort(x[idx])]                         # top-5, now ordered
"""), md("""
**Counting distinct values:**

| Tool | Input | Speed | Returns |
| :--- | :--- | :--- | :--- |
| `np.unique(return_counts=True)` | any dtype | O(n log n) | values + counts |
| `np.bincount` | non-negative ints only | O(n) | counts indexed 0..max |
| `collections.Counter` | any Python object | slow | dict |

**Common mistakes:**
- `np.bincount` on values with a huge maximum — it allocates `max+1` slots, so one value of 1e9 allocates a billion
- `np.bincount` on negative numbers or floats — raises; it is integers only
- `np.digitize` off-by-one: it returns 1-based bin numbers, and `right=` flips which edge is inclusive
- `np.searchsorted` on an unsorted array returns nonsense without warning
- Using `np.argsort` for a top-k when `np.argpartition` is O(n)
- Using Python's `in` on an array (`3 in a`) inside a loop instead of the vectorised `np.isin`
""")]

QA = md("""
---
## Interview Q&A

**Q: Why is NumPy faster than a Python list for numeric work?**
A: Contiguous, fixed-dtype memory plus loops that run in C rather than the
interpreter. A list holds pointers to boxed objects scattered in memory, so it
loses both cache locality and per-element type information. That also explains
the limitation: the speed disappears the moment you force an object dtype.

**Q: View or copy?**
A: Basic slicing gives a view sharing memory; fancy indexing (a boolean mask or
an integer array) gives a copy. The practical consequence is that writing to a
slice mutates the parent, which is a common source of silent bugs. `arr.base`
tells you whether an array is a view.

**Q: Compute a rolling mean without a loop.**
A: `sliding_window_view(x, k).mean(axis=1)`. It builds a strided view, so there
is no copying — but the view is read-only and overlapping, so anything mutating
needs an explicit `.copy()`.

**Q: Your array of counts came out enormous. Why?**
A: Almost certainly `np.bincount` on data with a large maximum — it allocates
`max+1` slots regardless of how many distinct values exist. `np.unique(...,
return_counts=True)` is the right tool for sparse or large-valued integers.

**Q: How would you find the 10 largest values in a million-element array?**
A: `np.argpartition(x, -10)[-10:]`, which is O(n), then sort just those ten if
order matters. A full `argsort` is O(n log n) and does far more work than the
question needs.

**Q: `np.random.seed()` or `default_rng()`?**
A: `default_rng()`. The legacy global seed mutates process-wide state, so any
library call can perturb your stream. A `Generator` is an explicit, isolated
object you can pass around and reason about.

### Gotchas
- Integer arrays overflow silently: `np.int8(127) + 1` wraps to `-128`
- Comparing floats with `==`; use `np.isclose` / `np.allclose`
- `np.nan != np.nan`, so `arr == np.nan` never matches; use `np.isnan`
- `axis=0` collapses rows (result is per-column) — the opposite of most people's first guess
- `np.append` and `np.concatenate` reallocate the whole array each call; build a list and convert once
- Assigning into a boolean-masked selection works, but chaining two fancy indexes (`a[m1][m2] = x`) writes to a temporary copy
""")


def main() -> None:
    nb = json.loads(TARGET.read_text(encoding="utf-8"))
    before = len(nb["cells"])
    if any("## §10" in "".join(c["source"]) for c in nb["cells"]):
        print("  already updated, skipping")
        return

    nb["cells"][34:34] = S9      # after §9's common-mistakes cell (idx 33)
    nb["cells"][20:20] = S5      # after §5's dot/matmul cell (idx 19)

    dg = next(i for i, c in enumerate(nb["cells"])
              if "# Decision Guide" in "".join(c["source"]))
    nb["cells"][dg:dg] = S10 + [QA]

    TARGET.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  Reference_NumPy: {before} -> {len(nb['cells'])} cells "
          f"(+{len(nb['cells']) - before})")


if __name__ == "__main__":
    main()
