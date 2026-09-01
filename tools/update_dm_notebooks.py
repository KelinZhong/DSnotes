#!/usr/bin/env python3
"""Add one new workflow section to each of DM_Basic and DM_Intermediate.

DM_Basic  §8 — Validation Contracts
    The notebook audits a dataset by hand in §1 but never encodes those checks
    as something re-runnable. This closes the loop and is the natural bridge to
    ds7_deployment's monitoring section.

DM_Intermediate §7 — As-of & Cross Joins
    merge_asof and how='cross' are workflow patterns rather than lookup
    entries, so they get a worked section: build a complete scaffold with
    cross, then attach point-in-time state with as-of.

Both notebooks are build-excluded, but the code here is written to run against
each notebook's existing setup frame so it can be copy-pasted and verified.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDL = ROOT / "Python Data Libs"


def md(t): return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8],
                   "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)}


def code(t): return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
                     "execution_count": None, "outputs": [],
                     "source": t.strip("\n").splitlines(keepends=True)}


# ══ DM_Basic §8 ══════════════════════════════════════════════════════════════
BASIC = [md("""
---
## §8 — Validation Contracts

§1 audits a dataset once, by eye. A **contract** turns those same checks into
something that runs on every refresh and fails loudly when the data drifts.

The distinction that matters in an interview: cleaning is a one-off, validation
is a control. Anything you discovered in §1–§7 should end up here so it cannot
silently regress.
"""), code("""
# ── A contract as a plain function: no dependencies, works anywhere ─────────
def validate(df: pd.DataFrame) -> list[str]:
    \"\"\"Return a list of contract violations. Empty list == data is good.\"\"\"
    errors = []

    # 1. Schema: required columns present
    required = {'user_id', 'event_date', 'age', 'revenue', 'country'}
    missing = required - set(df.columns)
    if missing:
        errors.append(f"missing columns: {sorted(missing)}")

    # 2. Uniqueness: the primary key really is a key
    if 'user_id' in df and df['user_id'].duplicated().any():
        n = df['user_id'].duplicated().sum()
        errors.append(f"user_id not unique: {n} duplicate rows")

    # 3. Nullability: columns that must never be null
    for col in ['user_id', 'country']:
        if col in df and df[col].isna().any():
            errors.append(f"{col} has {df[col].isna().sum()} nulls")

    # 4. Range: values inside plausible bounds
    if 'age' in df:
        bad = ~df['age'].between(0, 120) & df['age'].notna()
        if bad.any():
            errors.append(f"age out of [0,120] in {bad.sum()} rows")

    # 5. Membership: categoricals restricted to a known set
    allowed = {'US', 'Canada', 'UK', 'EU'}
    if 'country' in df:
        unknown = set(df['country'].dropna()) - allowed
        if unknown:
            errors.append(f"unexpected country values: {sorted(unknown)}")

    # 6. Freshness: the data actually covers recent dates
    if 'event_date' in df and pd.api.types.is_datetime64_any_dtype(df['event_date']):
        stale = (pd.Timestamp.now() - df['event_date'].max()).days
        if stale > 7:
            errors.append(f"stale data: newest row is {stale} days old")

    return errors


for e in validate(df):
    print("FAIL:", e)
"""), code("""
# ── assert_frame_equal — the regression test for a cleaning pipeline ────────
from pandas.testing import assert_frame_equal, assert_series_equal

expected = pd.DataFrame({'user_id': [1, 2], 'revenue': [1200.0, 950.0]})
actual   = pd.DataFrame({'user_id': [1, 2], 'revenue': [1200.0, 950.0]})

assert_frame_equal(actual, expected)                      # raises on any difference
assert_frame_equal(actual, expected, check_dtype=False)   # ignore int64 vs float64
assert_frame_equal(actual, expected, rtol=1e-3)           # float tolerance
assert_frame_equal(actual.sort_index(axis=1),
                   expected.sort_index(axis=1))           # ignore column order

# Pin the whole schema in one assertion
def assert_schema(df, schema: dict):
    assert set(schema) <= set(df.columns), set(schema) - set(df.columns)
    for col, dtype in schema.items():
        assert df[col].dtype == dtype, f"{col}: {df[col].dtype} != {dtype}"

# assert_schema(clean, {'user_id': 'int64', 'revenue': 'float64'})
"""), md("""
**Declarative alternative — `pandera`.** Once the checks outgrow a function,
a schema object is easier to read and produces better error messages:

```python
import pandera.pandas as pa

schema = pa.DataFrameSchema({
    "user_id":    pa.Column(int, unique=True, nullable=False),
    "age":        pa.Column(int, pa.Check.in_range(0, 120), nullable=True),
    "revenue":    pa.Column(float, pa.Check.ge(0)),
    "country":    pa.Column(str, pa.Check.isin(["US", "Canada", "UK", "EU"])),
    "event_date": pa.Column("datetime64[ns]", nullable=False),
})

clean = schema.validate(df, lazy=True)   # lazy=True collects ALL failures at once
```

`lazy=True` matters: without it validation stops at the first failure, so you
fix one problem per run instead of seeing the whole picture.

**Where each check belongs:**

| Check | Contract | Runs when |
| :--- | :--- | :--- |
| Required columns exist | schema | every load |
| Primary key unique | uniqueness | every load |
| Value in range | range | every load |
| Category in allowed set | membership | every load |
| Row count within ±20% of yesterday | volume | scheduled refresh |
| Null rate stable vs last week | drift | scheduled refresh |

**Common mistakes:**
- Validating *after* cleaning only — the interesting failures happen at ingest, so check both ends
- Raising on the first error instead of collecting all of them, which turns debugging into a loop
- Hard-coding thresholds that were true once; volume and null-rate checks should be relative to a recent baseline
- Treating validation as a data-quality nicety — silent schema drift is one of the most common causes of a model degrading in production ({doc}`ds7_deployment <../Data Science Workflow/ds7_deployment>`)
""")]

BASIC_QA = md("""
---
## Interview Q&A

**Q: A CSV column reads as `object` but should be numeric. Walk me through it.**
A: I'd look before converting. `df['col'].str.replace(r'[\\d.,-]', '', regex=True)
.value_counts()` shows exactly which non-numeric characters are present — usually
currency symbols, thousands separators, or a literal `'N/A'`. Then strip those and
use `pd.to_numeric(..., errors='coerce')`, and finally count how many rows became
NaN. Converting first and inspecting later hides the size of the problem.

**Q: 30% of a column is missing. Drop it?**
A: Not on the percentage alone — the mechanism matters more than the rate. If it
is missing at random and the column is weak, dropping is fine. If missingness is
itself informative — income blank because the user declined — then the null *is*
the signal and I'd add an indicator column rather than impute. I'd also check
whether missingness correlates with the target before deciding.

**Q: How do you handle outliers?**
A: First establish whether it is an error or a real extreme value. A 130-year-old
user is an error; a $50,000 order might be your best customer. Errors get fixed or
removed; genuine extremes get kept, and I'd change the *model* instead — winsorise,
log-transform, or use a robust loss. Deleting real extremes because they are
inconvenient is how you build a model that fails on exactly the rows that matter.

**Q: `drop_duplicates()` removed fewer rows than expected. Why?**
A: It compares all columns by default, so rows differing in a timestamp or an id
are not duplicates. The fix is `subset=` on the columns that define identity, plus
`keep='last'` if recency wins. Trailing whitespace and case differences also
defeat it, which is why string normalisation comes before dedup.

**Q: What is the first thing you run on a new dataset?**
A: `.info()` for shape, dtypes and null counts, then `.describe(include='all')`,
then `.nunique()` per column. Those three answer: is it the size I expected, did
anything load as the wrong type, and which columns are keys versus categoricals.

### Gotchas
- `df.dropna()` with no arguments drops any row with a null anywhere — usually far more than intended
- `astype(int)` on a column containing NaN raises; use `Int64` or fill first
- Cleaning strings after deduplicating means `' us '` and `'US'` survive as separate rows — normalise first, then dedup
- `replace()` without `regex=True` matches whole values only, which is a frequent silent no-op
- `pd.to_numeric(errors='coerce')` converts failures to NaN silently; always count them
""")

# ══ DM_Intermediate §7 ═══════════════════════════════════════════════════════
INTER = [md("""
---
## §7 — As-of & Cross Joins

Two joins that answer questions an equality join cannot:

- **cross** — every combination of left and right, no key. Builds complete
  scaffolds so that "no activity" shows up as a zero rather than a missing row.
- **merge_asof** — match each row to the most recent *earlier* row in another
  frame. This is the point-in-time join, and the standard tool for attaching
  state (price, plan, experiment assignment) as it stood at event time.
"""), code("""
# ── CROSS JOIN: build a complete user x month scaffold ──────────────────────
months = pd.DataFrame({'month': pd.date_range('2024-01-01', '2024-03-01', freq='MS')})
users  = df[['user_id']].drop_duplicates()

scaffold = users.merge(months, how='cross')      # 3 users x 3 months = 9 rows
print(f"{len(users)} users x {len(months)} months = {len(scaffold)} rows")

# Attach real activity; months with none become 0 rather than vanishing
actual = (df.assign(month=df['event_date'].dt.to_period('M').dt.to_timestamp())
            .groupby(['user_id', 'month'], as_index=False)['revenue'].sum())

complete = (scaffold.merge(actual, on=['user_id', 'month'], how='left')
                    .fillna({'revenue': 0}))
print(complete)

# Why this matters: grouping the raw data alone gives 5 rows, not 9. Any
# retention or run-rate metric computed on those 5 rows silently treats
# "no activity" as "no data" and overstates the average.
"""), code("""
# ── MERGE_ASOF: attach the plan price in force at event time ────────────────
prices = pd.DataFrame({
    'effective_date': pd.to_datetime(['2023-12-01', '2024-02-01']),
    'price':          [9.99, 12.99],
})

events = df[['event_date', 'user_id']].sort_values('event_date')
prices = prices.sort_values('effective_date')          # BOTH sides must be sorted

priced = pd.merge_asof(
    events, prices,
    left_on='event_date', right_on='effective_date',
    direction='backward',                    # most recent price at or before the event
)
print(priced)

# A plain merge cannot do this: event dates never equal effective dates.
# direction='forward' -> next change instead; 'nearest' -> closest either side.
"""), code("""
# ── merge_asof with a group key and a staleness guard ───────────────────────
quotes = pd.DataFrame({
    'time':   pd.to_datetime(['2024-01-01 09:00', '2024-01-01 09:05',
                              '2024-01-01 09:00']),
    'symbol': ['AAPL', 'AAPL', 'MSFT'],
    'bid':    [180.0, 181.0, 370.0],
}).sort_values('time')

trades = pd.DataFrame({
    'time':   pd.to_datetime(['2024-01-01 09:03', '2024-01-01 09:06',
                              '2024-01-01 15:00']),
    'symbol': ['AAPL', 'AAPL', 'MSFT'],
    'qty':    [10, 5, 2],
}).sort_values('time')

matched = pd.merge_asof(
    trades, quotes, on='time', by='symbol',        # match within symbol
    tolerance=pd.Timedelta('10min'),               # else leave NaN
    direction='backward',
)
print(matched)
# The 15:00 MSFT trade gets NaN: the only MSFT quote is 6 hours stale and the
# tolerance correctly refuses it. Without tolerance it would silently match.
"""), md("""
**Which join answers which question:**

| Question | Tool |
| :--- | :--- |
| Every user × every day, including empty days | `merge(how='cross')` |
| What was the price when this order was placed? | `merge_asof(direction='backward')` |
| When did the next change happen after this event? | `merge_asof(direction='forward')` |
| Closest reading either side | `merge_asof(direction='nearest')` |
| Match only within the same instrument/user | `merge_asof(by=...)` |
| Reject matches that are too old | `merge_asof(tolerance=...)` |

**Common mistakes:**
- **Not sorting.** `merge_asof` does not raise on unsorted input — it returns wrong matches. Sort both frames on the join key first
- Forgetting `tolerance`, which lets a value from weeks earlier attach silently
- Using `on=` when the two frames name the key differently — use `left_on`/`right_on`, and remember both must still be sorted
- Running `how='cross'` before filtering: output is `len(left) × len(right)`, so 50k × 50k is 2.5 **billion** rows
- Building a scaffold with cross and then inner-joining the data, which throws away exactly the empty combinations the scaffold existed to preserve — it must be a left join
""")]

INTER_QA = md("""
---
## Interview Q&A

**Q: Wide or long?**
A: Long for anything computational — groupby, plotting libraries, and most model
APIs expect one observation per row. Wide for human reading and for matrix-shaped
input. I treat long as the working format and pivot to wide only at the very end,
for display.

**Q: After a merge your row count went up. What happened?**
A: Duplicate keys on the right side, so it became many-to-many. I'd check
`right['key'].duplicated().any()` and pass `validate='m:1'` so pandas raises
instead of silently multiplying rows. `indicator=True` then shows how many rows
matched on each side.

**Q: `pivot` or `pivot_table`?**
A: `pivot` is a pure reshape and raises if the index/column pair is not unique.
`pivot_table` aggregates duplicates, defaulting to mean. If `pivot` raises, that
is real information — you had duplicate keys you did not know about — so I would
not reflexively switch to `pivot_table` without deciding what the duplicates mean.

**Q: Compute a running total per user.**
A: `df.sort_values(['user_id','date']).groupby('user_id')['amount'].cumsum()`.
The sort is the part people skip; `cumsum` follows row order, so an unsorted frame
gives a running total in arbitrary order without any warning.

**Q: You need each user's activity for every day in a month, including inactive days.**
A: Cross join a distinct-user frame with a `date_range` to make the full grid,
then left-join the activity and fill with zero. Grouping the events alone cannot
produce rows for days that have no events, which is the classic reason retention
curves come out too flat.

### Gotchas
- `melt` loses the index unless you `reset_index()` first
- `pivot_table` silently drops NaN groups; pass `dropna=False` to keep them
- `explode` leaves duplicate index labels — `reset_index(drop=True)` after
- `concat` of frames with different columns fills the gaps with NaN rather than raising; check `.columns` alignment first
- `groupby().apply()` is the slow path; prefer `agg`/`transform` when they can express the operation
""")


def main() -> None:
    for name, body, qa in [("DM_Basic", BASIC, BASIC_QA),
                           ("DM_Intermediate", INTER, INTER_QA)]:
        p = PDL / f"{name}.ipynb"
        nb = json.loads(p.read_text(encoding="utf-8"))
        before = len(nb["cells"])
        if any("## Interview Q&A" in "".join(c["source"]) for c in nb["cells"]):
            print(f"  {name}: already updated, skipping")
            continue
        dg = next(i for i, c in enumerate(nb["cells"])
                  if "## Decision Guide" in "".join(c["source"]))
        nb["cells"][dg:dg] = body + [qa]
        p.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"  {name}: {before} -> {len(nb['cells'])} cells (+{len(nb['cells']) - before})")

    # DM_Advanced gets Q&A only -- its section coverage was already complete.
    p = PDL / "DM_Advanced.ipynb"
    nb = json.loads(p.read_text(encoding="utf-8"))
    if not any("## Interview Q&A" in "".join(c["source"]) for c in nb["cells"]):
        adv = md("""
---
## Interview Q&A

**Q: How do you encode a high-cardinality categorical like zip code?**
A: One-hot is out — thousands of sparse columns. I'd reach for target encoding
computed **inside** the cross-validation folds, or a frequency encoding, or group
rare levels into an "other" bucket. The critical part is that target encoding
uses out-of-fold means; computing it on the full training set leaks the label and
produces validation scores that do not survive contact with production.

**Q: Where does leakage come from in feature engineering?**
A: Four usual sources: scaling or imputing before the split, target encoding on
the full data, aggregate features computed over the whole time range rather than
point-in-time, and any feature that could only be known after the label. The
defence is to fit every transform inside a `Pipeline` on the training fold only,
and to build time-based features with an as-of cutoff.

**Q: Standardise or normalise?**
A: Standardise (z-score) when the feature is roughly symmetric and the model
assumes scale — linear models, SVM, kNN, PCA, neural nets. Min-max when a bounded
range matters. Neither is needed for tree ensembles, which are invariant to
monotone transforms. For heavy skew I'd log-transform before scaling.

**Q: How do you build lag features without leaking?**
A: Sort by entity and time, group by entity, then `shift`. The mistakes are
forgetting the sort, forgetting the groupby so lags cross entity boundaries, and
using a centred rolling window, which reads the future. Every window has to be
strictly backward-looking.

### Gotchas
- `get_dummies` on train and test separately produces different column sets; fit once and reindex, or use `OneHotEncoder(handle_unknown='ignore')`
- `qcut` on a column with heavy ties raises on duplicate bin edges; pass `duplicates='drop'`
- Fitting a scaler on the full dataset before splitting is the most common leak in interview take-homes
- Interaction terms explode the feature count quadratically; select before expanding
""")
        dg = next(i for i, c in enumerate(nb["cells"])
                  if "## Decision Guide" in "".join(c["source"]))
        nb["cells"].insert(dg, adv)
        p.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"  DM_Advanced: +1 Interview Q&A cell")


if __name__ == "__main__":
    main()
