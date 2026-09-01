#!/usr/bin/env python3
"""Add missing API coverage to Reference_Pandas.

Fills the 27 gaps found by the coverage audit, matching the notebook's existing
house style: a supplementary markdown lead-in, a snippet-style code cell using
an undefined ``df`` placeholder, and a ``**Common mistakes:**`` block. The
notebook is build-excluded, so these cells are read, not executed.

Largest gap addressed: §5 had no ``how='cross'``, no ``merge_asof``, no
``merge_ordered``, no index-based merge, no ``.compare()`` and no Index set
operations.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "Python Data Libs" / "Reference_Pandas.ipynb"


def md(t): return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8],
                   "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)}


def code(t): return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
                     "execution_count": None, "outputs": [],
                     "source": t.strip("\n").splitlines(keepends=True)}


# ── §2 Selection ──────────────────────────────────────────────────────────────
S2 = [md("""
**Five more §2 selectors:** column selection by pattern, fast scalar access, locating the *row* of an extreme value, and MultiIndex slicing.
"""), code("""
# Select columns by name pattern (no boolean mask needed)
df.filter(like='revenue')                 # columns containing 'revenue'
df.filter(regex=r'_(usd|eur)$')           # columns matching a regex
df.filter(items=['a', 'b'])               # explicit list, silently skips missing
df.filter(like='2024', axis=0)            # can filter the INDEX instead

# .at / .iat — scalar access only, but much faster than .loc / .iloc
df.at[3, 'price']                         # label-based single value
df.iat[3, 2]                              # position-based single value
df.at[3, 'price'] = 9.99                  # assignment works too

# Locate the ROW of a max/min, not the value
df['price'].idxmax()                      # index label of the largest price
df.loc[df['price'].idxmax()]              # the whole winning row
df.loc[df.groupby('cat')['price'].idxmax()]   # best row PER category

# MultiIndex selection
mi = df.set_index(['region', 'date'])
mi.loc['West']                            # all rows for one outer level
mi.loc[('West', '2024-01-01')]            # a specific (outer, inner) pair
mi.xs('2024-01-01', level='date')         # slice on an INNER level
mi.xs('West', level='region', drop_level=False)   # keep the level in the result

idx = pd.IndexSlice
mi.loc[idx['West':'East', '2024-01':'2024-03'], :]   # range on both levels
"""), md("""
**`.loc` vs `.at` for a single value:**

| | Accepts | Returns | Relative speed |
| :--- | :--- | :--- | :--- |
| `.loc[r, c]` | labels, slices, masks, lists | scalar *or* Series/DataFrame | 1× |
| `.at[r, c]` | a single label pair only | always a scalar | ~5–10× faster |

**Common mistakes:**
- Using `.max()` when you want `.idxmax()` — `.max()` gives the value and loses which row produced it
- `.idxmax()` on an all-NaN column raises; guard with `.notna().any()` first
- Calling `.at` with a slice or list — it only accepts single labels, unlike `.loc`
- Forgetting `pd.IndexSlice` and writing `mi.loc['West':'East', '2024-01':]` directly, which is ambiguous across levels
""")]

# ── §3 NULL handling ──────────────────────────────────────────────────────────
S3 = [md("""
**Per-column fills and the modern NA types.** `np.nan` is a float, which is why an integer column with one missing value silently becomes `float64`. The nullable dtypes fix that.
"""), code("""
# Different fill value per column, in one call
df.fillna({'salary': 0, 'city': 'Unknown', 'score': df['score'].median()})

# Limit how far a fill propagates
df['v'].ffill(limit=2)                    # carry forward at most 2 rows

# ── np.nan vs pd.NA vs pd.NaT ────────────────────────────────────────────────
# np.nan  float missing        -> forces the column to float64
# pd.NaT  datetime missing
# pd.NA   dtype-agnostic missing, used by the nullable extension dtypes

s = pd.Series([1, 2, None])
s.dtype                                   # float64  <- int column silently widened

s = pd.Series([1, 2, None], dtype='Int64')    # capital I: nullable integer
s.dtype                                   # Int64, values stay integers, missing is pd.NA

# Nullable dtypes across the board
df = df.convert_dtypes()                  # infer Int64 / string / boolean automatically
df['flag'].astype('boolean')              # nullable bool: True / False / pd.NA
df['name'].astype('string')               # nullable string (not object)

# Three-valued logic: comparisons with pd.NA propagate rather than returning False
(pd.NA > 1)                               # <NA>, not False
pd.Series([True, pd.NA], dtype='boolean').any()    # True  (NA ignored)
pd.Series([False, pd.NA], dtype='boolean').all()   # False
"""), md("""
**Missing-value markers:**

| Marker | Belongs to | Forces float? | Equal to itself? |
| :--- | :--- | :--- | :--- |
| `np.nan` | numpy float | ✅ yes | ❌ no |
| `pd.NaT` | datetime / timedelta | n/a | ❌ no |
| `pd.NA` | nullable extension dtypes | ❌ no | ❌ no (propagates) |

**Common mistakes:**
- Expecting `Int64` (nullable) to behave like `int64` (numpy) — only the capitalised one accepts missing values
- Filtering with `df[df['flag']]` on a `boolean` column containing `pd.NA` — raises, because NA is not a valid mask value; use `df[df['flag'].fillna(False)]`
- Assuming `convert_dtypes()` is free — it copies the frame and can be slow on wide data
""")]

# ── §4 Manipulation ───────────────────────────────────────────────────────────
S4 = [md("""
**Chaining and index plumbing.** `.pipe()` completes the chainable trio with `.assign()` and `.query()`; the rest are the MultiIndex/alignment tools that show up the moment a groupby returns more than one level.
"""), code("""
# .pipe — insert your own function into a method chain
def add_margin(d, rate):
    return d.assign(margin=d['revenue'] * rate)

(df
   .query('revenue > 0')
   .pipe(add_margin, rate=0.3)            # instead of add_margin(df.query(...), 0.3)
   .sort_values('margin', ascending=False)
   .head(10))

# sort_values(key=) — sort by a transform WITHOUT adding a helper column
df.sort_values('name', key=lambda s: s.str.lower())        # case-insensitive
df.sort_values('code', key=lambda s: s.str.len())          # by string length
df.sort_values(['dept', 'sal'], ascending=[True, False])   # mixed directions

# MultiIndex plumbing
g = df.groupby(['region', 'year'])['sales'].agg(['sum', 'mean'])
g.droplevel(0)                            # drop an index level
g.droplevel(0, axis=1)                    # drop a COLUMN level (after multi-agg)
g.swaplevel(0, 1).sort_index()            # reorder levels, then re-sort
g.columns = ['_'.join(c) for c in g.columns]   # flatten a MultiIndex column

# align — put two objects on a common index before combining
a2, b2 = a.align(b, join='outer', axis=0, fill_value=0)
a2 + b2                                   # now safe: identical index
"""), md("""
**Common mistakes:**
- Adding a temporary column just to sort by it, then dropping it — `sort_values(key=)` does it in one step
- Forgetting that `key=` receives the **whole Series**, not one element, so `key=lambda x: x.lower()` fails; use `x.str.lower()`
- Arithmetic on two Series with different indexes producing surprise NaNs — pandas aligns on index first; use `.align()` or `.reset_index(drop=True)` deliberately
- Leaving a MultiIndex on columns after `.agg(['sum','mean'])` and then failing to select by name
""")]

# ── §5 Joins & Merges  (the flagged gap) ─────────────────────────────────────
S5 = [md("""
**The six join tools §5 did not cover.** `cross` builds every combination, `merge_asof` joins on *nearest earlier* key rather than equality, and `.compare()` answers "what changed between these two frames?".
"""), code("""
# ── CROSS JOIN — every row of left paired with every row of right ────────────
# No join key. Result has len(left) * len(right) rows.
users.merge(dates, how='cross')

# Classic use: build a complete scaffold so no (user, day) combination is missing
scaffold = users[['user_id']].merge(
    pd.DataFrame({'date': pd.date_range('2024-01-01', '2024-01-31')}),
    how='cross'
)
# then left-join actual events onto the scaffold -> zero-filled gaps
full = scaffold.merge(events, on=['user_id', 'date'], how='left').fillna({'n': 0})

# Other cross uses: parameter grids, all pairwise combinations
params = grid_a.merge(grid_b, how='cross')

# Pre-1.2 equivalent, still seen in older code:
left.assign(_k=1).merge(right.assign(_k=1), on='_k').drop(columns='_k')

# ── MERGE ON INDEX ───────────────────────────────────────────────────────────
left.merge(right, left_index=True, right_index=True, how='inner')
left.merge(right, left_on='user_id', right_index=True, how='left')   # mixed
left.join(right, how='left')              # .join() defaults to index-on-index

# ── MERGE_ASOF — join to the most recent EARLIER row ─────────────────────────
# Both frames MUST be sorted on the `on` key first.
trades = trades.sort_values('time')
quotes = quotes.sort_values('time')

pd.merge_asof(trades, quotes, on='time')                    # nearest earlier quote
pd.merge_asof(trades, quotes, on='time', by='symbol')       # match within symbol
pd.merge_asof(trades, quotes, on='time',
              tolerance=pd.Timedelta('2min'),               # else leave NaN
              direction='backward')                         # 'forward' | 'nearest'

# ── MERGE_ORDERED — outer merge preserving order, with optional group fill ───
pd.merge_ordered(df1, df2, on='date', fill_method='ffill', left_by='group')

# ── COMPARE — row-level diff of two like-shaped frames ───────────────────────
before.compare(after)                             # only changed cells, self/other
before.compare(after, keep_equal=True)            # show unchanged values too
before.compare(after, align_axis=0)               # stack instead of side-by-side

# ── INDEX SET OPERATIONS — often clearer than a merge ────────────────────────
a.index.intersection(b.index)             # in both
a.index.union(b.index)                    # in either
a.index.difference(b.index)               # in a only
a.index.symmetric_difference(b.index)     # in exactly one
df.loc[a.index.difference(b.index)]       # rows dropped between two snapshots
"""), md("""
**Choosing a join:**

| Goal | Tool |
| :--- | :--- |
| Match on equal keys | `merge(on=...)` |
| Every combination, no key | `merge(how='cross')` |
| Match on *nearest earlier* key | `merge_asof(direction='backward')` |
| Match on nearest key either side | `merge_asof(direction='nearest')` |
| Align on index | `merge(left_index=True, right_index=True)` or `.join()` |
| Stack rows / columns | `pd.concat` |
| Fill gaps from a second frame | `combine_first` |
| See what changed | `.compare()` |
| Which keys are shared | `Index.intersection` / `.difference` |

**Common mistakes:**
- Running `how='cross'` on two large frames — output is the **product** of row counts; 10k × 10k is 100M rows. Filter first
- Calling `merge_asof` on unsorted input — it does not raise, it returns wrong matches. Sort both sides on the `on` key
- Using `merge_asof(..., by=...)` and forgetting the `by` column must also be present in both frames
- Omitting `tolerance` in `merge_asof`, which happily matches a quote from three weeks earlier
- Expecting `.compare()` to work on differently-shaped frames — it requires identical labels; reindex first
""")]

# ── §6 GroupBy ────────────────────────────────────────────────────────────────
S6 = [md("""
**Three §6 gaps:** returning the winning *row* per group, grouping by an index level, and grouping by time and category together.
"""), code("""
# idxmax / idxmin inside a groupby — the row, not just the value
best = df.loc[df.groupby('category')['revenue'].idxmax()]      # top row per category
df.groupby('category')['revenue'].idxmin()                     # index labels only

# Equivalent alternatives, and when each wins
df.sort_values('revenue').groupby('category').tail(1)          # ties -> keeps one
df.loc[df.groupby('category')['revenue'].transform('max') == df['revenue']]  # keeps ALL ties

# Group by an INDEX level rather than a column
mi = df.set_index(['region', 'date'])
mi.groupby(level='region')['sales'].sum()
mi.groupby(level=[0, 1]).size()
mi.groupby(level='region', group_keys=False).apply(lambda g: g.head(2))

# pd.Grouper — time buckets combined with ordinary columns
df.groupby(pd.Grouper(key='order_date', freq='ME'))['amount'].sum()   # month end
df.groupby([pd.Grouper(key='order_date', freq='W'), 'region'])['amount'].sum()
df.set_index('order_date').groupby([pd.Grouper(freq='QE'), 'channel']).size()
"""), md("""
**Getting the top row per group:**

| Approach | Ties | Speed | Note |
| :--- | :--- | :--- | :--- |
| `loc[groupby.idxmax()]` | keeps one arbitrarily | fast | fails on all-NaN groups |
| `sort_values().groupby().tail(1)` | keeps one | medium | order is explicit |
| `transform('max') == col` | keeps **all** ties | medium | usually the correct answer |
| `groupby().apply(nlargest)` | configurable | slowest | avoid on large frames |

**Common mistakes:**
- Using `.idxmax()` on a group that is entirely NaN — raises `ValueError`; filter empty groups first
- Assuming `idxmax` breaks ties fairly; it returns the first occurrence
- `pd.Grouper(freq='M')` is deprecated in favour of `'ME'` (month **end**) in pandas 2.2+; `'MS'` is month start
- Grouping by a datetime *column* without `pd.Grouper` — you get one group per distinct timestamp
""")]

# ── §7 Window ─────────────────────────────────────────────────────────────────
S7 = [md("""
**Five §7 gaps.** The important one is time-based rolling: `rolling(7)` means *seven rows*, `rolling('7D')` means *seven days* — they differ whenever the series has gaps.
"""), code("""
# ── Row-based vs time-based windows ──────────────────────────────────────────
df['ma7_rows'] = df['sales'].rolling(7).mean()          # last 7 ROWS
df = df.set_index('date').sort_index()
df['ma7_days'] = df['sales'].rolling('7D').mean()       # last 7 DAYS (gap-aware)
# Time-based rolling REQUIRES a sorted DatetimeIndex.

# Centred window — aligns the label to the middle, not the right edge
df['smooth'] = df['sales'].rolling(7, center=True).mean()

# Exponentially weighted — recent points weighted more, no fixed cutoff
df['ewm'] = df['sales'].ewm(span=7).mean()              # span ~ "like a 7-period MA"
df['sales'].ewm(halflife='3D', times=df.index).mean()   # time-aware decay
df['sales'].ewm(alpha=0.3, adjust=False).mean()         # explicit smoothing factor

# Custom function over a window
df['rng'] = df['sales'].rolling(7).apply(lambda w: w.max() - w.min(), raw=True)
# raw=True passes a numpy array instead of a Series -> substantially faster

# Rolling relationships between two series
df['corr30'] = df['price'].rolling(30).corr(df['index_price'])
df['cov30']  = df['price'].rolling(30).cov(df['index_price'])
df['beta30'] = df['cov30'] / df['index_price'].rolling(30).var()

# Guard against thin windows at the start of the series
df['sales'].rolling(7, min_periods=7).mean()            # NaN until 7 obs exist
"""), md("""
**Window types:**

| | Window | Weights | Use when |
| :--- | :--- | :--- | :--- |
| `rolling(7)` | fixed 7 rows | equal | regular, gapless series |
| `rolling('7D')` | fixed 7 days | equal | irregular timestamps or missing days |
| `expanding()` | all prior rows | equal | cumulative-to-date metrics |
| `ewm(span=7)` | all prior rows | decaying | recent data matters more |

**Common mistakes:**
- Using `rolling(7)` on a series with missing days and calling it a 7-day average — it is a 7-*observation* average
- Time-based rolling on an unsorted or non-datetime index — raises, or silently misaligns
- `center=True` in a production feature — it looks at future rows and leaks; only for visual smoothing
- Leaving `raw=False` in `rolling().apply()`, which builds a Series per window and is far slower
- Comparing `ewm(span=n)` to `rolling(n)` as if equivalent — `ewm` never fully forgets old data
""")]

# ── §8 Date & Time ────────────────────────────────────────────────────────────
S8 = [md("""
**Four §8 gaps:** surviving unparseable dates, calendar-aware offsets, business days, and the boundary properties.
"""), code("""
# errors='coerce' — turn unparseable values into NaT instead of raising
df['date'] = pd.to_datetime(df['raw_date'], errors='coerce')
df['raw_date'][df['date'].isna()].unique()      # inspect what failed to parse

pd.to_datetime(df['d'], format='%d/%m/%Y', errors='coerce')   # explicit format: fastest + safest
pd.to_datetime(df['d'], format='mixed', dayfirst=True)        # genuinely mixed formats
pd.to_datetime(df['epoch'], unit='s')                         # unix seconds
pd.to_datetime(df['epoch_ms'], unit='ms')                     # unix milliseconds

# ── Offsets: calendar-aware, unlike Timedelta ────────────────────────────────
from pandas.tseries.offsets import MonthEnd, MonthBegin, BDay, QuarterEnd, Week

df['date'] + MonthEnd(0)                  # snap to end of THIS month
df['date'] + MonthEnd(1)                  # end of next month
df['date'] - MonthBegin(1)                # start of this month
df['date'] + BDay(3)                      # 3 business days later (skips weekends)
df['date'] + QuarterEnd(0)                # end of current quarter
df['date'] + pd.DateOffset(months=1)      # calendar month: Jan 31 -> Feb 28/29

# Timedelta is a FIXED duration and does not know about calendars
df['date'] + pd.Timedelta(days=30)        # always exactly 30 days

# ── Business days ────────────────────────────────────────────────────────────
pd.bdate_range('2024-01-01', '2024-01-31')                # weekdays only
pd.bdate_range('2024-01-01', periods=10, freq='C',        # custom calendar
               holidays=['2024-01-15'])
np.busday_count('2024-01-01', '2024-02-01')               # count business days

# ── Boundary properties ──────────────────────────────────────────────────────
d = df['date'].dt
d.is_month_end, d.is_month_start
d.is_quarter_end, d.is_year_end
d.days_in_month, d.dayofweek, d.dayofyear, d.isocalendar().week
df[d.dayofweek >= 5]                      # weekend rows
"""), md("""
**`Timedelta` vs `DateOffset` vs offset objects:**

| | Jan 31 + 1 month | Skips weekends? |
| :--- | :--- | :--- |
| `Timedelta(days=30)` | Mar 1 | ❌ |
| `DateOffset(months=1)` | Feb 29 | ❌ |
| `MonthEnd(1)` | Feb 29 | ❌ |
| `BDay(1)` | next weekday | ✅ |

**Common mistakes:**
- `errors='coerce'` without checking how many rows became `NaT` — silent data loss
- Letting `to_datetime` infer the format on a large column; it is slow and can flip day/month
- Using `Timedelta(days=30)` for "one month later" — wrong for every month that is not 30 days
- Assuming `MonthEnd(1)` from a date already at month end moves one month; use `MonthEnd(0)` to snap, `MonthEnd(1)` to advance
""")]

# ── §9 String ops ─────────────────────────────────────────────────────────────
S9 = [md("""
**Six §9 gaps**, ending with the one interviews actually probe: matching names that are *nearly* equal.
"""), code("""
# Concatenate across columns / down a Series
df['full'] = df['first'].str.cat(df['last'], sep=' ', na_rep='')
df['tags'].str.cat(sep=', ')                      # collapse a whole Series to one string

# One-hot encode a delimited column in a single call
df['tags'].str.get_dummies(sep='|')               # 'a|b' -> columns a, b

# Padding and zero-fill (IDs, zip codes)
df['zip'].astype(str).str.zfill(5)                # '123' -> '00123'
df['code'].str.pad(10, side='right', fillchar='.')

# Find every match, not just the first
df['text'].str.findall(r'\\d+')                    # list of all numbers per row
df['text'].str.count(r'\\berror\\b')                # occurrences per row
df['text'].str.extractall(r'(\\d+)')               # long-format frame of all matches

# Unicode normalisation — the invisible cause of failed joins
df['name'].str.normalize('NFKD')                                  # decompose accents
(df['name'].str.normalize('NFKD')
           .str.encode('ascii', 'ignore').str.decode('utf-8'))     # strip them entirely
df['name'].str.replace('\\u00a0', ' ', regex=False)                 # non-breaking space

# ── Fuzzy matching — when keys are close but not equal ──────────────────────
import difflib
difflib.get_close_matches('Jonh Smith', known_names, n=3, cutoff=0.8)

df['match'] = df['raw_name'].apply(
    lambda x: (difflib.get_close_matches(x, known_names, n=1, cutoff=0.85) or [None])[0]
)
# For large joins prefer rapidfuzz (C++ backed, orders of magnitude faster):
#   from rapidfuzz import process
#   process.extractOne(x, known_names, score_cutoff=85)
"""), md("""
**Extraction methods:**

| Method | Returns | Matches |
| :--- | :--- | :--- |
| `str.extract` | DataFrame, one row per input | first only |
| `str.extractall` | long DataFrame, MultiIndex | all |
| `str.findall` | Series of lists | all |
| `str.count` | Series of ints | all (count only) |

**Common mistakes:**
- Normalising case but not unicode — `"José"` and `"José"` can differ by encoding and never join
- `str.zfill` on a numeric column; cast to `str` first or it fails silently
- `str.get_dummies` on a high-cardinality column, which explodes the column count
- Fuzzy-matching in a loop over a large frame — `difflib` is O(n·m) and will hang; use `rapidfuzz`
- Forgetting `na=False` in `str.contains` when NaNs are present, which yields NaN and breaks the mask
""")]

# ── Interview Q&A ─────────────────────────────────────────────────────────────
QA = md("""
---
## Interview Q&A

**Q: You need every user paired with every day in January, including days with no activity. How?**
A: A cross join to build the scaffold, then a left join for the data.
`users[['user_id']].merge(pd.DataFrame({'date': pd.date_range(...)}), how='cross')`
gives the complete grid, and left-joining events onto it with a `fillna(0)` yields
zero rows rather than missing rows. Doing it the other way round — grouping the
events — silently drops days nobody was active, which is exactly the bug in most
retention calculations.

**Q: Join each trade to the prevailing quote at that moment.**
A: `merge_asof`, not `merge`. An equality join fails because timestamps rarely
match exactly. I'd sort both frames on time, use `by='symbol'` so matching stays
within an instrument, and set a `tolerance` so a stale quote from hours earlier
does not get attached silently.

**Q: `.loc` vs `.iloc` vs `.at`?**
A: `.loc` is label-based and its slices are end-inclusive; `.iloc` is positional
and end-exclusive; `.at` is a single-value fast path. The practical trap is that
`df.loc[0:5]` returns six rows while `df.iloc[0:5]` returns five.

**Q: An integer column turned into floats after a merge. Why?**
A: Unmatched rows introduced `np.nan`, which is a float, so the column widened to
`float64`. Using the nullable `Int64` dtype keeps the values integral and stores
missing as `pd.NA`.

**Q: Difference between `rolling(7)` and `rolling('7D')`?**
A: Seven rows versus seven calendar days. They agree only when the series has one
row per day with no gaps. On event data with missing days, `rolling(7)` quietly
spans a different amount of time for every row.

**Q: How do you check what changed between two versions of a table?**
A: `before.compare(after)` for cell-level differences on identically-labelled
frames, and `Index.difference` in both directions for rows added and removed. For
a quick summary I'd also merge with `indicator=True` and count the `_merge` column.

### Gotchas
- Chained indexing (`df['a'][mask] = x`) may write to a temporary copy; use `df.loc[mask, 'a'] = x`
- `merge` silently produces a row explosion on many-to-many keys — pass `validate='1:m'` to make it raise instead
- `how='cross'` has no key to deduplicate on, so it multiplies row counts; always check `len(left) * len(right)` first
- Comparing float columns with `==` after arithmetic; use `np.isclose`
- `inplace=True` is not faster and is being phased out — prefer reassignment
""")


INSERTS = [(32, S9), (29, S8), (26, S7), (23, S6), (19, S5), (14, S4), (9, S3), (6, S2)]


def main() -> None:
    nb = json.loads(TARGET.read_text(encoding="utf-8"))
    before = len(nb["cells"])
    if any("## Interview Q&A" in "".join(c["source"]) for c in nb["cells"]):
        print("  already updated, skipping")
        return

    # descending order so earlier indices stay valid
    for pos, cells in INSERTS:
        nb["cells"][pos + 1:pos + 1] = cells

    # Q&A immediately before the final Decision Guide
    dg = next(i for i, c in enumerate(nb["cells"])
              if "# Decision Guide" in "".join(c["source"]))
    nb["cells"].insert(dg, QA)

    TARGET.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  Reference_Pandas: {before} -> {len(nb['cells'])} cells "
          f"(+{len(nb['cells']) - before})")


if __name__ == "__main__":
    main()
