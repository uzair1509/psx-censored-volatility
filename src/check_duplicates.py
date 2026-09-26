"""Duplicate/renamed-security check.

Run this on YOUR machine (needs data/universe.csv and data/panel.parquet,
which aren't in the sandbox this was written in):

    python -m src.check_duplicates

What it does:
  1. Flags symbol pairs whose price HISTORIES are suspiciously complementary
     (little/no date overlap, and the second one's start is close to the
     first one's end) -- the signature of a ticker rename/re-listing, e.g.
     BYCO -> CNERGY (2020) or ICI -> LCI (2023).
  2. Flags pairs with near-identical price levels/returns on overlapping
     dates (the signature of a straight duplicate row, not a rename).
  3. Does NOT auto-merge anything -- renames need a human to confirm the
     correct symbol history and check for adjustment-factor breaks (splits,
     bonus issues) at the switch date before splicing two histories into one
     continuous series. This script just tells you where to look.

Output: prints candidate pairs; if any look real, decide per pair whether to
(a) merge into one continuous symbol history, (b) keep only the currently-
listed ticker and drop the dead one, or (c) keep both but exclude from any
"N distinct companies" count and footnote it.
"""
import pandas as pd
import numpy as np

def main():
    uni = pd.read_csv("data/universe.csv")
    panel = pd.read_parquet("data/panel.parquet")

    ranges = panel.groupby("symbol")["date"].agg(["min", "max", "count"])
    syms = ranges.index.tolist()

    print(f"{len(syms)} symbols with data.\n")
    print("=== Candidate rename/re-listing pairs (little date overlap, "
          "back-to-back date ranges) ===")
    hits = []
    for i, a in enumerate(syms):
        ra = ranges.loc[a]
        for b in syms[i + 1:]:
            rb = ranges.loc[b]
            # one ends close to (within 40 trading days) where the other starts
            gap1 = (rb["min"] - ra["max"]).days if rb["min"] >= ra["max"] else None
            gap2 = (ra["min"] - rb["max"]).days if ra["min"] >= rb["max"] else None
            gap = gap1 if gap1 is not None else gap2
            if gap is not None and gap <= 60:
                hits.append((a, b, ra["min"], ra["max"], rb["min"], rb["max"], gap))
    for a, b, amin, amax, bmin, bmax, gap in sorted(hits, key=lambda x: x[-1]):
        print(f"  {a:12s} [{amin.date()} - {amax.date()}]  ->(gap {gap:>3}d)->  "
              f"{b:12s} [{bmin.date()} - {bmax.date()}]")
    if not hits:
        print("  none found")

    print("\n=== Candidate straight duplicates (overlapping dates, "
          "near-identical closes) ===")
    dup_hits = []
    piv = panel.pivot_table(index="date", columns="symbol", values="close")
    cols = piv.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            both = piv[[a, b]].dropna()
            if len(both) < 30:
                continue
            corr = both[a].corr(both[b])
            if corr > 0.999:
                rel_diff = ((both[a] - both[b]).abs() / both[a]).median()
                if rel_diff < 0.02:
                    dup_hits.append((a, b, len(both), corr, rel_diff))
    for a, b, n, corr, rd in sorted(dup_hits, key=lambda x: -x[3]):
        print(f"  {a:12s} vs {b:12s}  n_overlap={n:4d}  corr={corr:.4f}  "
              f"median_rel_diff={rd:.2%}")
    if not dup_hits:
        print("  none found")

    print(f"\nTotal candidate pairs to review by hand: {len(hits) + len(dup_hits)}")
    print("Known real renames to sanity-check against: BYCO->CNERGY (2020), "
          "ICI->LCI (2023). If those two show up above, the detector is working.")


if __name__ == "__main__":
    main()
