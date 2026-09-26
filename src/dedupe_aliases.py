"""Cluster near-identical price series (same company under multiple ticker
aliases -- almost certainly a psxdata characteristic where client.stocks()
returns a company's full history regardless of which historical ticker you
query) and keep exactly one symbol per cluster.

Run on your machine (needs data/panel.parquet):

    python -m src.dedupe_aliases

Writes:
    data/alias_map.csv   every dropped symbol -> the canonical symbol kept
    data/alias_drops.txt  just the dropped symbols, one per line (for
                           excluding them in filtered_symbols())

This does NOT touch panel.parquet or your results -- it only decides who to
drop. Wire the drop list into src/fit_garch.py's filtered_symbols() (see the
note printed at the end) before refitting.
"""
import pandas as pd
import numpy as np
from pathlib import Path

CORR_THRESH = 0.999
RELDIFF_THRESH = 0.02
MIN_OVERLAP = 30


def find_pairs(panel: pd.DataFrame):
    piv = panel.pivot_table(index="date", columns="symbol", values="close")
    cols = piv.columns.tolist()
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            both = piv[[a, b]].dropna()
            if len(both) < MIN_OVERLAP:
                continue
            corr = both[a].corr(both[b])
            if corr > CORR_THRESH:
                rel_diff = ((both[a] - both[b]).abs() / both[a]).median()
                if rel_diff < RELDIFF_THRESH:
                    pairs.append((a, b))
    return pairs


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def main():
    panel = pd.read_parquet("data/panel.parquet")
    n_by_symbol = panel.groupby("symbol").size()

    pairs = find_pairs(panel)
    print(f"{len(pairs)} near-identical pairs found.\n")

    uf = UnionFind()
    for a, b in pairs:
        uf.union(a, b)

    clusters = {}
    for a, b in pairs:
        r = uf.find(a)
        clusters.setdefault(r, set()).update([a, b])

    rows = []
    keep_all, drop_all = [], []
    for members in clusters.values():
        members = sorted(members, key=lambda s: -n_by_symbol.get(s, 0))
        canonical = members[0]  # longest history = kept
        keep_all.append(canonical)
        for m in members[1:]:
            drop_all.append(m)
            rows.append({"dropped_symbol": m, "canonical_symbol": canonical,
                         "n_dropped": int(n_by_symbol.get(m, 0)),
                         "n_canonical": int(n_by_symbol.get(canonical, 0))})
        print(f"cluster: {members}  -> keeping {canonical}")

    out = pd.DataFrame(rows)
    out.to_csv("data/alias_map.csv", index=False)
    Path("data/alias_drops.txt").write_text("\n".join(sorted(drop_all)), encoding="utf-8")

    print(f"\n{len(clusters)} clusters, {len(drop_all)} symbols to drop as duplicate aliases.")
    print("Wrote data/alias_map.csv and data/alias_drops.txt.")
    print("\nNext: in src/fit_garch.py's filtered_symbols(), add:")
    print('    dropped = set(open("data/alias_drops.txt").read().split())')
    print('    keep = keep[~keep.symbol.isin(dropped)]')
    print("then rerun fit_garch / fit_garch_t / make_tables --refit.")


if __name__ == "__main__":
    main()
