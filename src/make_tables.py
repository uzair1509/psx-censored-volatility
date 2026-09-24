"""Every table and figure in the paper, generated from the result files.

    python -m src.make_tables          # uses cached DiD fits if present
    python -m src.make_tables --refit  # recompute DiD window fits (~3-5 min)

Inputs : data/summary.csv, data/universe.csv, data/panel.parquet,
         results/garch_fits.csv, results/garch_fits_t.csv, results/mc_results.csv
Outputs: results/tables/T*.md + .csv, results/figures/F*.png
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr, ttest_ind, mannwhitneyu

from src.band_regimes import regime_table, floor_threshold
from src.fit_garch import filtered_symbols

TAB, FIG, CACHE = Path("results/tables"), Path("results/figures"), Path("results/did_windows")


def save(df, name, note=""):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=False)
    fmt = lambda v: f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    (TAB / f"{name}.md").write_text("\n".join(lines) + (f"\n\n{note}\n" if note else "\n"), encoding="utf-8")
    print(f"\n== {name} ==\n" + "\n".join(lines) + (f"\n{note}" if note else ""))


# ---------- T1 band regimes ----------
def t1():
    t = regime_table()
    t["floor binds below Rs"] = t["pct"].map(lambda p: round(floor_threshold(p), 2))
    t["pct"] = (t["pct"] * 100).round(1).astype(str) + "%"
    t["start"] = t["start"].dt.date.astype(str)
    save(t[["start", "pct", "floor binds below Rs", "source"]], "T1_band_regimes")


# ---------- T2 sample ----------
def t2(summ, keep):
    k = summ[summ.symbol.isin(keep)]
    rows = [
        ("listed instruments", 1028), ("equities after type/sector filters", 659),
        ("with data", len(summ)), ("final sample (liquidity + pref filters)", len(k)),
        ("stock-days (final)", int(k.n.sum())),
        ("median days per stock", int(k.n.median())),
        ("median censoring rate", round(k.censor_rate.median(), 4)),
        ("median share of floor-bound days", round(k.floor_share.median(), 4)),
        ("median breaches per stock", float(k.breaches.median())),
    ]
    save(pd.DataFrame(rows, columns=["item", "value"]), "T2_sample")


# ---------- T3 H1: floor vs censoring ----------
def t3(panel):
    pk = panel[~panel.breach]
    g = pk.groupby(["symbol", "floor_binds"])["censored"].mean().unstack().dropna()
    g.columns = ["pct_days", "floor_days"]
    k, n = int((g.floor_days < g.pct_days).sum()), len(g)
    pooled = pk.groupby("floor_binds")["censored"].mean()
    rows = [
        ("stocks with both day types", n),
        ("stocks censoring less on floor days", f"{k} ({k/n:.1%})"),
        ("sign test p", f"{binomtest(k, n).pvalue:.2e}"),
        ("median within-stock gap (floor - pct)", round((g.floor_days - g.pct_days).median(), 4)),
        ("pooled censoring rate, pct days", round(pooled[False], 4)),
        ("pooled censoring rate, floor days", round(pooled[True], 4)),
    ]
    save(pd.DataFrame(rows, columns=["H1", "value"]), "T3_floor_censoring",
         "Pooled rates are descriptive only: stock-days are not independent.")


# ---------- T4 alpha understatement ----------
def alpha_row(d, label):
    rel = d.alpha_gap / d.alpha_n
    k = int((d.alpha_gap > 0).sum())
    rs = spearmanr(d.censor_rate, d.alpha_gap)
    return {"spec": label, "n": len(d), "gap>0": f"{k/len(d):.1%}",
            "sign p": f"{binomtest(k, len(d)).pvalue:.1e}",
            "median alpha naive": round(d.alpha_n.median(), 3),
            "median alpha latent": round(d.alpha_l.median(), 3),
            "median understatement": f"{rel.median():.1%}",
            "IQR": f"{rel.quantile(.25):.1%} to {rel.quantile(.75):.1%}",
            "Spearman(gap, censoring)": round(rs.statistic, 2)}


def t4():
    fn = pd.read_csv("results/garch_fits.csv"); fn = fn[fn.error.fillna("") == ""]
    ft = pd.read_csv("results/garch_fits_t.csv"); ft = ft[ft.error.fillna("") == ""]
    degenerate = (ft[["alpha_n", "alpha_l"]].min(axis=1) < 1e-4) | (ft[["nu_n", "nu_l"]].max(axis=1) > 190)
    rows = [
        alpha_row(ft[~degenerate], "Student-t (main)"),
        alpha_row(ft[~ft.at_boundary.astype(bool)], "Student-t, strict"),
        alpha_row(ft, "Student-t, all fits"),
        alpha_row(fn[~fn.at_boundary.astype(bool)], "Normal"),
    ]
    save(pd.DataFrame(rows), "T4_alpha_understatement",
         f"Median nu (naive t): {ft.nu_n.median():.1f}. Main spec drops alpha~0 and nu-at-cap fits only.")
    return fn, ft[~degenerate]


# ---------- T5 band changes: DiD ----------
def window_fits(name, a, b, panel, syms, refit, min_obs=350):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{name}.csv"
    if f.exists() and not refit:
        return pd.read_csv(f, index_col="symbol")
    from joblib import Parallel, delayed
    from src import fit_garch as mod
    sub = panel[(panel.date >= a) & (panel.date <= b) & panel.symbol.isin(syms)]
    groups = [(s, g) for s, g in sub.groupby("symbol") if len(g) >= min_obs]
    out = pd.DataFrame(Parallel(n_jobs=-2)(delayed(mod.fit_one)(s, g) for s, g in groups))
    out = out[out.error.fillna("") == ""]
    out = out[~out.at_boundary.astype(bool)].set_index("symbol")
    out = out[["censor_rate", "alpha_gap"]].join(sub.groupby("symbol")["floor_binds"].mean())
    out.to_csv(f)
    return out


def did(label, pre, post, panel, syms, refit):
    A = window_fits(f"{label}_pre", *pre, panel, syms, refit)
    B = window_fits(f"{label}_post", *post, panel, syms, refit)
    d = (B[["censor_rate", "alpha_gap"]] - A[["censor_rate", "alpha_gap"]]).dropna()
    fs = pd.concat([A.floor_binds.rename("a"), B.floor_binds.rename("b")], axis=1).dropna()
    grp = pd.Series(np.select([(fs.a < .1) & (fs.b < .1), (fs.a > .9) & (fs.b > .9)],
                              ["treated", "control"], "mixed"), index=fs.index)
    d = d.join(grp.rename("group")).query("group != 'mixed'").dropna()
    res = {"event": label, "n treated": int((d.group == "treated").sum()),
           "n control": int((d.group == "control").sum())}
    for c, short in [("censor_rate", "cens"), ("alpha_gap", "gap")]:
        t, k = d.loc[d.group == "treated", c], d.loc[d.group == "control", c]
        est = t.mean() - k.mean()                         # 2x2 DiD on first differences
        se = np.sqrt(t.var(ddof=1) / len(t) + k.var(ddof=1) / len(k))
        res[f"DiD {short}"] = est
        res[f"SE {short}"] = se
        res[f"p {short} (Welch)"] = ttest_ind(t, k, equal_var=False).pvalue
        res[f"p {short} (MW)"] = mannwhitneyu(t, k).pvalue
    return res


def t5(panel, syms, refit):
    ev = [("placebo_2018", ("2016-09-27", "2018-05-31"), ("2018-06-01", "2020-01-19")),
          ("band_2020",    ("2016-09-27", "2020-01-19"), ("2020-03-20", "2024-05-26")),
          ("band_2024",    ("2020-03-20", "2024-05-26"), ("2024-07-22", "2026-12-31"))]
    df = pd.DataFrame([did(l, a, b, panel, syms, refit) for l, a, b in ev])
    save(df, "T5_did")
    # Rambachan-Roth relative-magnitudes bound (single pre-period): theta +/- M*|delta_pre|
    pl, ev20 = df.set_index("event").loc["placebo_2018"], df.set_index("event").loc["band_2020"]
    rows = []
    for short in ["cens", "gap"]:
        th, dp = ev20[f"DiD {short}"], abs(pl[f"DiD {short}"])
        rows.append({"outcome": short, "DiD 2020": th, "|placebo|": dp,
                     "bound M=1 low": th - dp, "bound M=1 high": th + dp,
                     "breakdown M*": abs(th) / dp if dp > 0 else np.inf})
    save(pd.DataFrame(rows), "T5b_rambachan_roth",
         "Relative-magnitudes bound (Rambachan & Roth 2023) using the 2018 placebo as the "
         "pre-period violation. Effect is robust while the post-period violation is < M* x placebo.")


# ---------- T6 Monte Carlo + F1 ----------
def t6_f1():
    p = Path("results/mc_results.csv")
    if not p.exists():
        print("\n(skip T6/F1: run python -m src.simulate first)"); return
    from src.simulate import summarise, A
    s = summarise(pd.read_csv(p))
    save(s.round(4), "T6_monte_carlo", f"True alpha = {A}.")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, dist in zip(axes, ["normal", "t"]):
        d = s[s.dist == dist].sort_values("censor_rate")
        for m, mk in [("naive", "o"), ("partial", "s"), ("latent", "^")]:
            ax.plot(d.censor_rate * 100, d[f"{m}_bias"], marker=mk, label=m)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(f"{'Normal' if dist == 'normal' else 'Student-t'} innovations")
        ax.set_xlabel("censoring rate (%)")
    axes[0].set_ylabel("bias in alpha (true 0.15)"); axes[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "F1_monte_carlo_bias.png", dpi=200); plt.close(fig)


# ---------- F2 gap vs censoring ----------
def f2(fn, ft):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.scatter(ft.censor_rate * 100, ft.alpha_gap / ft.alpha_n * 100, s=8, alpha=.6, label="Student-t")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("censoring rate (%)"); ax.set_ylabel("alpha understatement (%)")
    ax.set_ylim(-50, 150); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "F2_gap_vs_censoring.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", action="store_true")
    a = ap.parse_args()
    summ = pd.read_csv("data/summary.csv")
    syms = filtered_symbols()
    panel = pd.read_parquet("data/panel.parquet")
    panel = panel[panel.symbol.isin(syms)]
    t1(); t2(summ, syms); t3(panel)
    fn, ft = t4()
    t5(panel, syms, a.refit)
    t6_f1(); f2(fn, ft)
    print(f"\nTables -> {TAB}/   Figures -> {FIG}/")
