"""H2: naive vs censored-aware GARCH(1,1) across the filtered PSX universe.

THREE models per stock:
  naive   : ignores limits (Gaussian GARCH on observed returns)
  partial : your universe.ipynb model -- censored likelihood, but the variance
            recursion feeds the CLIPPED return. Monte Carlo (true a=0.15, 4.5%
            censoring, 8 seeds): naive 0.123, partial 0.163, latent 0.147.
            partial over-corrects: it inflates alpha to make up for clipped r^2.
  latent  : censored likelihood AND recursion uses E[r*^2 | censored] from the
            truncated normal -> ~unbiased. THIS is the headline model.

Model = the one in universe.ipynb (your likelihood), with 4 changes:
  1. variance recursion via scipy lfilter (same maths, ~100x faster)
  2. breach days (ex-dates / bad data): excluded from the likelihood and
     their return set to 0 in the recursion, so fake jumps don't hit sigma^2
  3. log(1-cdf) -> norm.logsf (numerically stable in the far tail)
  4. censored fit warm-starts from the naive solution + default start; best kept

    python -m src.fit_garch            # all stocks, parallel
    python -m src.fit_garch --limit 10 # smoke test
Output: results/garch_fits.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.stats import norm
import math

try:  # optional speed-up; pure Python is fine (~1s per stock)
    from numba import njit
except ImportError:
    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]

X0 = np.array([1e-4, 0.10, 0.85])
OPTS = dict(method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-6})


def garch_variance_path(r, omega, alpha, beta):
    """sigma2[t] = omega + alpha*r[t-1]^2 + beta*sigma2[t-1], sigma2[0] = var(r)."""
    s0 = np.var(r)
    x = omega + alpha * r[:-1] ** 2
    y = lfilter([1.0], [1.0, -beta], x, zi=[beta * s0])[0]
    return np.concatenate(([s0], y))


def _valid(p):
    w, a, b = p
    return w > 0 and a >= 0 and b >= 0 and a + b < 1


def naive_nll(p, r, use):
    if not _valid(p):
        return 1e10
    s = np.sqrt(garch_variance_path(r, *p))
    return -norm.logpdf(r[use], scale=s[use]).sum()


def censored_nll(p, r, r_up, r_lo, up, lo, use):
    if not _valid(p):
        return 1e10
    s = np.sqrt(garch_variance_path(r, *p))
    mid = use & ~up & ~lo
    ll = norm.logpdf(r[mid], scale=s[mid]).sum()
    ll += norm.logsf(r_up[up & use] / s[up & use]).sum()   # P(latent >= upper)
    ll += norm.logcdf(r_lo[lo & use] / s[lo & use]).sum()  # P(latent <= lower)
    return -ll


@njit(cache=True)
def _latent_nll(w, a, b, r, ru, rl, up, lo, use):
    n = len(r); s2 = np.var(r); ll = 0.0
    c = math.sqrt(2 * math.pi)
    for t in range(n):
        s = math.sqrt(s2)
        if use[t]:
            if up[t]:
                z = ru[t] / s
                tail = max(0.5 * math.erfc(z / math.sqrt(2)), 1e-300)
                ll += math.log(tail)
                e2 = s2 * (1 + z * math.exp(-0.5 * z * z) / c / tail)   # E[r*^2 | r* >= upper]
            elif lo[t]:
                z = rl[t] / s
                tail = max(0.5 * math.erfc(-z / math.sqrt(2)), 1e-300)
                ll += math.log(tail)
                e2 = s2 * (1 - z * math.exp(-0.5 * z * z) / c / tail)   # E[r*^2 | r* <= lower]
            else:
                ll += -0.5 * math.log(2 * math.pi * s2) - 0.5 * r[t] * r[t] / s2
                e2 = r[t] * r[t]
        else:
            e2 = s2  # breach day: no information, use expected value
        s2 = w + a * e2 + b * s2
    return -ll


def latent_nll(p, r, r_up, r_lo, up, lo, use):
    if not _valid(p):
        return 1e10
    return _latent_nll(p[0], p[1], p[2], r, r_up, r_lo, up, lo, use)


def prep(g: pd.DataFrame):
    g = g.sort_values("date")
    prev = g["upper"] - g["band"]
    r = np.log(g["close"] / prev).to_numpy()
    r_up = np.log(g["upper"] / prev).to_numpy()
    r_lo = np.log(g["lower"] / prev).to_numpy()
    breach = g["breach"].to_numpy()
    use = ~breach & np.isfinite(r)
    r = np.where(use, r, 0.0)
    return r, r_up, r_lo, g["censored_up"].to_numpy(), g["censored_dn"].to_numpy(), use


def fit_one(sym, g):
    try:
        r, r_up, r_lo, up, lo, use = prep(g)
        n = minimize(naive_nll, X0, args=(r, use), **OPTS)
        starts = [n.x, X0] if _valid(n.x) else [X0]
        args = (r, r_up, r_lo, up, lo, use)
        c = min((minimize(censored_nll, s, args=args, **OPTS) for s in starts), key=lambda z: z.fun)
        L = min((minimize(latent_nll, s, args=args, **OPTS) for s in starts), key=lambda z: z.fun)
        return {
            "symbol": sym, "n": int(use.sum()),
            "censor_rate": float((up | lo)[use].mean()),
            "omega_n": n.x[0], "alpha_n": n.x[1], "beta_n": n.x[2], "ll_n": -n.fun, "ok_n": bool(n.success),
            "omega_c": c.x[0], "alpha_c": c.x[1], "beta_c": c.x[2], "ll_c": -c.fun, "ok_c": bool(c.success),
            "omega_l": L.x[0], "alpha_l": L.x[1], "beta_l": L.x[2], "ll_l": -L.fun, "ok_l": bool(L.success),
            "alpha_gap": L.x[1] - n.x[1],          # headline: latent - naive
            "alpha_gap_partial": c.x[1] - n.x[1],  # comparable to the preprint
            "at_boundary": bool(min(n.x[1], L.x[1]) < 1e-4 or max(n.x[1] + n.x[2], L.x[1] + L.x[2]) > 0.999),
            "error": "",
        }
    except Exception as e:
        return {"symbol": sym, "error": repr(e)[:200]}


# Stable-band windows (phase-in months dropped). Tighter band -> more censoring
# -> the naive bias should be larger. Same stocks, policy-driven variation.
REGIMES = {
    "5.0%":  ("2016-01-01", "2020-01-19"),
    "7.5%":  ("2020-03-20", "2024-05-26"),
    "10.0%": ("2024-07-22", "2099-01-01"),
}
MIN_OBS = 400


def fit_by_regime(sym, g):
    rows = []
    for name, (a, b) in REGIMES.items():
        w = g[(g.date >= a) & (g.date <= b)]
        if len(w) < MIN_OBS:
            continue
        r = fit_one(sym, w)
        r["regime"] = name
        rows.append(r)
    return rows


def filtered_symbols():
    summ = pd.read_csv("data/summary.csv")
    uni = pd.read_csv("data/universe.csv")
    pref = set(uni.loc[uni["name"].str.contains("PREF", case=False, na=False), "symbol"])
    pref |= {s for s in summ.symbol if s.endswith(("CPS", "PS"))}
    keep = summ[~summ.symbol.isin(pref) & (summ.n >= 500) & (summ.zero_vol_share <= 0.20)]
    return keep.symbol.tolist()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=-2)  # all cores but one
    ap.add_argument("--by-regime", action="store_true")
    a = ap.parse_args()
    syms = filtered_symbols()[: a.limit]
    panel = pd.read_parquet("data/panel.parquet")
    panel = panel[panel.symbol.isin(syms)]
    groups = [(s, g) for s, g in panel.groupby("symbol")]
    print(f"fitting {len(groups)} stocks...")
    Path("results").mkdir(exist_ok=True)
    if a.by_regime:
        nested = Parallel(n_jobs=a.jobs, verbose=5)(delayed(fit_by_regime)(s, g) for s, g in groups)
        out = pd.DataFrame([r for rows in nested for r in rows])
        out.to_csv("results/garch_fits_regime.csv", index=False)
    else:
        rows = Parallel(n_jobs=a.jobs, verbose=5)(delayed(fit_one)(s, g) for s, g in groups)
        out = pd.DataFrame(rows)
        out.to_csv("results/garch_fits.csv", index=False)
    err = out["error"].fillna("").ne("").sum()
    print(f"\ndone: {len(out)} | errors: {err} | "
          f"converged naive+latent: {(out.ok_n & out.ok_l).sum()} | at boundary: {out.at_boundary.sum()}")
