"""Robustness: Student-t innovations for all three models (naive / partial / latent).

Question answered: is the naive alpha understatement just fat tails in disguise?
If the latent-t vs naive-t gap survives, the bias is from the limits, not tails.

r* = s * X,  X ~ t_nu,  s = sigma * sqrt((nu-2)/nu)   (so Var(r*) = sigma^2)
Truncated second moment of a standard t (derived by parts), c = limit / s:
  E[X^2 ; X > c] = nu/(nu-2) * S_{nu-2}(c*sqrt((nu-2)/nu)) + c*f_nu(c)*(nu+c^2)/(nu-1)
  E[X^2 | X > c] = that / S_nu(c);  lower tail by symmetry (c -> -c).

    python -m src.fit_garch_t --limit 10
    python -m src.fit_garch_t
Output: results/garch_fits_t.csv
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.special import stdtr, gammaln
from scipy.stats import t as tdist

from src.fit_garch import garch_variance_path, prep, filtered_symbols

X0 = np.array([1e-4, 0.10, 0.85, 8.0])
OPTS = dict(method="Nelder-Mead", options={"maxiter": 6000, "xatol": 1e-7, "fatol": 1e-6})
NU_MAX = 200.0


def _valid(p):
    w, a, b, nu = p
    return w > 0 and a >= 0 and b >= 0 and a + b < 1 and 2.05 < nu < NU_MAX


def _scale(sig, nu):
    return sig * math.sqrt((nu - 2) / nu) if np.isscalar(sig) else sig * np.sqrt((nu - 2) / nu)


def naive_t_nll(p, r, use):
    if not _valid(p):
        return 1e10
    s = _scale(np.sqrt(garch_variance_path(r, *p[:3])), p[3])
    return -tdist.logpdf(r[use], p[3], scale=s[use]).sum()


def partial_t_nll(p, r, ru, rl, up, lo, use):
    if not _valid(p):
        return 1e10
    nu = p[3]
    s = _scale(np.sqrt(garch_variance_path(r, *p[:3])), nu)
    mid = use & ~up & ~lo
    ll = tdist.logpdf(r[mid], nu, scale=s[mid]).sum()
    ll += tdist.logsf(ru[up & use] / s[up & use], nu).sum()
    ll += tdist.logcdf(rl[lo & use] / s[lo & use], nu).sum()
    return -ll


def trunc_m2(c, nu):
    """E[X^2 | X > c] for standard t_nu."""
    f = math.exp(gammaln((nu + 1) / 2) - gammaln(nu / 2)) / math.sqrt(nu * math.pi) \
        * (1 + c * c / nu) ** (-(nu + 1) / 2)
    tail = max(1 - stdtr(nu, c), 1e-300)
    num = nu / (nu - 2) * (1 - stdtr(nu - 2, c * math.sqrt((nu - 2) / nu))) + c * f * (nu + c * c) / (nu - 1)
    return num / tail, tail


def latent_t_nll(p, r, ru, rl, up, lo, use):
    if not _valid(p):
        return 1e10
    w, a, b, nu = p
    k = math.sqrt((nu - 2) / nu)
    const = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * math.log(nu * math.pi)
    s2 = float(np.var(r)); ll = 0.0
    for t in range(len(r)):
        s = math.sqrt(s2) * k
        if use[t]:
            if up[t]:
                m2, tail = trunc_m2(ru[t] / s, nu); ll += math.log(tail); e2 = s * s * m2
            elif lo[t]:
                m2, tail = trunc_m2(-rl[t] / s, nu); ll += math.log(tail); e2 = s * s * m2
            else:
                z = r[t] / s
                ll += const - math.log(s) - (nu + 1) / 2 * math.log1p(z * z / nu)
                e2 = r[t] * r[t]
        else:
            e2 = s2
        s2 = w + a * e2 + b * s2
    return -ll


def fit_one_t(sym, g):
    try:
        r, ru, rl, up, lo, use = prep(g)
        args = (r, ru, rl, up, lo, use)
        n = minimize(naive_t_nll, X0, args=(r, use), **OPTS)
        starts = [n.x, X0] if _valid(n.x) else [X0]
        c = min((minimize(partial_t_nll, s, args=args, **OPTS) for s in starts), key=lambda z: z.fun)
        L = min((minimize(latent_t_nll, s, args=args, **OPTS) for s in starts), key=lambda z: z.fun)
        return {
            "symbol": sym, "n": int(use.sum()), "censor_rate": float((up | lo)[use].mean()),
            "alpha_n": n.x[1], "beta_n": n.x[2], "nu_n": n.x[3], "ok_n": bool(n.success),
            "alpha_c": c.x[1], "nu_c": c.x[3], "ok_c": bool(c.success),
            "alpha_l": L.x[1], "beta_l": L.x[2], "nu_l": L.x[3], "ok_l": bool(L.success),
            "alpha_gap": L.x[1] - n.x[1], "alpha_gap_partial": c.x[1] - n.x[1],
            "at_boundary": bool(min(n.x[1], L.x[1]) < 1e-4 or max(n.x[1] + n.x[2], L.x[1] + L.x[2]) > 0.999
                                or max(n.x[3], L.x[3]) > NU_MAX * 0.95),
            "error": "",
        }
    except Exception as e:
        return {"symbol": sym, "error": repr(e)[:200]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=-2)
    a = ap.parse_args()
    syms = filtered_symbols()[: a.limit]
    panel = pd.read_parquet("data/panel.parquet")
    panel = panel[panel.symbol.isin(syms)]
    groups = [(s, g) for s, g in panel.groupby("symbol")]
    print(f"fitting {len(groups)} stocks (Student-t; slower, pure Python)...")
    # Import by module path so workers unpickle src.fit_garch_t.*, not __main__.*
    # (scipy ufuncs like gammaln/stdtr can't be found in the workers' __main__ on Windows)
    from src import fit_garch_t as mod
    rows = Parallel(n_jobs=a.jobs, verbose=5)(delayed(mod.fit_one_t)(s, g) for s, g in groups)
    out = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    out.to_csv("results/garch_fits_t.csv", index=False)
    err = out["error"].fillna("").ne("").sum()
    print(f"\ndone: {len(out)} | errors: {err} | converged naive+latent: {(out.ok_n & out.ok_l).sum()} "
          f"| at boundary: {out.at_boundary.sum()}")
