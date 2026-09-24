"""Monte Carlo: bias of naive / partial / latent GARCH(1,1) under daily price limits.

DGP: GARCH(1,1), omega=2e-5, alpha=0.15, beta=0.80, symmetric limit +/-L on
daily returns (observed return = clip(latent, -L, L)). Innovations normal or
Student-t (nu=5). Three limit widths give low / mid / high censoring.
Each model is fitted with the SAME distribution as the DGP.

    python -m src.simulate --reps 50
Output: results/mc_results.csv (+ summary printed)
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize

from src import fit_garch as N
from src import fit_garch_t as T

W, A, B, NU = 2e-5, 0.15, 0.80, 5.0
LIMITS = [0.05, 0.04, 0.03]      # ~2%, ~5%, ~13% censoring under normal
LEN = 2500                       # ~10 years of trading days, like the PSX sample


def simulate(dist, L, seed, n=LEN):
    rng = np.random.default_rng(seed)
    s2 = W / (1 - A - B)
    k = math.sqrt((NU - 2) / NU)
    x = np.empty(n)
    for t in range(n):
        z = rng.standard_normal() if dist == "normal" else k * rng.standard_t(NU)
        x[t] = math.sqrt(s2) * z
        s2 = W + A * x[t] ** 2 + B * s2
    r = np.clip(x, -L, L)
    return r, r >= L, r <= -L


def one(dist, L, seed):
    r, up, lo = simulate(dist, L, seed)
    use = np.ones(len(r), bool)
    ru, rl = np.full(len(r), L), np.full(len(r), -L)
    args = (r, ru, rl, up, lo, use)
    M = N if dist == "normal" else T
    naive, partial, latent = (M.naive_nll, M.censored_nll, M.latent_nll) if dist == "normal" \
        else (T.naive_t_nll, T.partial_t_nll, T.latent_t_nll)
    n = minimize(naive, M.X0, args=(r, use), **M.OPTS).x
    start = n if M._valid(n) else M.X0
    c = minimize(partial, start, args=args, **M.OPTS).x
    l = minimize(latent, start, args=args, **M.OPTS).x
    return {"dist": dist, "L": L, "seed": seed, "censor_rate": float((up | lo).mean()),
            "alpha_naive": n[1], "alpha_partial": c[1], "alpha_latent": l[1]}


def summarise(df):
    g = df.groupby(["dist", "L"])
    out = g["censor_rate"].mean().rename("censor_rate").to_frame()
    for m in ["naive", "partial", "latent"]:
        out[f"{m}_mean"] = g[f"alpha_{m}"].mean()
        out[f"{m}_bias"] = out[f"{m}_mean"] - A
        out[f"{m}_rmse"] = g[f"alpha_{m}"].apply(lambda s: float(np.sqrt(((s - A) ** 2).mean())))
    out["reps"] = g.size()
    return out.reset_index()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--jobs", type=int, default=-2)
    a = ap.parse_args()
    from src import simulate as mod  # workers unpickle src.simulate.*, not __main__ (Windows)
    tasks = [(d, L, s) for d in ("normal", "t") for L in LIMITS for s in range(a.reps)]
    print(f"{len(tasks)} simulated fits (true alpha = {A})...")
    rows = Parallel(n_jobs=a.jobs, verbose=5)(delayed(mod.one)(*t) for t in tasks)
    df = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/mc_results.csv", index=False)
    print(summarise(df).round(4).to_string(index=False))
