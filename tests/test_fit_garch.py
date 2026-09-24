import numpy as np
from src.fit_garch import garch_variance_path, naive_nll, censored_nll, latent_nll, X0
from scipy.optimize import minimize


def _loop(r, w, a, b):
    s = np.zeros(len(r)); s[0] = np.var(r)
    for t in range(1, len(r)):
        s[t] = w + a * r[t-1]**2 + b * s[t-1]
    return s


def test_lfilter_matches_original_loop():
    r = np.random.default_rng(0).normal(0, 0.02, 500)
    assert np.allclose(garch_variance_path(r, 2e-5, 0.12, 0.83), _loop(r, 2e-5, 0.12, 0.83))


def test_latent_model_recovers_alpha_on_simulated_limits():
    rng = np.random.default_rng(1)
    w, a, b, T, L = 2e-5, 0.15, 0.80, 3000, 0.04
    s2 = w / (1 - a - b); lat = np.zeros(T)
    for t in range(T):
        lat[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = w + a * lat[t]**2 + b * s2
    r = np.clip(lat, -L, L)
    up, lo, use = r >= L, r <= -L, np.ones(T, bool)
    ru, rl = np.full(T, L), np.full(T, -L)
    opts = dict(method="Nelder-Mead", options={"maxiter": 4000})
    n = minimize(naive_nll, X0, args=(r, use), **opts).x
    l = minimize(latent_nll, n, args=(r, ru, rl, up, lo, use), **opts).x
    assert n[1] < a                      # naive attenuates alpha
    assert abs(l[1] - a) < 0.03, (n, l)  # latent model recovers it
