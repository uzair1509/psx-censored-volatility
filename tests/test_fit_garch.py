import numpy as np
from src.fit_garch import (garch_variance_path, garch_variance_path_breach,
                            naive_nll, censored_nll, latent_nll, X0)
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


def test_breach_day_uses_variance_forecast_not_zero():
    # A breach day (ex-date jump, bad tick, etc.) should feed the model's OWN
    # current variance forecast into next period's recursion (e2 = s2[t]),
    # the same convention _latent_nll already uses on excluded days --
    # NOT a zeroed return (which silently understates next-day variance
    # relative to what the latent model does on the same day).
    w, a, b = 2e-5, 0.15, 0.80
    r = np.array([0.01, 0.5, -0.02, 0.03, 0.015])  # index 1 is a breach (huge, spurious)
    breach = np.array([False, True, False, False, False])
    s2 = garch_variance_path_breach(r, w, a, b, breach)
    # s2[2] should be computed from e2 = s2[1] (forecast), not r[1]**2 = 0.25
    expected_s2_1 = np.var(r)  # s2[0]
    expected_s2_1 = w + a * (r[0] ** 2) + b * expected_s2_1  # s2[1], breach[0]=False
    expected_s2_2 = w + a * expected_s2_1 + b * expected_s2_1  # breach[1]=True -> e2=s2[1]
    assert np.isclose(s2[2], expected_s2_2)
    assert not np.isclose(s2[2], w + a * (r[1] ** 2) + b * expected_s2_1)  # not the old zero/raw-shock behavior


def test_no_breach_matches_fast_lfilter_path():
    r = np.random.default_rng(3).normal(0, 0.02, 300)
    breach = np.zeros(len(r), dtype=bool)
    assert np.allclose(garch_variance_path_breach(r, 2e-5, 0.12, 0.83, breach),
                        garch_variance_path(r, 2e-5, 0.12, 0.83))
