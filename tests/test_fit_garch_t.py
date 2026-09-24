import math
import numpy as np
from scipy.integrate import quad
from scipy.stats import t as tdist
from scipy.optimize import minimize
from src.fit_garch_t import trunc_m2, naive_t_nll, latent_t_nll, X0, OPTS


def test_truncated_second_moment_matches_numerical_integral():
    for nu in (3.5, 6.0, 20.0):
        for c in (-1.0, 0.5, 2.0, 3.5):
            num = quad(lambda x: x * x * tdist.pdf(x, nu), c, np.inf)[0] / tdist.sf(c, nu)
            assert math.isclose(trunc_m2(c, nu)[0], num, rel_tol=1e-6)


def test_latent_t_recovers_alpha_on_simulated_fat_tailed_limits():
    rng = np.random.default_rng(7)
    w, a, b, nu, T, L = 2e-5, 0.15, 0.80, 6.0, 2500, 0.04
    s2 = w / (1 - a - b); x = np.zeros(T)
    k = math.sqrt((nu - 2) / nu)
    for i in range(T):
        x[i] = math.sqrt(s2) * k * rng.standard_t(nu)
        s2 = w + a * x[i] ** 2 + b * s2
    r = np.clip(x, -L, L)
    up, lo, use = r >= L, r <= -L, np.ones(T, bool)
    ru, rl = np.full(T, L), np.full(T, -L)
    n = minimize(naive_t_nll, X0, args=(r, use), **OPTS).x
    l = minimize(latent_t_nll, n, args=(r, ru, rl, up, lo, use), **OPTS).x
    assert n[1] < a and abs(l[1] - a) < 0.04, (n, l)
