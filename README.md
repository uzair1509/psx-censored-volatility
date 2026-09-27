# psx-censored-volatility
Analysing the effect of PSX price limit censoring on GARCH volatility estimates. Developing a max likelihood GARCH estimator and comparing with a naive GARCH specification. Monte Carlo simulations are used to measure parameter bias as censor intensity increases.

## Package

The censored-GARCH estimator (naive / censored / latent) developed here is published as a standalone Python package on PyPI:

```bash
pip install mtgarch
```

PyPI: https://pypi.org/project/mtgarch/
