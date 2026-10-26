# psx-censored-volatility

Analysing the effect of PSX price limit censoring on GARCH volatility estimates. I build a max likelihood GARCH estimator that accounts for the limits, and compare it with a naive GARCH. Monte Carlo simulations measure how the parameter bias grows as more returns get censored.

## The problem

The Pakistan Stock Exchange (PSX) has a daily price limit. A stock cannot move more than the band in one day. The band is the larger of a percentage of the last close and Re. 1.

So on a limit day, the return I see is cut off. The real move was bigger. A normal GARCH does not know this. It treats the cut return as the real one, and the volatility looks smaller than it is.

The band also changed over time. It was 5%, went up to 7.5% in 2020, and to 10% in 2024, in steps.

## What I do

I fit three GARCH(1,1) models on every stock:

- **naive**: ignores the limits.
- **partial**: uses a censored likelihood, but the variance still uses the cut return. This over-corrects.
- **latent**: uses a censored likelihood, and replaces the cut squared return with its expected value given that the limit was hit. This is the main model. It is almost unbiased in simulation.

I also repeat everything with Student-t errors, to check that the result is not just fat tails.

## Results

**Monte Carlo** (true alpha = 0.15, normal errors, 50 reps per row):

| censoring rate | naive alpha | partial alpha | latent alpha |
|---|---|---|---|
| 11.6% | 0.110 | 0.192 | 0.156 |
| 4.7% | 0.127 | 0.171 | 0.153 |
| 2.0% | 0.139 | 0.161 | 0.153 |

The naive model gets worse as censoring goes up. The latent model stays close to 0.15.

**Real PSX data:** 449 stocks, about 834,000 stock-days. The median stock has 4.8% of days censored.

With Student-t errors, the latent alpha is higher than the naive alpha in 65% of stocks. The median stock is 8% higher. The gap is bigger for stocks with more censoring. With normal errors the gap is larger: 97% of stocks, median 21% higher.

All tables are in `results/tables/` and the figures are in `results/figures/`.

## Package

The estimator (naive / censored / latent) is published as a separate Python package on PyPI:

```bash
pip install mtgarch
```

PyPI: https://pypi.org/project/mtgarch/

## Data

The raw PSX price data is not included in this repo, and I do not redistribute it. The files in `results/` are my own fitted estimates and summary tables, not PSX data.

To reproduce the results, get the PSX price data yourself under PSX's terms (for a licence, see the PSX Market Data team) and put it in `data/raw/`. `src/build_dataset.py` shows the format the other scripts expect.

## Code

- `src/band_regimes.py`: the PSX price band rules and the censoring flags.
- `src/build_dataset.py`: builds the full PSX dataset.
- `src/fit_garch.py`: fits naive, partial and latent models on every stock.
- `src/fit_garch_t.py`: the same with Student-t errors.
- `src/simulate.py`: the Monte Carlo.
- `src/make_tables.py`: makes every table and figure from the result files.
- `tests/`: tests for the band rules and the fitting code.

## Run it

```bash
python -m src.build_dataset
python -m src.fit_garch
python -m src.fit_garch_t
python -m src.simulate --reps 50
python -m src.make_tables
```

Add `--limit 10` to the build and fit scripts to test on 10 stocks first. Run the tests with `pytest tests`.

## License

MIT
