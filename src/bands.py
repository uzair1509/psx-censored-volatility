# src/bands.py — PSX scrip-level circuit breakers, 2015-2026
# Rule: limit_rs = max(pct * prev_close, FLOOR)
#
# SOURCING STATUS:
#   2020 steps  - CONFIRMED against PSX notice (Jan 2020) + independent sources
#   2024 steps  - CONFIRMED against PSX notice (May 2024) + independent sources
#   2020-2024 flat at 7.5% - CONFIRMED, no intermediate change (independent May 2024 source)
#   pre-2020    - PROVISIONAL. post-2008 5% regime, no primary source yet
#   Mar-Aug 20  - UNVERIFIED whether ad hoc emergency measures overrode this
#                 schedule during the COVID crash days specifically
#
# The floor is why cheap stocks get wider effective bands: at 5%,
# anything under Rs.20 is governed by the Re.1 floor, not the percentage.

import numpy as np
import pandas as pd

FLOOR = 1.0

SCHEDULE = [
  ("2015-01-01", 0.050),  # provisional
  ("2020-01-20", 0.055),
  ("2020-02-04", 0.060),
  ("2020-02-19", 0.065),
  ("2020-03-05", 0.070),
  ("2020-03-20", 0.075),
  ("2024-05-27", 0.080),
  ("2024-06-10", 0.085),
  ("2024-06-24", 0.090),
  ("2024-07-08", 0.095),
  ("2024-07-22", 0.100),
]

_STEPS = pd.DataFrame(SCHEDULE, columns=["date", "pct"])
_STEPS["date"] = pd.to_datetime(_STEPS["date"])
_STEPS = _STEPS.sort_values("date").reset_index(drop=True)

def pct_on(dates):
  """Band percentage in force on each date. Step function, forward-filled."""
  idx = pd.DatetimeIndex(dates)
  s = pd.Series(_STEPS["pct"].values, index=_STEPS["date"])
  return s.reindex(s.index.union(idx)).ffill().reindex(idx)

def limits(df):
  """Given OHLCV with a date index, return per-day upper/lower price limits.

  Limits are set off the PREVIOUS close, so row 0 is NaN by construction.
  """
  prev = df["close"].shift(1)
  pct = pct_on(df.index).values
  band_rs = np.maximum(pct * prev.values, FLOOR)

  out = pd.DataFrame(index=df.index)
  out["prev_close"] = prev
  out["pct"] = pct
  out["band_rs"] = band_rs
  out["upper"] = prev.values + band_rs
  out["lower"] = prev.values - band_rs
  out["floor_binds"] = band_rs > (pct * prev.values) + 1e-9
  return out

def censoring_status(df, lim, tol=0.01):
  """
  Classify each day's close relative to that day's band.
  Returns a Series: 'upper' (hit/exceeded upper limit), 'lower'
  (hit/exceeded lower limit), or 'none' (normal, uncensored close).
  First row is NaN - no prev_close to band against.

  tol accounts for price rounding (PSX quotes to the paisa).
  """
  close = df["close"]
  status = pd.Series("none", index=df.index, dtype=object)
  status[close >= lim["upper"] - tol] = "upper"
  status[close <= lim["lower"] + tol] = "lower"
  status[lim["prev_close"].isna()] = np.nan
  return status

def crossover_price(pct):
  """Below this price the Re.1 floor governs instead of the percentage."""
  return FLOOR / pct

if __name__ == "__main__":
  for _, r in _STEPS.iterrows():
    print(f"{r['date'].date()}  {r['pct']:.1%}  floor binds below Rs.{crossover_price(r['pct']):.2f}")