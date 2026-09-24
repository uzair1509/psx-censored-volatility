"""PSX scrip-level circuit-breaker regimes + censoring flags.

Band = max(pct * LDCP, Re.1). Sources in REGIMES. Unverified rows flagged.
"""
import numpy as np
import pandas as pd

ABS_FLOOR = 1.0  # Re.1
PSX_TICK = 0.01  # fixed; infer_tick() is a diagnostic only (unreliable on thin stocks)

# (start_date, pct, source, verified). Each regime runs until the next start.
REGIMES = [
    ("2010-01-01", 0.050, "5% since 2008 cut (Tribune/Profit Jan-2020); PSX DPS data starts 2016-09", True),
    ("2020-01-20", 0.055, "PSX notice 13-Dec-2019", True),
    ("2020-02-04", 0.060, "PSX notice 13-Dec-2019", True),
    ("2020-02-19", 0.065, "PSX notice 13-Dec-2019", True),
    ("2020-03-05", 0.070, "PSX notice 13-Dec-2019", True),
    ("2020-03-20", 0.075, "PSX notice 13-Dec-2019", True),
    ("2024-05-27", 0.080, "PSX notice May-2024", True),
    ("2024-06-10", 0.085, "PSX notice May-2024", True),
    ("2024-06-24", 0.090, "PSX notice May-2024", True),
    ("2024-07-08", 0.095, "PSX notice May-2024", True),
    ("2024-07-22", 0.100, "PSX notice May-2024", True),
]
SAMPLE_START = pd.Timestamp(REGIMES[0][0])


def regime_table() -> pd.DataFrame:
    df = pd.DataFrame(REGIMES, columns=["start", "pct", "source", "verified"])
    df["start"] = pd.to_datetime(df["start"])
    return df.sort_values("start").reset_index(drop=True)


def floor_threshold(pct: float) -> float:
    """Price below which the Re.1 floor is the binding band."""
    return ABS_FLOOR / pct


def infer_tick(close: pd.Series) -> float:
    """Smallest price increment observed (robust to float noise)."""
    d = np.abs(np.diff(np.round(close.dropna().to_numpy(), 4)))
    d = d[d > 1e-6]
    return float(np.round(d.min(), 4)) if d.size else 0.01


def attach_limits(px: pd.DataFrame, tick: float = PSX_TICK) -> pd.DataFrame:
    """px: one symbol, columns ['date','close', ...].

    Adds: pct, band, upper, lower, floor_binds, censored_up, censored_dn,
    censored, breach. `breach` = close outside the band -> corporate action
    or bad data; those rows must NOT count as censored or enter the GARCH.
    """
    out = px.copy()
    out["date"] = pd.to_datetime(out["date"])
    # psxdata returns rows out of order and occasionally duplicated
    out = out.sort_values("date").drop_duplicates("date", keep="last")
    out = out[out["date"] >= SAMPLE_START].reset_index(drop=True)

    out = pd.merge_asof(out, regime_table()[["start", "pct"]],
                        left_on="date", right_on="start", direction="backward")
    out = out.drop(columns="start")

    prev = out["close"].shift(1)
    pct_band = prev * out["pct"]
    out["band"] = pct_band.clip(lower=ABS_FLOOR)
    out["floor_binds"] = pct_band < ABS_FLOOR
    out["upper"] = prev + out["band"]
    out["lower"] = (prev - out["band"]).clip(lower=tick)

    # 1-tick tolerance: covers PSX rounding the limit up/down/nearest.
    tol = tick + 1e-9
    out["censored_up"] = (out["close"] - out["upper"]).abs() <= tol
    out["censored_dn"] = (out["close"] - out["lower"]).abs() <= tol
    out["breach"] = (out["close"] > out["upper"] + tol) | (out["close"] < out["lower"] - tol)
    out.loc[out["breach"], ["censored_up", "censored_dn"]] = False
    out["censored"] = out["censored_up"] | out["censored_dn"]
    out["tick"] = tick
    return out.iloc[1:].reset_index(drop=True)
