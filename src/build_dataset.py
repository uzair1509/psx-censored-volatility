"""One-shot full-exchange PSX build. Resumable: rerun after any crash.

    python -m src.build_dataset            # fetch + flag everything
    python -m src.build_dataset --limit 20 # smoke test on 20 symbols first

Outputs
    data/universe.csv          symbols kept + why others were dropped
    data/raw/{SYM}.parquet      raw OHLCV per symbol (skip if exists)
    data/fetch_failures.csv     every failed symbol + error
    data/panel.parquet          all symbols, with band/censoring flags
    data/summary.csv            one row per symbol: n, start, censor rate, breaches, ...
"""
import argparse
import time
from pathlib import Path

import pandas as pd
from psxdata import PSXClient
from psxdata.exceptions import PSXRateLimitError

from src.band_regimes import attach_limits, infer_tick, SAMPLE_START

DATA = Path("data")
RAW = DATA / "raw"
# Non-equity instruments only. Leasing cos, inv. banks, REITs = listed equity, kept.
EXCLUDE_SECTOR_WORDS = ("MODARABA", "MUTUAL FUND", "BOND", "BILL", "SUKUK",
                        "TFC", "EXCHANGE TRADED")


def build_universe(client: PSXClient) -> list[str]:
    sym = client.symbols()
    sym["drop_reason"] = ""
    for col in ("is_etf", "is_debt", "is_gem"):
        if col in sym:
            sym.loc[sym[col].fillna(False).astype(bool) & (sym["drop_reason"] == ""),
                    "drop_reason"] = col
    sec = sym["sector_name"].fillna("").str.upper()
    for w in EXCLUDE_SECTOR_WORDS:
        sym.loc[sec.str.contains(w) & (sym["drop_reason"] == ""), "drop_reason"] = f"sector:{w}"
    sym.loc[(sec == "") & (sym["drop_reason"] == ""), "drop_reason"] = "no_sector"
    sym.to_csv(DATA / "universe.csv", index=False)
    print(sym["sector_name"].value_counts().to_string())  # eyeball once
    kept = sym.loc[sym["drop_reason"] == "", "symbol"].tolist()
    print(f"\nuniverse: {len(sym)} listed -> {len(kept)} kept")
    return kept


def fetch_all(client: PSXClient, symbols: list[str]) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    fails = []
    todo = [s for s in symbols if not (RAW / f"{s}.parquet").exists()]
    print(f"fetching {len(todo)} (cached: {len(symbols) - len(todo)})")
    for i, s in enumerate(todo, 1):
        for attempt in range(3):
            try:
                df = client.stocks(s)
                if df.empty:
                    fails.append((s, "empty"))
                else:
                    df.to_parquet(RAW / f"{s}.parquet", index=False)
                break
            except PSXRateLimitError:
                time.sleep(30 * (attempt + 1))  # library doesn't retry 429
            except Exception as e:  # log everything, never drop silently
                fails.append((s, repr(e)[:200]))
                break
        if i % 50 == 0:
            print(f"  {i}/{len(todo)}")
    if fails:
        pd.DataFrame(fails, columns=["symbol", "error"]).to_csv(
            DATA / "fetch_failures.csv", index=False)
    print(f"failures: {len(fails)}")


def flag_all(symbols: list[str]) -> None:
    frames, rows = [], []
    for s in symbols:
        p = RAW / f"{s}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        # Keep OHLC-anomaly rows: the bad field is usually high/low, close is fine,
        # and dropping rows would make the next day's prev_close wrong.
        n_anom = int(df["is_anomaly"].fillna(False).sum())
        if len(df) < 3:
            continue
        f = attach_limits(df)
        f.insert(0, "symbol", s)
        frames.append(f)
        rows.append({
            "symbol": s, "n": len(f),
            "start": f["date"].min(), "end": f["date"].max(),
            "tick_inferred": infer_tick(f["close"]),
            "n_anomaly": n_anom,
            "zero_vol_share": (f["volume"] == 0).mean(),
            "censor_rate": f["censored"].mean(),
            "censor_rate_floor": f.loc[f["floor_binds"], "censored"].mean(),
            "censor_rate_pct": f.loc[~f["floor_binds"], "censored"].mean(),
            "floor_share": f["floor_binds"].mean(),
            "breaches": int(f["breach"].sum()),
        })
    pd.concat(frames).to_parquet(DATA / "panel.parquet", index=False)
    summ = pd.DataFrame(rows)
    summ.to_csv(DATA / "summary.csv", index=False)
    print(summ.describe().T.to_string())
    print(f"\nearliest data: {summ['start'].min().date()}  (sample start {SAMPLE_START.date()})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    c = PSXClient()
    syms = build_universe(c)[: a.limit]
    fetch_all(c, syms)
    flag_all(syms)
