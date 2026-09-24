#month by month scrape

import time
from datetime import date, datetime
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup

URL = "https://dps.psx.com.pk/historical"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
UA = {"User-Agent": "Mozilla/5.0 (research; PSX censoring study)"}
PAUSE = 0.4  

def months(start, end):
  out, y, m = [], start.year, start.month
  while (y, m) <= (end.year, end.month):
    out.append((y, m))
    y, m = (y + 1, 1) if m == 12 else (y, m + 1)
  return out

def fetch_month(session, ticker, year, month):
  r = session.post(URL, data={"month": month, "year": year, "symbol": ticker},
                   headers=UA, timeout=30)
  r.raise_for_status()
  soup = BeautifulSoup(r.text, "html.parser")
  heads = [h.getText().strip().lower() for h in soup.select("th")]
  if not heads:
    return pd.DataFrame()

  rows = []
  for tr in soup.select("tr"):
    cols = [td.getText().strip() for td in tr.select("td")]
    if len(cols) == len(heads):
      rows.append(dict(zip(heads, cols)))
  return pd.DataFrame(rows)

def clean(df):
  if df.empty:
    return df
  df = df.rename(columns={"date": "date"})
  df["date"] = pd.to_datetime(df["date"], format="%b %d, %Y")
  for c in ["open", "high", "low", "close", "volume"]:
    df[c] = df[c].str.replace(",", "", regex=False).astype(float)
  return df.set_index("date").sort_index()

def load(ticker, start=date(2015, 1, 1), end=date(2026, 8, 31), refresh=False):
  """Returns unadjusted daily OHLCV. Caches each month; only misses hit the network."""
  cache = RAW / ticker
  cache.mkdir(parents=True, exist_ok=True)
  frames = []

  with requests.Session() as s:
    for y, m in months(start, end):
      f = cache / f"{y}-{m:02d}.csv"
      if f.exists() and not refresh:
        frames.append(pd.read_csv(f, parse_dates=["date"], index_col="date"))
        continue
      try:
        df = clean(fetch_month(s, ticker, y, m))
      except Exception as e:
        print(f"  warn {ticker} {y}-{m:02d}: {e}")
        continue
      if not df.empty:
        df.to_csv(f)
        frames.append(df)
      time.sleep(PAUSE)

  if not frames:
    raise RuntimeError(f"no data for {ticker}")

  out = pd.concat(frames).sort_index()
  out = out[~out.index.duplicated(keep="last")]
  return out.loc[str(start):str(end)]

if __name__ == "__main__":
  df = load("OGDC", date(2024, 1, 1), date(2024, 3, 31))
  print(df)
  print(df.dtypes)