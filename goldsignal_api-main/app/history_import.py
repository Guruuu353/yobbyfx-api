"""
Import candle history from a CSV file (typically exported from MetaTrader 5) into the persistent candle store.

Why: Yahoo only serves ~60 days of 5m / 15m / 30m candles. MT5 keeps years. Import them once and the chart's range
buttons (3M / 6M / 1Y), the engine's past-signal replay and the backtest can all use the longer history.

Accepted layouts (comma, tab or semicolon separated; header optional):
  MT5 History Center export     2024.01.02,00:00,open,high,low,close,tickvol,vol
  MT5 terminal / MQL export     <DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>
  one datetime column           2024-01-02 00:00:00,open,high,low,close[,volume]
  unix seconds / milliseconds   1704153600,open,high,low,close[,volume]
Times are the BROKER server time: `tz_offset` is its offset from UTC in hours (GMT+3 -> 3).
"""
from __future__ import annotations

import io
import re
import time

import numpy as np
import pandas as pd

from . import candle_store
from .candle_store import STEP

MAX_BYTES = 80 * 1024 * 1024


class ImportError_(ValueError):
    pass


def _read(text: str) -> pd.DataFrame:
    head = text[:4000]
    first = next((ln for ln in head.splitlines() if ln.strip()), "")
    sep = max((",", "\t", ";"), key=first.count)
    if first.count(sep) == 0:
        sep = r"\s+"
    df = pd.read_csv(io.StringIO(text), sep=sep, header=None, dtype=str, engine="python", skip_blank_lines=True)
    if df.empty:
        raise ImportError_("The file is empty.")
    # drop a header row (anything whose first cell has letters, e.g. <DATE>, Date, time)
    if re.search(r"[A-Za-z]", str(df.iloc[0, 0])):
        df = df.iloc[1:].reset_index(drop=True)
    return df


def parse(text: str, tz_offset: float = 0.0) -> pd.DataFrame:
    """-> DataFrame indexed by UTC open time with Open/High/Low/Close/Volume."""
    df = _read(text)
    ncol = df.shape[1]
    if ncol < 5:
        raise ImportError_(f"Expected at least 5 columns (time, open, high, low, close); found {ncol}.")
    c0 = df.iloc[:, 0].astype(str).str.strip()
    c1 = df.iloc[:, 1].astype(str).str.strip()
    if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", str(c1.iloc[0])):             # separate DATE and TIME columns
        stamp = pd.to_datetime(c0.str.replace(".", "-", regex=False) + " " + c1, errors="coerce")
        px = df.iloc[:, 2:]
    elif re.fullmatch(r"\d{9,13}", str(c0.iloc[0])):                          # epoch
        n = pd.to_numeric(c0, errors="coerce")
        stamp = pd.to_datetime(n, unit="ms" if n.iloc[0] > 1e11 else "s", errors="coerce")
        px = df.iloc[:, 1:]
    else:                                                                      # one datetime column
        stamp = pd.to_datetime(c0.str.replace(".", "-", regex=False), errors="coerce")
        px = df.iloc[:, 1:]
    if px.shape[1] < 4:
        raise ImportError_("Could not find open, high, low and close columns after the time.")
    out = pd.DataFrame({"Open": pd.to_numeric(px.iloc[:, 0], errors="coerce"),
                        "High": pd.to_numeric(px.iloc[:, 1], errors="coerce"),
                        "Low": pd.to_numeric(px.iloc[:, 2], errors="coerce"),
                        "Close": pd.to_numeric(px.iloc[:, 3], errors="coerce")})
    vol = None
    if px.shape[1] > 4:
        tick = pd.to_numeric(px.iloc[:, 4], errors="coerce").fillna(0)
        real = pd.to_numeric(px.iloc[:, 5], errors="coerce").fillna(0) if px.shape[1] > 5 else None
        vol = real if (real is not None and real.sum() > 0) else tick
    out["Volume"] = vol if vol is not None else 0.0
    out.index = stamp
    out = out[out.index.notna() & out["Close"].notna() & out["Open"].notna() & out["High"].notna() & out["Low"].notna()]
    if out.empty:
        raise ImportError_("No readable rows. Check the file has date/time and open, high, low, close columns.")
    out.index = (out.index - pd.Timedelta(hours=float(tz_offset))).tz_localize("UTC")
    out.index.name = "time"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    bad = (out["High"] < out[["Open", "Close", "Low"]].max(axis=1) - 1e-9) | (out["Low"] > out[["Open", "Close", "High"]].min(axis=1) + 1e-9)
    return out[~bad]


def import_text(symbol: str, interval: str, text: str, tz_offset: float = 0.0) -> dict:
    if interval not in STEP:
        raise ImportError_(f"Unsupported timeframe {interval!r}.")
    df = parse(text, tz_offset)
    step = STEP[interval]
    ts = df.index.as_unit("s").asi8
    # the file's spacing must match the timeframe chosen, or the store would be filled with the wrong candles
    if len(ts) > 20:
        gaps = np.diff(ts)
        typical = int(np.median(gaps))
        if typical != step:
            raise ImportError_(f"These candles are {typical // 60} minutes apart but you chose {interval}. Pick the matching timeframe.")
    now = time.time()
    keep = (ts + step) <= now + 1                        # closed candles only
    keep &= ts >= now - candle_store.RETENTION_DAYS.get(interval, 9999) * 86400
    sub = df[keep]
    if sub.empty:
        raise ImportError_("Nothing to import (all rows are in the future or older than the retention window).")
    before = candle_store.bounds(symbol, interval)
    n = candle_store.write_rows(symbol, interval, sub)
    after = candle_store.bounds(symbol, interval)
    return {"symbol": symbol, "interval": interval, "rows_read": int(len(df)), "rows_stored": int(n),
            "file_from": int(sub.index.as_unit("s").asi8[0]), "file_to": int(sub.index.as_unit("s").asi8[-1]),
            "store_from": after[0] if after else None, "store_to": after[1] if after else None,
            "store_rows": after[2] if after else 0,
            "added_rows": (after[2] - (before[2] if before else 0)) if after else 0}
