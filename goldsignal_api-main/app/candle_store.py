"""
Persistent candle store.

Why: Yahoo only serves a short window of intraday history (7 days of 1m, ~60 days of 5m / 15m / 30m). The old
disk cache was a CSV that was OVERWRITTEN with the latest window on every download, so nothing was ever
accumulated - scrolling back on a chart ran out after a few weeks, and a backtest could never see more than that.

This stores every closed candle we ever download in SQLite, keyed by (symbol, interval, open-time), and hands the
accumulated history back merged with the fresh download (the fresh data wins where they overlap). History grows
by itself, for as long as the app runs. Prices are stored RAW (before the MT5 calibration offset).

Set GOLDSIGNAL_CANDLE_DB to move the file. Demo data is never stored (it is regenerated on every start).
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time

import pandas as pd

CACHE_DIR = os.path.join(os.environ.get("GOLDSIGNAL_DATA_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "cache")
STEP = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "1d": 86400, "1wk": 604800}
# how long we keep each resolution (days) - bounded so the file stays small
RETENTION_DAYS = {"1m": 45, "5m": 800, "15m": 1600, "30m": 2400, "1h": 3650, "1d": 9999, "1wk": 9999}

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None
_path: str | None = None


def db_path() -> str:
    return os.environ.get("GOLDSIGNAL_CANDLE_DB") or os.path.join(CACHE_DIR, "candles.db")


def _db() -> sqlite3.Connection:
    global _conn, _path
    path = db_path()
    with _lock:
        if _conn is None or _path != path:
            if _conn is not None:
                _conn.close()
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            _conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
            _conn.execute("PRAGMA journal_mode=WAL")
            _conn.execute("PRAGMA synchronous=NORMAL")
            _conn.execute("""CREATE TABLE IF NOT EXISTS candles(
                symbol TEXT NOT NULL, interval TEXT NOT NULL, ts INTEGER NOT NULL,
                o REAL NOT NULL, h REAL NOT NULL, l REAL NOT NULL, c REAL NOT NULL, v REAL NOT NULL DEFAULT 0,
                PRIMARY KEY(symbol, interval, ts)) WITHOUT ROWID""")
            _conn.commit()
            _path = path
        return _conn


def bounds(symbol: str, interval: str) -> tuple[int, int, int] | None:
    """(first_ts, last_ts, rows) stored for this series, or None."""
    with _lock:
        r = _db().execute("SELECT MIN(ts), MAX(ts), COUNT(*) FROM candles WHERE symbol=? AND interval=?",
                          (symbol, interval)).fetchone()
    return (int(r[0]), int(r[1]), int(r[2])) if r and r[2] else None


def upsert(symbol: str, interval: str, df: pd.DataFrame, closed_before: float | None = None) -> int:
    """Store the CLOSED candles of `df` (index = UTC open time). The newest still-forming candle is skipped.
    Only rows near / after what we already hold are written, so a refresh costs milliseconds."""
    if df is None or df.empty or interval not in STEP:
        return 0
    step = STEP[interval]
    now = closed_before if closed_before is not None else time.time()
    ts = df.index.as_unit("s").asi8
    keep = (ts + step) <= now + 1                      # candle has finished
    b = bounds(symbol, interval)
    if b:
        keep &= ts >= b[1] - 3 * 86400                  # older rows are already stored (re-writing them is waste)
    if not keep.any():
        return 0
    sub = df[keep]
    rows = list(zip([symbol] * len(sub), [interval] * len(sub), ts[keep].tolist(),
                    sub["Open"].astype(float).tolist(), sub["High"].astype(float).tolist(),
                    sub["Low"].astype(float).tolist(), sub["Close"].astype(float).tolist(),
                    sub["Volume"].fillna(0).astype(float).tolist()))
    with _lock:
        con = _db()
        con.executemany("INSERT OR REPLACE INTO candles(symbol,interval,ts,o,h,l,c,v) VALUES (?,?,?,?,?,?,?,?)", rows)
        days = RETENTION_DAYS.get(interval, 9999)
        con.execute("DELETE FROM candles WHERE symbol=? AND interval=? AND ts < ?", (symbol, interval, int(time.time() - days * 86400)))
        con.commit()
    return len(rows)


def write_rows(symbol: str, interval: str, df: pd.DataFrame) -> int:
    """Store EVERY row of `df` (an import): unlike upsert() it also writes older rows. Existing rows are replaced."""
    if df is None or df.empty or interval not in STEP:
        return 0
    ts = df.index.as_unit("s").asi8
    rows = list(zip([symbol] * len(df), [interval] * len(df), ts.tolist(), df["Open"].astype(float).tolist(),
                    df["High"].astype(float).tolist(), df["Low"].astype(float).tolist(), df["Close"].astype(float).tolist(),
                    df["Volume"].fillna(0).astype(float).tolist()))
    with _lock:
        con = _db()
        con.executemany("INSERT OR REPLACE INTO candles(symbol,interval,ts,o,h,l,c,v) VALUES (?,?,?,?,?,?,?,?)", rows)
        con.commit()
    return len(rows)


def load(symbol: str, interval: str, since_ts: int | None = None, until_ts: int | None = None,
         limit: int | None = None) -> pd.DataFrame:
    """Stored candles (ascending). `limit` keeps the NEWEST `limit` rows inside the window."""
    q = "SELECT ts,o,h,l,c,v FROM candles WHERE symbol=? AND interval=?"
    args: list = [symbol, interval]
    if since_ts is not None:
        q += " AND ts>=?"
        args.append(int(since_ts))
    if until_ts is not None:
        q += " AND ts<?"
        args.append(int(until_ts))
    q += " ORDER BY ts DESC"
    if limit:
        q += " LIMIT ?"
        args.append(int(limit))
    with _lock:
        rows = _db().execute(q, args).fetchall()
    if not rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"],
                            index=pd.DatetimeIndex([], tz="UTC", name="time"))
    rows.reverse()
    arr = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    arr.index = pd.to_datetime(arr.pop("ts"), unit="s", utc=True)
    arr.index.name = "time"
    return arr


def merge(fresh: pd.DataFrame, symbol: str, interval: str, max_days: float | None = None) -> pd.DataFrame:
    """`fresh` (a just-downloaded window) + everything older we have stored. Fresh wins on overlap."""
    if fresh is None or fresh.empty:
        return fresh
    b = bounds(symbol, interval)
    first = int(fresh.index.as_unit("s").asi8[0])
    if not b or b[0] >= first:
        return fresh
    since = None if max_days is None else int(time.time() - max_days * 86400)
    old = load(symbol, interval, since_ts=since, until_ts=first)
    if old.empty:
        return fresh
    out = pd.concat([old, fresh])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.index.name = "time"
    return out


def stats(symbol: str | None = None) -> list[dict]:
    q = "SELECT symbol, interval, MIN(ts), MAX(ts), COUNT(*) FROM candles"
    args: list = []
    if symbol:
        q += " WHERE symbol=?"
        args.append(symbol)
    q += " GROUP BY symbol, interval ORDER BY symbol, interval"
    with _lock:
        rows = _db().execute(q, args).fetchall()
    return [{"symbol": r[0], "interval": r[1], "from": int(r[2]), "to": int(r[3]), "rows": int(r[4]),
             "days": round((r[3] - r[2]) / 86400, 1)} for r in rows]
