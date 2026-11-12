"""
Live MT5 feed.

Why: Yahoo's 1-minute tape runs minutes behind, so every candle-close signal reached the bridge late, orders aged out
(`max_age_bars`, `expire_min`) or were sent after the price had gone. Your MT5 terminal already has the broker's live
ticks. The Expert Advisor now pushes, every poll (~2 s), the live quote and the last 1-minute candles (including the
forming one) of every allowed pair. This module keeps them in memory and `data_fetcher` lays them over the Yahoo 1m
frame, so EVERY timeframe (5m ... 1D, built from the 1m tape by `merge_live`) is as fresh as the broker feed.

Prices are stored RAW (broker price minus the calibration offset), like the rest of the candle store, so the engine's
`+offset` step in get_candles() gives back exactly the broker price.

Clock: the EA also sends the terminal's last-tick time (server time). Comparing it with THIS server's UTC clock gives the
broker's real UTC offset - independent of the VPS time zone setting, which is why changing the VPS to Nairobi time never
moved the chart.
"""
from __future__ import annotations

import threading
import time

import pandas as pd

MAX_BARS = 2000                 # per symbol (~33 hours of 1m)
FRESH_SECONDS = 20              # a quote older than this is not "live"
_lock = threading.RLock()
_bars: dict[str, dict[int, tuple]] = {}        # symbol -> {open_ts_utc: (o, h, l, c, v)}
_quote: dict[str, dict] = {}                   # symbol -> {bid, ask, spread, at}
_clock: dict = {}                              # {"server_offset": h, "skew_sec": s, "at": t}


# ------------------------------------------------------------------------------------------ clock
def detect_offset(server_time: float | int | None, now: float | None = None) -> float | None:
    """Broker server offset from UTC in hours, from the terminal's last-tick time (server clock) vs our UTC clock.
    None when the answer is not trustworthy (tick older than ~2 min / market closed). The EA only sends a fresh tick time."""
    if not server_time:
        return None
    now = time.time() if now is None else now
    diff = float(server_time) - now
    if abs(diff) > 15 * 3600:
        return None
    half = round((diff + 30.0) / 1800.0)
    if not (-120.0 <= diff - half * 1800.0 <= 60.0):   # the tick must be at most ~2 min old (or 1 min "from the future"): else it is not a clean multiple
        return None
    return half * 0.5


def note_clock(server_time, gmt_time=None, now: float | None = None) -> dict:
    """Remember the broker offset; also report how wrong the VPS clock is (EA TimeGMT vs this server's UTC)."""
    now = time.time() if now is None else now
    off = detect_offset(server_time, now)
    with _lock:
        if off is not None:
            _clock["server_offset"] = off
            _clock["at"] = now
        if gmt_time:
            skew = float(gmt_time) - now
            _clock["skew_sec"] = int(round(skew))
        return dict(_clock)


def clock() -> dict:
    with _lock:
        return dict(_clock)


# ------------------------------------------------------------------------------------------ ingest
def ingest(symbol: str, rows: list, tz_offset_hours: float, raw_offset: float = 0.0, quote: dict | None = None,
           now: float | None = None) -> int:
    """rows = [[server_epoch, o, h, l, c, vol], ...] (forming bar included). Returns how many bars were stored."""
    now = time.time() if now is None else now
    n = 0
    with _lock:
        d = _bars.setdefault(symbol, {})
        for r in rows or []:
            try:
                t = int(float(r[0]) - tz_offset_hours * 3600.0)
                o, h, l, c = (float(r[1]) - raw_offset, float(r[2]) - raw_offset, float(r[3]) - raw_offset, float(r[4]) - raw_offset)
                v = float(r[5]) if len(r) > 5 else 0.0
            except (TypeError, ValueError, IndexError):
                continue
            if not (c > 0 and h >= max(o, c) - 1e-9 and l <= min(o, c) + 1e-9):
                continue
            if t > now + 120:                                     # from the future: wrong offset, never store
                continue
            t -= t % 60
            d[t] = (o, h, l, c, v)
            n += 1
        if len(d) > MAX_BARS:
            for k in sorted(d)[:len(d) - MAX_BARS]:
                del d[k]
        if quote:
            try:
                bid, ask = float(quote["bid"]) - raw_offset, float(quote["ask"]) - raw_offset
                _quote[symbol] = {"bid": bid, "ask": ask, "spread": float(quote.get("spread") or 0), "at": now}
            except (KeyError, TypeError, ValueError):
                pass
    return n


def quote(symbol: str, max_age: float = FRESH_SECONDS, now: float | None = None) -> dict | None:
    now = time.time() if now is None else now
    with _lock:
        q = _quote.get(symbol)
        return dict(q) if q and now - q["at"] <= max_age else None


def fresh(symbol: str, max_age: float = FRESH_SECONDS, now: float | None = None) -> bool:
    return quote(symbol, max_age, now) is not None


def any_fresh(max_age: float = FRESH_SECONDS, now: float | None = None) -> bool:
    now = time.time() if now is None else now
    with _lock:
        return any(now - q["at"] <= max_age for q in _quote.values())


def age(symbol: str, now: float | None = None) -> float | None:
    now = time.time() if now is None else now
    with _lock:
        q = _quote.get(symbol)
        return (now - q["at"]) if q else None


def last_price(symbol: str, max_age: float = FRESH_SECONDS) -> float | None:
    """Mid price of the live quote, RAW (no calibration offset), or None."""
    q = quote(symbol, max_age)
    return (q["bid"] + q["ask"]) / 2.0 if q else None


# ------------------------------------------------------------------------------------------ overlay
def overlay(symbol: str, m1: pd.DataFrame, max_age: float = 90.0) -> pd.DataFrame:
    """Yahoo's 1m frame with the broker's candles laid over its tail (EA wins from its first bar on). Untouched when the
    EA feed is not live, so everything falls back to Yahoo by itself when the terminal is off."""
    if not fresh(symbol, max_age):
        return m1
    with _lock:
        d = dict(_bars.get(symbol) or {})
    if not d:
        return m1
    ts = sorted(d)
    ea = pd.DataFrame([d[t] for t in ts], columns=["Open", "High", "Low", "Close", "Volume"],
                      index=pd.to_datetime(ts, unit="s", utc=True))
    ea.index.name = "time"
    if m1 is None or m1.empty:
        return ea
    base = m1[m1.index < ea.index[0]]
    out = pd.concat([base, ea])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.index.name = "time"
    return out


def status(symbol: str) -> dict:
    a = age(symbol)
    with _lock:
        n = len(_bars.get(symbol) or {})
    return {"live": a is not None and a <= FRESH_SECONDS, "age_sec": None if a is None else round(a, 1), "bars": n}


def reset() -> None:
    with _lock:
        _bars.clear()
        _quote.clear()
        _clock.clear()
