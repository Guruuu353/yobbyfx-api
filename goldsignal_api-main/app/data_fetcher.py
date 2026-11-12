"""
Market-data layer.

Why this file exists (the "JSONDecodeError('Expecting value: line 1 column 1')" bug)
------------------------------------------------------------------------------------
Yahoo Finance regularly answers with an HTML / plain-text "Too Many Requests"
or consent page instead of JSON.  Old yfinance releases (the project pinned
0.2.43) crash while decoding it and  yf.download()  swallows the error and just
returns an empty frame  ->  "1 Failed download: ['GC=F']: JSONDecodeError".

What we do about it:
  1. requirements.txt now asks for a modern yfinance (curl_cffi based, browser
     impersonation) - the actual fix.  Upgrade with:
         pip install -U -r requirements.txt
  2. Every request goes through a chain of independent download methods
     (direct Yahoo chart API -> yfinance Ticker.history -> yf.download); the
     first that returns data wins.
  3. Results are cached (memory + disk) with a per-interval TTL; if every method
     fails we serve the last good cache (marked stale) instead of a 500.
  4. Failures are remembered for a short cool-down so the dashboard polling
     cannot hammer Yahoo while it is throttling us.
  5. If there is truly nothing, a DataUnavailable exception is raised and the
     routers translate it into a clean HTTP 503 with diagnostics.

Set GOLDSIGNAL_DEMO=1 to use synthetic data (offline UI/strategy testing).
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from . import candle_store, sessions, settings

log = logging.getLogger("goldsignal.data")

CACHE_DIR = os.path.join(os.environ.get("GOLDSIGNAL_DATA_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

DEMO = os.environ.get("GOLDSIGNAL_DEMO", "").lower() in ("1", "true", "yes")

# public timeframe -> Yahoo interval
YF_INTERVAL = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
               "1h": "1h", "4h": "1h", "1d": "1h", "1w": "1h"}  # 4h/1d/1w are cut from 1h (broker-aligned)
INTERVAL_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800,
                    "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800}
SUPPORTED = list(INTERVAL_SECONDS)

# 1m is the "live tape": every other timeframe is patched with it (see merge_live), so the
# slower frames can be cached longer without the newest candle ever lagging.
DEFAULT_PERIOD = {"1m": "5d", "5m": "59d", "15m": "59d", "30m": "59d",
                  "1h": "729d", "1d": "2y", "1wk": "5y"}
# the longest window Yahoo will serve per request - older history comes from candle_store (it accumulates)
YF_MAX_DAYS = {"1m": 7, "5m": 59, "15m": 59, "30m": 59, "1h": 729, "1d": 7300, "1wk": 7300}
TTL = {"1m": 12, "5m": 90, "15m": 120, "30m": 180, "1h": 300, "1d": 120, "1wk": 600}
FAIL_COOLDOWN = 25  # seconds before retrying a failed download (doubles on every repeated failure, capped below)
FAIL_COOLDOWN_MAX = 300

# Yahoo no longer serves some spot symbols ("No data found, symbol may be delisted"). When the symbol itself
# returns nothing from every method, the closest live instrument is tried instead and the result is marked
# as a proxy in the feed metadata. (Gold futures trade a little above spot - use the price calibration in Profile.)
SYMBOL_FALLBACKS = {
    "XAUUSD=X": ("XAUUSD=X", "GC=F", "MGC=F"),
    "XAGUSD=X": ("XAGUSD=X", "SI=F"),
}

SYMBOL_RE = re.compile(r"^[A-Za-z0-9=^.\-]{1,20}$")

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")


class DataUnavailable(Exception):
    def __init__(self, message: str, diagnostics: list[str] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or []


# ------------------------------------------------------------------ utilities
def check_symbol(symbol: str) -> str:
    if not SYMBOL_RE.match(symbol or ""):
        raise DataUnavailable(f"Invalid symbol '{symbol}'")
    return symbol


def is_futures(symbol: str) -> bool:
    return not symbol.endswith("=X")


def period_days(period: str) -> float:
    m = re.fullmatch(r"(\d+)(d|wk|mo|y)", period.strip().lower())
    if not m:
        raise ValueError(f"bad period {period!r}")
    n, unit = int(m.group(1)), m.group(2)
    return n * {"d": 1, "wk": 7, "mo": 30.5, "y": 365.25}[unit]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    need = ["Open", "High", "Low", "Close"]
    if any(c not in df.columns for c in need):
        raise ValueError(f"missing OHLC columns: {list(df.columns)}")
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df = df[need + ["Volume"]].astype(float)
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    df.index = idx
    df["Volume"] = df["Volume"].fillna(0.0)
    df = df.dropna(subset=need)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "time"
    return df


# ------------------------------------------------------------- download methods
def parse_chart_json(payload: dict) -> pd.DataFrame:
    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise RuntimeError(str(chart["error"])[:120])
    res = (chart.get("result") or [None])[0]
    if not res or not res.get("timestamp"):
        raise RuntimeError("empty chart result")
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame(
        {"Open": q.get("open"), "High": q.get("high"), "Low": q.get("low"),
         "Close": q.get("close"), "Volume": q.get("volume")},
        index=pd.to_datetime(res["timestamp"], unit="s", utc=True),
    )
    return _normalize(df)


def _fetch_direct(symbol: str, yf_interval: str, period: str) -> pd.DataFrame:
    """Plain HTTP call to Yahoo's chart endpoint (no cookies/crumb needed)."""
    p2 = int(time.time()) + 60
    p1 = int(p2 - period_days(period) * 86400)
    last: Exception | None = None
    for host in ("query1", "query2"):
        url = f"https://{host}.finance.yahoo.com/v8/finance/chart/{requests.utils.quote(symbol, safe='')}"
        try:
            r = requests.get(
                url,
                params={"period1": p1, "period2": p2, "interval": yf_interval,
                        "includePrePost": "false", "events": "div,splits"},
                headers={"User-Agent": UA, "Accept": "application/json"},
                timeout=12,
            )
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:80]!r}")
            return parse_chart_json(r.json())
        except Exception as e:  # noqa: BLE001
            last = e
    raise last  # type: ignore[misc]


def _fetch_yf_history(symbol: str, yf_interval: str, period: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.Ticker(symbol).history(period=period, interval=yf_interval,
                                   auto_adjust=False, actions=False, timeout=15)
    if df is None or df.empty:
        raise RuntimeError("empty frame")
    return _normalize(df)


def _fetch_yf_download(symbol: str, yf_interval: str, period: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(symbol, period=period, interval=yf_interval, progress=False,
                     auto_adjust=False, threads=False, timeout=15)
    if df is None or df.empty:
        raise RuntimeError("empty frame")
    return _normalize(df)


# yfinance impersonates a browser (curl_cffi) and copes with Yahoo's cookie/crumb -> preferred.
# The raw "direct" call is a fallback: from many networks Yahoo answers it with HTTP 429.
METHODS = (("yfinance.history", _fetch_yf_history), ("direct", _fetch_direct),
           ("yfinance.download", _fetch_yf_download))

_method_block: dict[str, float] = {}      # method -> unix time until which we skip it
_gate = threading.Semaphore(1)            # one Yahoo request at a time (yfinance/curl_cffi is not thread-safe)
_last_call = [0.0]
_call_lock = threading.Lock()
BLOCK_SECONDS = 120


def _throttle() -> None:
    """Space Yahoo requests >= 0.2 s apart so bursts of dashboard polling don't trigger 429."""
    with _call_lock:
        wait = 0.2 - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()


def _download(symbol: str, yf_interval: str, period: str) -> tuple[pd.DataFrame, str, list[str]]:
    """Try the symbol itself, then (for known-dead spot symbols) its fallback instruments."""
    diag: list[str] = []
    for alt in SYMBOL_FALLBACKS.get(symbol, (symbol,)):
        try:
            df, method, d = _download_one(alt, yf_interval, period)
            if alt != symbol:
                method = f"{method} (proxy {alt})"
            return df, method, diag + d
        except DataUnavailable as e:
            diag += [f"[{alt}] {x}" for x in e.diagnostics]
            if any("429" in x or "Too Many" in x for x in e.diagnostics):
                break          # being rate limited: trying more symbols would only make it worse
    raise DataUnavailable(f"Could not download {symbol} {yf_interval} from Yahoo Finance", diag)


def _download_one(symbol: str, yf_interval: str, period: str) -> tuple[pd.DataFrame, str, list[str]]:
    diag: list[str] = []
    now = time.time()
    order = [m for m in METHODS if _method_block.get(m[0], 0) <= now] or list(METHODS)
    for name, fn in order:
        try:
            with _gate:
                _throttle()
                df = fn(symbol, yf_interval, period)
            if len(df):
                return df, name, diag
            diag.append(f"{name}: no rows")
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {str(e)[:110]}"
            diag.append(f"{name}: {msg}")
            if "429" in msg or "Too Many" in msg or "Rate" in msg:
                _method_block[name] = time.time() + BLOCK_SECONDS  # stop poking a method that is being throttled
    raise DataUnavailable(f"Could not download {symbol} {yf_interval} from Yahoo Finance", diag)


# ---------------------------------------------------------------------- caching
_mem: dict[tuple, tuple[float, pd.DataFrame]] = {}
_meta: dict[tuple, dict] = {}
_fail_until: dict[tuple, tuple[float, list[str]]] = {}
_fail_count: dict[tuple, int] = {}
_locks: dict[tuple, threading.Lock] = {}
_glock = threading.Lock()
LAST_DIAGNOSTICS: dict = {"ok": None, "at": None, "detail": []}


def _lock(key: tuple) -> threading.Lock:
    with _glock:
        return _locks.setdefault(key, threading.Lock())


def _disk_path(symbol: str, interval: str, period: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]", "", symbol)
    return os.path.join(CACHE_DIR, f"{safe}_{interval}_{period}.csv")


def _read_disk(path: str) -> pd.DataFrame | None:
    try:
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "time"
        return df if len(df) else None
    except Exception:  # noqa: BLE001
        return None


def _refresh_period(symbol: str, yf_interval: str, period: str | None) -> tuple[str, bool]:
    """(period to download, is_default_request). Once the store holds recent history a short top-up is enough."""
    custom = period is not None
    period = period or DEFAULT_PERIOD[yf_interval]
    maxd = YF_MAX_DAYS.get(yf_interval, 365)
    if period_days(period) > maxd:
        period = f"{int(maxd)}d"                                   # Yahoo refuses more: the store supplies the rest
    if not custom and not DEMO and yf_interval in ("5m", "15m", "30m", "1h"):
        b = candle_store.bounds(symbol, yf_interval)
        if b and b[2] > 150:
            age = (time.time() - b[1]) / 86400.0
            if age < 5:
                period = f"{int(min(maxd, max(5, age + 3)))}d"       # top-up only; weekends are < 3 days
    return period, not custom


def _get_raw(symbol: str, yf_interval: str, period: str | None = None) -> pd.DataFrame:
    """Cached fetch of a native Yahoo interval (1m,5m,15m,30m,1h,1d,1wk), merged with the stored history."""
    want = period or DEFAULT_PERIOD[yf_interval]
    key = (symbol, yf_interval, want)
    ttl = TTL.get(yf_interval, 120)

    hit = _mem.get(key)
    if hit and time.time() - hit[0] <= ttl:
        return hit[1]

    with _lock(key):
        hit = _mem.get(key)
        if hit and time.time() - hit[0] <= ttl:
            return hit[1]

        # ---- demo mode
        if DEMO:
            from .demo_data import demo_candles
            df = demo_candles(symbol, yf_interval, period_days(want))
            _mem[key] = (time.time(), df)
            _meta[key] = {"source": "demo", "fetched_at": int(time.time()), "stale": False}
            return df

        def _stored(max_days: float | None) -> pd.DataFrame | None:
            try:
                st = candle_store.load(symbol, yf_interval,
                                       since_ts=None if max_days is None else int(time.time() - max_days * 86400))
                return st if len(st) else None
            except Exception:  # noqa: BLE001
                return None

        # ---- recently failed -> don't hammer Yahoo
        if hit:
            stale = hit[1]
        else:
            stale = _stored(period_days(want) if period else None)
            if stale is None:
                stale = _read_disk(_disk_path(symbol, yf_interval, want))
        cool = _fail_until.get(key)
        if cool and time.time() < cool[0]:
            if stale is not None:
                _meta[key] = {"source": "cache", "stale": True, "error": cool[1][:3]}
                return stale
            raise DataUnavailable(f"Data feed unavailable for {symbol} {yf_interval}", cool[1])

        try:
            dl_period, is_default = _refresh_period(symbol, yf_interval, period)
            df, method, diag = _download(symbol, yf_interval, dl_period)
            try:
                candle_store.upsert(symbol, yf_interval, df)
                df = candle_store.merge(df, symbol, yf_interval, None if is_default else period_days(want))
            except Exception as e:  # noqa: BLE001 - storing is a bonus, never break the feed over it
                log.warning("candle store failed: %s", e)
            if not is_default:
                cut = df.index[-1] - pd.Timedelta(days=period_days(want))
                df = df[df.index >= cut]
            _mem[key] = (time.time(), df)
            _meta[key] = {"source": method, "fetched_at": int(time.time()), "stale": False}
            LAST_DIAGNOSTICS.update(ok=True, at=int(time.time()),
                                    detail=[f"{symbol} {yf_interval}: {method}"] + diag)
            try:
                df.tail(20000).to_csv(_disk_path(symbol, yf_interval, want))
            except OSError:
                pass
            _fail_until.pop(key, None)
            _fail_count.pop(key, None)
            return df
        except DataUnavailable as e:
            n = _fail_count[key] = _fail_count.get(key, 0) + 1
            _fail_until[key] = (time.time() + min(FAIL_COOLDOWN * 2 ** (n - 1), FAIL_COOLDOWN_MAX), e.diagnostics)
            LAST_DIAGNOSTICS.update(ok=False, at=int(time.time()), detail=[f"{symbol} {yf_interval}"] + e.diagnostics)
            log.warning("Data download failed %s %s: %s", symbol, yf_interval, e.diagnostics)
            if stale is not None:
                _meta[key] = {"source": "cache", "stale": True, "error": e.diagnostics[:3]}
                return stale
            raise


# ------------------------------------------------------------------- public API
def resample_ohlc(df: pd.DataFrame, rule: str, symbol: str = "GC=F") -> pd.DataFrame:
    """Cut 4h / 1d / 1w candles the way MetaTrader 5 does (see settings.py):
    ny_close -> broker server time = New York time + 7 h (GMT+2 winter / GMT+3 summer, DST follows the US),
                so the trading day starts at 17:00 New York and 4H candles start 17,21,01,05,09,13 NY time
    utc      -> candles cut on UTC boundaries"""
    if df.empty:
        return df
    idx = df.index
    utc_naive = idx.tz_convert("UTC").tz_localize(None)
    if settings.anchor() == "ny_close":
        et = idx.tz_convert("America/New_York").tz_localize(None)
        off = et - utc_naive                       # negative: NY is behind UTC
        shift = pd.Timedelta(hours=7)
        srv = et + shift
    else:
        off = utc_naive - utc_naive                # zero timedeltas
        shift = pd.Timedelta(0)
        srv = utc_naive
    if rule == "4h":
        key = srv.floor("4h")
    elif rule == "1d":
        key = srv.floor("D")
    elif rule == "1w":
        d = srv.floor("D")
        key = d - pd.to_timedelta(d.dayofweek, unit="D")   # week starts Monday 00:00 server time
    else:
        raise ValueError(rule)
    tmp = df.reset_index(drop=True).assign(_k=key, _off=off)
    g = tmp.groupby("_k")
    out = pd.DataFrame({"Open": g["Open"].first(), "High": g["High"].max(), "Low": g["Low"].min(),
                        "Close": g["Close"].last(), "Volume": g["Volume"].sum(), "_off": g["_off"].first()})
    start = out.index.to_series() - shift - out["_off"]     # back to naive UTC
    out = out.drop(columns="_off")
    out.index = pd.DatetimeIndex(start.to_numpy()).tz_localize("UTC")
    out.index.name = "time"
    return out


def merge_live(base: pd.DataFrame, interval: str, symbol: str) -> pd.DataFrame:
    """Patch a (cached, possibly minutes-old) frame with the live 1-minute tape so the newest candle of EVERY
    timeframe keeps moving, and brand-new candles appear the moment they open.

    Only the TAIL is touched and every rebuilt bar is cut on an EPOCH-aligned boundary (floor(unix / step) * step)
    - the grid Yahoo's own 5m/15m/30m/1h bars sit on.

    v4 fix for the "gaps / wrong candles on 30m and below" bug: the old version REPLACED the last cached candle with
    one rebuilt from whatever 1-minute bars happened to be available. Yahoo's 1m feed has holes, so that candle came
    out too short (a bar missing its open, a clipped wick) and the next refresh snapped it back - gaps and jumping
    candles. Now an existing candle is only EXTENDED by the tape: its open stays, its high/low can only grow, and
    only its close follows the latest price. A candle is rebuilt from the tape alone only when it is brand new."""
    if interval == "1m" or base.empty:
        return base
    try:
        m1 = _get_raw(symbol, "1m")
    except DataUnavailable:
        return base
    if m1.empty:
        return base
    step = INTERVAL_SECONDS[interval]
    last_start = base.index[-1]
    seg = m1[m1.index >= last_start]                     # the tape from the start of the newest cached candle
    if seg.empty:
        return base
    secs = seg.index.as_unit("s").asi8
    bucket = (secs // step) * step
    g = seg.assign(_k=bucket).groupby("_k")
    tape = pd.DataFrame({"Open": g["Open"].first(), "High": g["High"].max(), "Low": g["Low"].min(),
                         "Close": g["Close"].last(), "Volume": g["Volume"].sum()})
    tape.index = pd.to_datetime(tape.index, unit="s", utc=True)
    tape.index.name = "time"
    if tape.empty:
        return base

    df = base.copy()
    newest = tape.index.max()
    for ts in tape.index.intersection(df.index):        # extend, never shrink, a candle we already have
        t = tape.loc[ts]
        row = df.loc[ts]
        df.loc[ts, "High"] = max(row["High"], t["High"])
        df.loc[ts, "Low"] = min(row["Low"], t["Low"])
        df.loc[ts, "Volume"] = max(row["Volume"], t["Volume"])
        if ts == newest:                                 # only the newest bucket's close follows the latest price
            df.loc[ts, "Close"] = t["Close"]
    new = tape[~tape.index.isin(df.index) & (tape.index > df.index[-1])]
    if len(new):
        df = pd.concat([df, new]).sort_index()
    return df[~df.index.duplicated(keep="last")]


# ------------------------------------------------------------------ candle hygiene
def clean_candles(df: pd.DataFrame, interval: str, futures: bool = True, max_fill: int = 12,
                  smooth_open: bool | None = None, info: dict | None = None) -> pd.DataFrame:
    """Make intraday candles chart-safe and analysis-safe. Applied to EVERY consumer (chart, signals, backtest)
    so what you see is what the engine analysed.

      1. align      every candle onto the epoch grid of its timeframe (5m -> :00 :05 ..., 30m -> :00 :30); candles
                    that land in the same slot are merged (open of the first, high/low extremes, close of the last)
      2. integrity  high >= max(open, close), low <= min(open, close)  (some feeds print bars that violate it)
      3. fill       a bar MISSING from the middle of an open market (Yahoo omits bars with no prints) is inserted as a
                    flat candle at the previous close, volume 0, so the series has no holes and the bar-count windows
                    the engine uses keep meaning the same real time. Only when the whole gap is inside open-market
                    hours and <= max_fill bars; weekends / the daily break are never filled.
      4. continuity (<= 30m only) when a candle opens a hair away from the previous close inside one continuous
                    session, the open snaps to the previous close - the way MetaTrader draws it - so the chart has no
                    fake micro-gaps. A real gap (> 0.5 x typical range) is left alone.
    `info` (optional dict) is filled with what was changed."""
    if df is None or df.empty or interval in ("4h", "1d", "1w", "1wk"):
        return df
    step = INTERVAL_SECONDS.get(interval) or INTERVAL_SECONDS[{"1wk": "1w"}.get(interval, interval)]
    info = info if info is not None else {}
    info.update(aligned=0, merged=0, fixed=0, filled=0, gaps=[])
    if smooth_open is None:
        smooth_open = step <= 1800
    out = df
    secs = out.index.as_unit("s").asi8

    # 1. align (sub-hour frames; 1h bars keep Yahoo's own start)
    if step < 3600:
        grid = (secs // step) * step
        moved = grid != secs
        if moved.any() or (np.diff(grid) == 0).any():
            info["aligned"] = int(moved.sum())
            g = out.assign(_k=grid).groupby("_k", sort=True)
            info["merged"] = int(len(out) - len(g))
            out = pd.DataFrame({"Open": g["Open"].first(), "High": g["High"].max(), "Low": g["Low"].min(),
                                "Close": g["Close"].last(), "Volume": g["Volume"].sum()})
            out.index = pd.to_datetime(out.index, unit="s", utc=True)
            out.index.name = "time"
            secs = out.index.as_unit("s").asi8
    if len(out) < 3:
        return out

    # 2. integrity
    o, h, l, c = (out[k].to_numpy(float).copy() for k in ("Open", "High", "Low", "Close"))
    h2, l2 = np.maximum.reduce([o, h, l, c]), np.minimum.reduce([o, h, l, c])
    bad = (h2 != h) | (l2 != l)
    if bad.any():
        info["fixed"] = int(bad.sum())
        out = out.copy()
        out["High"], out["Low"] = h2, l2
        h, l = h2, l2
    v = out["Volume"].to_numpy(float).copy()

    # 3. fill holes inside open markets
    delta = np.diff(secs)
    holes = np.nonzero((delta > step) & (delta % step == 0))[0]
    if len(holes):
        add_t, add_p = [], []
        for k in holes:
            n_missing = int(delta[k] // step) - 1
            miss = secs[k] + step * np.arange(1, n_missing + 1)
            idx = pd.to_datetime(miss, unit="s", utc=True)
            if not sessions.market_open_mask(idx, futures).all():
                continue                                         # weekend / daily break inside: a REAL gap
            if n_missing > max_fill:
                info["gaps"].append({"from": int(miss[0]), "to": int(miss[-1]), "bars": n_missing})
                continue
            add_t.extend(miss.tolist())
            add_p.extend([float(c[k])] * n_missing)
        if add_t:
            info["filled"] = len(add_t)
            flat = pd.DataFrame({"Open": add_p, "High": add_p, "Low": add_p, "Close": add_p, "Volume": 0.0},
                                index=pd.to_datetime(add_t, unit="s", utc=True))
            flat.index.name = "time"
            out = pd.concat([out, flat]).sort_index()
            secs = out.index.as_unit("s").asi8
            o, h, l, c, v = (out[k].to_numpy(float).copy() for k in ("Open", "High", "Low", "Close", "Volume"))

    # 4. open = previous close inside one continuous run of candles
    if smooth_open and len(out) > 60:
        adj = np.diff(secs) == step
        gap = o[1:] - c[:-1]
        rng = pd.Series(h - l).rolling(50, min_periods=10).median().to_numpy()[:-1]
        # Yahoo's FX / 1m bars are often single-tick (open = high = low = close) so the typical RANGE is ~0 and a
        # range-only threshold never snapped anything - the chart came out as dots and dashes. Typical close-to-close
        # movement is the fairer yardstick: a candle that opens within ~2x of it of the previous close is continuous.
        mov = pd.Series(np.abs(np.diff(c, prepend=c[0]))).rolling(100, min_periods=10).mean().to_numpy()[:-1]
        thr = np.maximum(0.5 * np.nan_to_num(rng, nan=0.0), 3.0 * np.nan_to_num(mov, nan=0.0))
        snap = adj & (np.abs(gap) > 0) & (np.abs(gap) <= thr)
        if snap.any():
            o2 = o.copy()
            o2[1:][snap] = c[:-1][snap]
            h = np.maximum(h, o2)
            l = np.minimum(l, o2)
            out = out.copy()
            out["Open"], out["High"], out["Low"] = o2, h, l
            info["smoothed"] = int(snap.sum())
    if info.get("filled"):
        out = out.copy()
        out["Filled"] = out.index.isin(flat.index)       # lets the chart draw these slots empty instead of fake dashes
    return out


def _candles(symbol: str, interval: str, period: str | None, live: bool) -> pd.DataFrame:
    if interval in ("4h", "1d", "1w"):
        base = _candles(symbol, "1h", period, live)
        return resample_ohlc(base, interval, symbol)
    df = _get_raw(symbol, YF_INTERVAL[interval], period)
    return merge_live(df, interval, symbol) if live else df


def get_candles(symbol: str, interval: str = "15m", period: str | None = None, live: bool = True) -> pd.DataFrame:
    """OHLCV DataFrame (UTC index, columns Open High Low Close Volume).
    Includes the still-forming last bar; use drop_incomplete() for analysis.
    live=True patches the newest bar(s) with the 1-minute feed (ignored when a custom period is asked for).
    4h/1d/1w are cut from 1h candles on MT5-style boundaries and every price gets the calibration offset."""
    check_symbol(symbol)
    if interval not in INTERVAL_SECONDS:
        raise DataUnavailable(f"Unsupported timeframe '{interval}'")
    df = _candles(symbol, interval, period, live and period is None)
    if interval in ("1m", "5m", "15m", "30m", "1h"):
        df = clean_candles(df, interval, is_futures(symbol), smooth_open=None if interval != "1h" else False)
    off = settings.offset(symbol)
    if off:
        df = df.copy()
        df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]] + off
    return df


def raw_last_price(symbol: str) -> float:
    """Latest Yahoo price WITHOUT the calibration offset (used to calibrate)."""
    m1 = _get_raw(symbol, "1m")
    return float(m1["Close"].iloc[-1])


CLOSE_GRACE = 15  # seconds after a bar's end before it counts as closed (lets the last 1m bar arrive)


def drop_incomplete(df: pd.DataFrame, interval: str, now: datetime | None = None) -> pd.DataFrame:
    """Remove the last bar if it has not closed yet."""
    if df.empty:
        return df
    from .sessions import now_utc
    now = now or now_utc()
    end = df.index[-1] + timedelta(seconds=INTERVAL_SECONDS[interval] + CLOSE_GRACE)
    return df.iloc[:-1] if now < end else df


def get_meta(symbol: str, interval: str) -> dict:
    yf_i = YF_INTERVAL.get(interval, interval)
    for (s, i, _p), m in _meta.items():
        if s == symbol and i == yf_i:
            return m
    return {}


STALE_AFTER = 12 * 60  # seconds without a new 1m bar (while the market is open) before we call the feed stale


def data_status(symbol: str, interval: str, df: pd.DataFrame) -> dict:
    """Freshness info for the UI. Lag is measured on the 1-minute tape (the freshest thing we have)."""
    from .sessions import market_open, now_utc
    meta = get_meta(symbol, interval)
    now = now_utc()
    lag = None
    try:
        m1 = _get_raw(symbol, "1m")
        if len(m1):
            lag = (now - m1.index[-1].to_pydatetime()).total_seconds() - 60
    except DataUnavailable:
        pass
    if lag is None and len(df):
        lag = (now - df.index[-1].to_pydatetime()).total_seconds() - INTERVAL_SECONDS[interval]
    mkt = market_open(None, is_futures(symbol))
    stale = bool(not DEMO and mkt and lag is not None and lag > STALE_AFTER)
    return {
        "source": meta.get("source", "unknown"),
        "last_bar": int(df.index[-1].timestamp()) if len(df) else None,
        "lag_sec": int(max(0, lag)) if lag is not None else None,
        "stale": stale,
        "market_open": mkt,
        "demo": DEMO,
    }


def feed_health(symbol: str = "GC=F") -> dict:
    """Live connectivity test used by /market/health (bypasses caches)."""
    if DEMO:
        return {"demo": True, "ok": True, "methods": [{"method": "demo", "ok": True}]}
    out = []
    for name, fn in METHODS:
        t0 = time.time()
        try:
            df = fn(symbol, "1h", "5d")
            out.append({"method": name, "ok": True, "rows": len(df), "ms": int((time.time() - t0) * 1000),
                        "last": df.index[-1].isoformat()})
        except Exception as e:  # noqa: BLE001
            out.append({"method": name, "ok": False, "error": f"{type(e).__name__}: {str(e)[:140]}",
                        "ms": int((time.time() - t0) * 1000)})
    try:
        import yfinance
        ver = yfinance.__version__
    except Exception:  # noqa: BLE001
        ver = "not installed"
    ok = any(m["ok"] for m in out)
    return {"demo": False, "ok": ok, "yfinance_version": ver, "methods": out,
            "hint": None if ok else
            "All download methods failed. Run: pip install -U -r requirements.txt (yfinance must be recent), "
            "then check this machine can reach query1.finance.yahoo.com (VPN / firewall / DNS)."}


def clear_caches() -> None:
    _mem.clear()
    _fail_until.clear()
    _fail_count.clear()