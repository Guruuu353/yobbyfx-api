"""
Synthetic market data for offline testing / demos.

Enable with the environment variable  GOLDSIGNAL_DEMO=1  (see README).
The generator builds one deterministic 1-minute series per symbol (layered
swings + noise + session-dependent volatility) and resamples it to every
timeframe, so all timeframes are mutually consistent.

It exists so the whole app (UI, chart overlays, alerts, backtest) can be
exercised without a live feed.  NEVER trade off this data.
"""
from __future__ import annotations

import zlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

START_PRICE = {
    "GC=F": 3350.0, "XAUUSD=X": 3335.0, "SI=F": 38.0,
    "EURUSD=X": 1.17, "GBPUSD=X": 1.35, "USDJPY=X": 147.0, "AUDUSD=X": 0.66,
    "GBPNZD=X": 2.25, "EURAUD=X": 1.75, "NZDUSD=X": 0.60, "USDCAD=X": 1.37, "USDCHF=X": 0.80,
    "EURGBP=X": 0.87, "EURJPY=X": 172.0, "GBPJPY=X": 198.0, "AUDJPY=X": 97.0, "NZDJPY=X": 88.0,
    "EURNZD=X": 1.95, "GBPAUD=X": 2.05, "GBPCAD=X": 1.85, "AUDNZD=X": 1.10, "CADJPY=X": 107.0,
    "EURCHF=X": 0.94, "XAGUSD=X": 38.0,
    "BTC-USD": 105000.0,
}
NO_VOLUME_SUFFIX = "=X"
_CACHE: dict[str, pd.DataFrame] = {}
BASE_DAYS = 760


def _market_open_mask(idx: pd.DatetimeIndex) -> np.ndarray:
    ny = idx.tz_convert("America/New_York")
    wd = np.asarray(ny.dayofweek)
    mins = np.asarray(ny.hour) * 60 + np.asarray(ny.minute)
    closed = (wd == 5) | ((wd == 4) & (mins >= 17 * 60)) | ((wd == 6) & (mins < 18 * 60))
    closed |= (mins >= 17 * 60) & (mins < 18 * 60)
    return ~closed


def _build_base(symbol: str) -> pd.DataFrame:
    seed = zlib.crc32(symbol.encode()) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(datetime.now(timezone.utc)).floor("min")
    idx = pd.date_range(end=end, periods=BASE_DAYS * 1440, freq="1min", tz="UTC")
    idx = idx[_market_open_mask(idx)]
    n = len(idx)
    ns = idx.as_unit("ns").asi8
    t_days = (ns - ns[0]) / 86_400e9

    # layered swings give clean HH/HL / LH/LL structure on every timeframe
    waves = np.zeros(n)
    for period, amp in ((1.7, 0.0045), (4.3, 0.011), (11.0, 0.024), (29.0, 0.04)):
        waves += amp * np.sin(2 * np.pi * t_days / period + rng.uniform(0, 6.28))

    hour = np.asarray(idx.hour)
    sess = np.where((hour >= 7) & (hour < 12), 1.2,
           np.where((hour >= 12) & (hour < 17), 1.5,
           np.where((hour >= 17) & (hour < 21), 0.9, 0.55)))
    # volatility clustering
    vol_state = np.abs(rng.normal(1.0, 0.25, n // 240 + 2)).repeat(240)[:n]
    sigma = 0.00019 * sess * vol_state
    noise = np.cumsum(rng.normal(0, 1, n) * sigma)
    logp = waves + noise
    close = START_PRICE.get(symbol, 100.0) * np.exp(logp - logp[0])
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(rng.normal(0, 1, (2, n))) * sigma * close * 0.6
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]

    if symbol.endswith(NO_VOLUME_SUFFIX):
        vol = np.zeros(n)
    else:
        spikes = np.where(rng.random(n) < 0.02, rng.uniform(2.5, 5, n), 1.0)
        vol = np.round(rng.lognormal(4.2, 0.45, n) * sess * spikes)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


def _base(symbol: str) -> pd.DataFrame:
    if symbol not in _CACHE:
        _CACHE[symbol] = _build_base(symbol)
    return _CACHE[symbol]


def _resample(df: pd.DataFrame, rule: str, **kw) -> pd.DataFrame:
    return (
        df.resample(rule, **kw)
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna(subset=["Open", "Close"])
    )


def demo_candles(symbol: str, interval: str, days: float) -> pd.DataFrame:
    """interval: 1m 5m 15m 30m 1h 1d 1wk"""
    base = _base(symbol)
    cutoff = base.index[-1] - pd.Timedelta(days=days)
    if interval == "1m":
        df = base
    elif interval in ("5m", "15m", "30m", "1h"):
        rule = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h"}[interval]
        df = _resample(base[base.index >= cutoff], rule)
    elif interval == "1d":
        df = _resample(base, "1D")
    elif interval == "1wk":
        df = _resample(base, "W-SUN", label="left", closed="left")
    else:
        raise ValueError(f"unsupported interval {interval}")
    return df[df.index >= cutoff].copy()
