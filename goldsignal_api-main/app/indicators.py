"""
Classic momentum / trend / volatility indicators (numpy + pandas, all CAUSAL: the value at bar i only
uses bars <= i, so it is safe to compute once over a whole frame and index it at any bar - the backtest
and the live engine therefore see exactly the same numbers).

Used by the analyst layer (market read, regime, confirmations) - the structure engine in structure.py
stays the source of direction / zones, these add the "is momentum really behind it" read a human
analyst does by eye.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).ewm(span=n, adjust=False, min_periods=1).mean().to_numpy()


def rsi(c: np.ndarray, n: int = 14) -> np.ndarray:
    """Wilder RSI (0-100). NaN-free: the first bars are 50."""
    if len(c) < 2:
        return np.full(len(c), 50.0)
    d = np.diff(c, prepend=c[0])
    up, dn = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    au = pd.Series(up).ewm(alpha=1.0 / n, adjust=False, min_periods=1).mean().to_numpy()
    ad = pd.Series(dn).ewm(alpha=1.0 / n, adjust=False, min_periods=1).mean().to_numpy()
    rs = np.divide(au, ad, out=np.full_like(au, np.inf), where=ad > 1e-12)
    out = 100.0 - 100.0 / (1.0 + rs)
    out[(au <= 1e-12) & (ad <= 1e-12)] = 50.0
    return out


def adx(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int = 14) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ADX, +DI, -DI). ADX > ~22 = trending, < ~16 = ranging / chop."""
    if len(c) < 3:
        z = np.zeros(len(c))
        return z, z, z
    up = np.diff(h, prepend=h[0])
    dn = -np.diff(l, prepend=l[0])
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    a = 1.0 / n
    atr_ = pd.Series(tr).ewm(alpha=a, adjust=False, min_periods=1).mean().to_numpy()
    pdi = 100.0 * pd.Series(pdm).ewm(alpha=a, adjust=False, min_periods=1).mean().to_numpy() / np.maximum(atr_, 1e-12)
    mdi = 100.0 * pd.Series(mdm).ewm(alpha=a, adjust=False, min_periods=1).mean().to_numpy() / np.maximum(atr_, 1e-12)
    dx = 100.0 * np.abs(pdi - mdi) / np.maximum(pdi + mdi, 1e-12)
    return pd.Series(dx).ewm(alpha=a, adjust=False, min_periods=1).mean().to_numpy(), pdi, mdi


def macd(c: np.ndarray, fast: int = 12, slow: int = 26, sig: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    line = ema(c, fast) - ema(c, slow)
    signal = ema(line, sig)
    return line, signal, line - signal


def bb_width(c: np.ndarray, n: int = 20, k: float = 2.0) -> np.ndarray:
    """Bollinger bandwidth as a fraction of price (squeeze = low, expansion = high)."""
    s = pd.Series(c)
    m = s.rolling(n, min_periods=5).mean()
    sd = s.rolling(n, min_periods=5).std(ddof=0)
    out = (2.0 * k * sd / m.replace(0, np.nan)).bfill().fillna(0.0)
    return out.to_numpy()


def efficiency_ratio(c: np.ndarray, n: int = 20) -> np.ndarray:
    """Kaufman efficiency: |net move| / path length over n bars. ~1 = clean trend, ~0 = chop."""
    s = pd.Series(c)
    net = (s - s.shift(n)).abs()
    path = s.diff().abs().rolling(n, min_periods=3).sum()
    out = (net / path.replace(0, np.nan)).fillna(0.0).clip(0, 1)
    return out.to_numpy()


def pct_rank(x: np.ndarray, window: int = 200) -> np.ndarray:
    """Rolling percentile (0-1) of each value versus the previous `window` values (causal)."""
    s = pd.Series(x)
    return s.rolling(window, min_periods=20).apply(lambda w: float((w[:-1] < w[-1]).mean()) if len(w) > 1 else 0.5,
                                                   raw=True).fillna(0.5).to_numpy()


class Ind:
    """Lazy, cached indicator bundle for a Bars object (computed once per frame)."""

    def __init__(self, b):
        self.b = b
        self._c: dict[str, object] = {}

    def _get(self, key, fn):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]

    @property
    def ema20(self):
        return self._get("e20", lambda: ema(self.b.c, 20))

    @property
    def ema50(self):
        return self._get("e50", lambda: ema(self.b.c, 50))

    @property
    def ema200(self):
        return self._get("e200", lambda: ema(self.b.c, 200))

    @property
    def rsi14(self):
        return self._get("rsi", lambda: rsi(self.b.c, 14))

    @property
    def adx14(self):
        return self._get("adx", lambda: adx(self.b.h, self.b.l, self.b.c, 14))

    @property
    def macd(self):
        return self._get("macd", lambda: macd(self.b.c))

    @property
    def bbw(self):
        return self._get("bbw", lambda: bb_width(self.b.c))

    @property
    def er(self):
        return self._get("er", lambda: efficiency_ratio(self.b.c, 20))

    @property
    def atr_rank(self):
        return self._get("atrr", lambda: pct_rank(self.b.atr, 200))


def ind(b) -> Ind:
    """Indicator bundle attached to a Bars instance (works for Bars built via __new__ too)."""
    got = getattr(b, "_ind", None)
    if got is None:
        got = Ind(b)
        try:
            b._ind = got
        except AttributeError:  # pragma: no cover
            pass
    return got
