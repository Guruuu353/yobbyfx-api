"""
Confirmation engine for the entry timeframe.

A "trigger" used to be one of two shapes (engulfing / rejection wick) backed by either a mini-BOS or a
volume spike - and volume does not exist for spot FX / XAUUSD on Yahoo, so on those pairs the single
mini-BOS carried the whole confirmation. This module adds the other things a discretionary analyst looks
for before pulling the trigger, and returns them as a list so the engine can demand N independent
confirmations instead of "one of two".

CANDLE PATTERNS (closed-bar only, scale-free: sizes are relative to that timeframe's own ATR)
  engulfing, rejection wick (pin bar), morning / evening star, piercing line / dark cloud cover,
  tweezer bottom / top, inside-bar breakout, momentum thrust (marobozu through the last few highs/lows)

CONTEXT CONFIRMATIONS (each counts once)
  mini BOS, volume spike, liquidity sweep, RSI reversal (hook out of OB/OS or divergence),
  MACD histogram turning with the trade, EMA-20 reclaim

Everything uses data <= bar i only.
"""
from __future__ import annotations

import numpy as np

from .indicators import ind
from .structure import BULL, BEAR, Bars, _pattern_from_ohlc

PATTERN_LABELS = {
    "engulfing": "engulfing", "rejection": "rejection wick", "star": "morning/evening star",
    "piercing": "piercing / dark-cloud", "tweezer": "tweezer", "inside_break": "inside-bar breakout",
    "thrust": "momentum thrust",
}


def _body(b: Bars, j: int) -> float:
    return abs(float(b.c[j] - b.o[j]))


def _rng(b: Bars, j: int) -> float:
    return max(float(b.h[j] - b.l[j]), 1e-12)


def _dir_candle(b: Bars, j: int, d: int) -> bool:
    return b.c[j] > b.o[j] if d == BULL else b.c[j] < b.o[j]


def candle_patterns(b: Bars, i: int, d: int) -> list[dict]:
    """Every confirmation-candle pattern present on closed bar i for direction d (strongest first)."""
    out: list[dict] = []
    if i < 3 or d == 0:
        return out
    a = float(b.atr[i]) or 1e-9
    bull = d == BULL
    side = "Bullish" if bull else "Bearish"

    # 1) engulfing / 2) rejection wick  (the original two - identical thresholds)
    base = _pattern_from_ohlc(b.o[i - 1], b.c[i - 1], b.o[i], b.h[i], b.l[i], b.c[i], a, d)
    if base:
        out.append({"type": base["type"], "family": "engulfing" if "engulf" in base["type"] else "rejection",
                    "label": base["label"], "strength": 3 if "engulf" in base["type"] else 2})

    # 3) morning / evening star: strong opposite candle, small indecision candle, strong reversal close
    #    beyond the midpoint of the first candle's body
    b0, b1_, b2 = _body(b, i - 2), _body(b, i - 1), _body(b, i)
    if (not _dir_candle(b, i - 2, d)) and b0 >= 0.6 * a and b1_ <= 0.4 * b0 and _dir_candle(b, i, d) and b2 >= 0.5 * a:
        mid = (b.o[i - 2] + b.c[i - 2]) / 2.0
        if (bull and b.c[i] > mid) or ((not bull) and b.c[i] < mid):
            out.append({"type": f"{side.lower()}_star", "family": "star",
                        "label": f"{'Morning' if bull else 'Evening'} star", "strength": 3})

    # 4) piercing line / dark cloud cover
    if (not _dir_candle(b, i - 1, d)) and _body(b, i - 1) >= 0.5 * a and _dir_candle(b, i, d) and b2 >= 0.5 * a:
        mid = (b.o[i - 1] + b.c[i - 1]) / 2.0
        if bull and b.o[i] <= b.c[i - 1] + 0.1 * a and mid < b.c[i] < b.o[i - 1]:
            out.append({"type": "bullish_piercing_line", "family": "piercing", "label": "Piercing line", "strength": 2})
        if (not bull) and b.o[i] >= b.c[i - 1] - 0.1 * a and b.o[i - 1] < b.c[i] < mid:
            out.append({"type": "bearish_dark_cloud", "family": "piercing", "label": "Dark-cloud cover", "strength": 2})

    # 5) tweezer bottom / top: two bars sharing the same extreme, second one closing in our direction
    tol = 0.12 * a
    if _dir_candle(b, i, d) and (not _dir_candle(b, i - 1, d)) and _rng(b, i) >= 0.5 * a:
        if bull and abs(b.l[i] - b.l[i - 1]) <= tol:
            out.append({"type": "bullish_tweezer_bottom", "family": "tweezer", "label": "Tweezer bottom", "strength": 2})
        if (not bull) and abs(b.h[i] - b.h[i - 1]) <= tol:
            out.append({"type": "bearish_tweezer_top", "family": "tweezer", "label": "Tweezer top", "strength": 2})

    # 6) inside-bar breakout: bar i-1 sits inside bar i-2 ("mother"), bar i closes through the mother's extreme
    if b.h[i - 1] <= b.h[i - 2] and b.l[i - 1] >= b.l[i - 2] and _rng(b, i - 2) >= 0.6 * a:
        if (bull and b.c[i] > b.h[i - 2]) or ((not bull) and b.c[i] < b.l[i - 2]):
            out.append({"type": f"{side.lower()}_inside_break", "family": "inside_break",
                        "label": f"{side} inside-bar breakout", "strength": 2})

    # 7) momentum thrust: big, clean body closing through the last 3 bars' extreme
    if _dir_candle(b, i, d) and b2 >= 1.1 * a and b2 / _rng(b, i) >= 0.7:
        ext = float(b.h[i - 3:i].max()) if bull else float(b.l[i - 3:i].min())
        if (bull and b.c[i] > ext) or ((not bull) and b.c[i] < ext):
            out.append({"type": f"{side.lower()}_thrust", "family": "thrust",
                        "label": f"{side} momentum candle", "strength": 2})

    out.sort(key=lambda p: -p["strength"])
    return out


def rsi_reversal(b: Bars, i: int, d: int, look: int = 6) -> dict | None:
    """RSI hook out of an extreme (oversold for buys / overbought for sells) within the last few bars,
    or simple divergence between the last two swing extremes inside the last 25 bars."""
    if i < 30:
        return None
    r = ind(b).rsi14
    seg = r[i - look:i + 1]
    if d == BULL and seg.min() <= 35 and r[i] > r[i - 1] and r[i] >= seg.min() + 4:
        return {"label": f"RSI hooking up from oversold ({seg.min():.0f})"}
    if d == BEAR and seg.max() >= 65 and r[i] < r[i - 1] and r[i] <= seg.max() - 4:
        return {"label": f"RSI turning down from overbought ({seg.max():.0f})"}
    # divergence: compare the lowest low in the last 8 bars with the lowest low of the 8-25 bar window
    lo_n = slice(i - 7, i + 1)
    lo_o = slice(i - 25, i - 7)
    if d == BULL:
        a_i = i - 7 + int(np.argmin(b.l[lo_n]))
        o_i = i - 25 + int(np.argmin(b.l[lo_o]))
        if b.l[a_i] < b.l[o_i] and r[a_i] > r[o_i] + 2 and r[i] > r[i - 1]:
            return {"label": "Bullish RSI divergence"}
    else:
        a_i = i - 7 + int(np.argmax(b.h[lo_n]))
        o_i = i - 25 + int(np.argmax(b.h[lo_o]))
        if b.h[a_i] > b.h[o_i] and r[a_i] < r[o_i] - 2 and r[i] < r[i - 1]:
            return {"label": "Bearish RSI divergence"}
    return None


def macd_turn(b: Bars, i: int, d: int) -> dict | None:
    """MACD histogram has turned toward the trade for 2 bars after being on the other side."""
    if i < 40:
        return None
    h = ind(b).macd[2]
    if d == BULL and h[i] > h[i - 1] > h[i - 2] and h[i - 2] < 0:
        return {"label": "MACD histogram turning up"}
    if d == BEAR and h[i] < h[i - 1] < h[i - 2] and h[i - 2] > 0:
        return {"label": "MACD histogram turning down"}
    return None


def ema_reclaim(b: Bars, i: int, d: int, look: int = 5) -> dict | None:
    """Price closed back through the 20 EMA after spending bars on the other side of it."""
    if i < 25:
        return None
    e = ind(b).ema20
    c = b.c
    if d == BULL and c[i] > e[i] and (c[i - look:i] < e[i - look:i]).any():
        return {"label": "Reclaimed the 20 EMA"}
    if d == BEAR and c[i] < e[i] and (c[i - look:i] > e[i - look:i]).any():
        return {"label": "Rejected back under the 20 EMA"}
    return None


def extension(b: Bars, i: int, d: int) -> dict:
    """How stretched the move into this entry already is (anti-chase gauge).
    rsi_dir: RSI oriented so that HIGH always means 'we are chasing' (buys use RSI, sells use 100-RSI)."""
    r = float(ind(b).rsi14[i])
    rsi_dir = r if d == BULL else 100.0 - r
    return {"rsi": r, "rsi_dir": rsi_dir}
