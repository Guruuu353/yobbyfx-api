"""Key reference levels for the 4H "map": previous day / week highs & lows,
daily & weekly opens, last Asia / London session ranges and round numbers."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import sessions
from .data_fetcher import drop_incomplete


def _row(key: str, label: str, price: float, kind: str, ts: int | None = None) -> dict:
    return {"key": key, "label": label, "price": float(price), "kind": kind, "ts": ts}


def _session_range(df: pd.DataFrame, name: str) -> tuple[float, float, int, bool] | None:
    """High/low of the most recent (possibly still running) run of `name` session bars."""
    if df.empty:
        return None
    mask = sessions.active_masks(df.index)[name]
    if not mask.any():
        return None
    last = np.nonzero(mask)[0][-1]
    first = last
    while first > 0 and mask[first - 1]:
        first -= 1
    seg = df.iloc[first:last + 1]
    forming = last == len(df) - 1
    return float(seg["High"].max()), float(seg["Low"].min()), int(seg.index[0].timestamp()), forming


def round_numbers(price: float) -> tuple[float, float]:
    """Nearest 'major' round numbers below / above price (e.g. 3300/3400 for gold)."""
    if price <= 0:
        return price, price
    mag = 10 ** math.floor(math.log10(price))
    step = mag / 10 if mag >= 1000 else mag / 10  # 100 for 1000-9999, 0.1 for FX ~1.x, 10 for ~150 (JPY)
    lo = math.floor(price / step) * step
    return lo, lo + step


def key_levels(d1: pd.DataFrame | None, w1: pd.DataFrame | None, h1: pd.DataFrame | None, price: float) -> list[dict]:
    lv: list[dict] = []
    try:
        if d1 is not None and len(d1) >= 2:
            done = drop_incomplete(d1, "1d")
            if len(done):
                p = done.iloc[-1]
                ts = int(done.index[-1].timestamp())
                lv += [_row("PDH", "PDH", p["High"], "daily", ts), _row("PDL", "PDL", p["Low"], "daily", ts)]
            if len(d1) > len(done):
                f = d1.iloc[-1]
                lv.append(_row("DO", "Day open", f["Open"], "open", int(d1.index[-1].timestamp())))
    except Exception:  # noqa: BLE001
        pass
    try:
        if w1 is not None and len(w1) >= 2:
            done = drop_incomplete(w1, "1w")
            if len(done):
                p = done.iloc[-1]
                ts = int(done.index[-1].timestamp())
                lv += [_row("PWH", "PWH", p["High"], "weekly", ts), _row("PWL", "PWL", p["Low"], "weekly", ts)]
            if len(w1) > len(done):
                lv.append(_row("WO", "Week open", w1.iloc[-1]["Open"], "open", int(w1.index[-1].timestamp())))
    except Exception:  # noqa: BLE001
        pass
    try:
        if h1 is not None and len(h1):
            for name, tag in (("asia", "Asia"), ("london", "London")):
                r = _session_range(h1, name)
                if r:
                    hi, lo, ts, forming = r
                    sfx = "" if not forming else "*"
                    lv += [_row(f"{tag[:3].upper()}H", f"{tag} H{sfx}", hi, "session", ts),
                           _row(f"{tag[:3].upper()}L", f"{tag} L{sfx}", lo, "session", ts)]
    except Exception:  # noqa: BLE001
        pass
    lo, hi = round_numbers(price)
    lv += [_row("RN-", f"{lo:g}", lo, "round"), _row("RN+", f"{hi:g}", hi, "round")]
    return lv
