"""AMD / Power-of-3: Accumulation -> Manipulation -> Distribution, on ANY timeframe.

  Accumulation  a tight consolidation (the range): few ATRs tall, price going nowhere (low efficiency)
  Manipulation  a raid on one side of that range - the wick runs beyond the high (or low) to take the stops resting
                there, then price closes BACK INSIDE (a Judas swing / stop hunt). It is a trap: the real move is the other way.
  Distribution  displacement out of the range the other way: a decisive candle through the range midpoint
                (or all the way through the opposite side)

`amd_state(bars, i)` reads only bars up to and including i (no look-ahead), so the live engine, the chart and the backtest
agree. It returns the newest sequence that is still fresh, or phase "none".
"""
from __future__ import annotations

import numpy as np

MAX_AGE = 45              # bars after the last event during which the read is still actionable
WINDOWS = (12, 18, 26, 38, 55)
MAX_RANGE_ATR = 3.2       # range height must be <= this many ATRs
MAX_ER = 0.33             # efficiency ratio inside the range (low = going nowhere)
SWEEP_MIN_ATR = 0.08      # the raid must poke at least this far beyond the range edge
RECLAIM_BARS = 3          # ...and close back inside within this many bars


def _find_range(h, l, c, atr, i: int, lookback: int):
    """Newest consolidation window ending somewhere in [i-lookback, i]. Returns (start, end, top, bot) or None."""
    best = None
    for e in range(i, max(i - lookback, 20), -1):
        a = float(atr[e]) or 1e-9
        for L in WINDOWS[::-1]:                       # longest first: prefer the more significant range
            s = e - L + 1
            if s < 0:
                continue
            top, bot = float(h[s:e + 1].max()), float(l[s:e + 1].min())
            if top - bot > MAX_RANGE_ATR * a:
                continue
            path = float(np.abs(np.diff(c[s:e + 1])).sum()) or 1e-9
            er = abs(float(c[e] - c[s])) / path
            if er > MAX_ER:
                continue
            # the edges must have been respected: at least two bars poking near the top and two near the bottom
            tol = 0.25 * (top - bot)
            if (h[s:e + 1] >= top - tol).sum() < 2 or (l[s:e + 1] <= bot + tol).sum() < 2:
                continue
            best = (s, e, top, bot)
            break
        if best:
            return best
    return None


def amd_state(b, i: int, lookback: int = 70) -> dict:
    """b: structure.Bars. i: index of the last closed bar to use."""
    none = {"phase": "none", "dir": 0}
    if i < 40 or i >= b.n:
        return none
    h, l, c, o, atr = b.h, b.l, b.c, b.o, b.atr
    rng = _find_range(h, l, c, atr, i, lookback)
    if rng is None:
        return none
    s, e, top, bot = rng
    a = float(atr[e]) or 1e-9
    mid = (top + bot) / 2.0
    base = {"start_ts": int(b.t[s]), "end_ts": int(b.t[e]), "top": top, "bottom": bot, "dir": 0}
    if e >= i - 1 and i - e < 2:                      # still inside it (or it just ended): accumulating
        return {**base, "phase": "accumulation", "age": i - e, "note": "Accumulation: price is coiling in a range - a raid of one side usually comes next"}
    # ---- manipulation: first bar after the range that raids an edge and closes back inside within RECLAIM_BARS
    sweep = None
    for k in range(e + 1, i + 1):
        up = h[k] > top + SWEEP_MIN_ATR * a
        dn = l[k] < bot - SWEEP_MIN_ATR * a
        if not (up or dn):
            if c[k] > top + 0.3 * a or c[k] < bot - 0.3 * a:      # closed decisively outside with no raid: a breakout, not a trap
                if sweep is None:
                    return {**base, "phase": "none", "dir": 0, "note": "Range broke out cleanly"}
            continue
        if up and dn:                                  # both sides in one bar: indecisive
            continue
        side = "high" if up else "low"
        ext = float(h[k] if up else l[k])
        reclaim = None
        for m in range(k, min(k + RECLAIM_BARS + 1, i + 1)):
            if (c[m] < top) if up else (c[m] > bot):
                reclaim = m
                break
            ext = max(ext, float(h[m])) if up else min(ext, float(l[m]))
        if reclaim is None:
            if k + RECLAIM_BARS <= i:                  # stayed outside: continuation breakout, not manipulation
                return {**base, "phase": "none", "dir": 0, "note": "Range broke out and held"}
            # still outside, waiting to see whether it reclaims
            return {**base, "phase": "manipulation_pending", "dir": 0, "sweep": {"side": side, "ts": int(b.t[k]), "extreme": ext},
                    "age": i - k, "note": f"Raid of the range {side} in progress - waiting for the close back inside"}
        sweep = {"side": side, "ts": int(b.t[k]), "extreme": ext, "reclaim_ts": int(b.t[reclaim]), "idx": k, "reclaim_idx": reclaim}
        break
    if sweep is None:
        return {**base, "phase": "none", "dir": 0}
    d = -1 if sweep["side"] == "high" else 1             # a raid of the highs is a bearish trap; of the lows a bullish one
    r = sweep["reclaim_idx"]
    if i - r > MAX_AGE:
        return {**base, "phase": "none", "dir": 0}
    # invalidation: price made a new extreme beyond the raid after reclaiming (the trap did not hold)
    after = slice(r + 1, i + 1)
    if r < i and ((h[after].max() > sweep["extreme"]) if d == -1 else (l[after].min() < sweep["extreme"])):
        return {**base, "phase": "none", "dir": 0, "note": "Raid extreme was taken out again - trap failed"}
    # ---- distribution: displacement through the midpoint (or the far edge)
    disp = None
    for m in range(r, i + 1):
        body = abs(float(c[m] - o[m]))
        through_mid = (c[m] < mid) if d == -1 else (c[m] > mid)
        through_far = (c[m] < bot) if d == -1 else (c[m] > top)
        ok_dir = (c[m] < o[m]) if d == -1 else (c[m] > o[m])
        if ok_dir and ((through_mid and body >= 0.8 * a) or through_far):
            disp = {"ts": int(b.t[m]), "idx": m, "full": bool(through_far)}
            break
    out = {**base, "dir": d, "sweep": {k: v for k, v in sweep.items() if not k.endswith("idx")}, "age": i - (disp["idx"] if disp else r)}
    if disp:
        out.update(phase="distribution", disp={"ts": disp["ts"], "full": disp["full"]},
                   note=f"Distribution: {'sell' if d == -1 else 'buy'}-side move after the {sweep['side']} of the range was raided")
    else:
        out.update(phase="manipulation",
                   note=f"Manipulation done: range {sweep['side']} raided and reclaimed - expect displacement {'down' if d == -1 else 'up'}")
    return out
